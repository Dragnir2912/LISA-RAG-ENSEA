"""Façade haut niveau : assemble embeddings + vectorstore + retriever + generator
en une API simple, utilisée par le notebook et par evaluation.py.
"""

from __future__ import annotations

import logging
import time

from config import Config, load_config
from conversation import (
    contextualize_question,
    generation_history,
    is_collection_request,
    is_exhaustive_request,
)
from embeddings import EmbeddingModel
from generator import OllamaGenerator, RAGResponse, extract_citations
from observability import ObservabilityRecorder, new_run_id
from retriever import HybridRetriever, Reranker
from routed_retriever import RoutedRetriever
from vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Point d'entrée unique du système RAG ENSEA.

    Usage :
        pipeline = RAGPipeline.from_config()
        response = pipeline.ask("Quelles sont les filières de l'ENSEA ?")
        print(response.answer)
        for c in response.citations:
            print(c.url, c.rerank_score)
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.embedding_model = EmbeddingModel(config.embeddings)
        self.store = VectorStoreManager(config)
        self.store.load_sparse()
        self.reranker = Reranker(config.reranker.model_name, config.reranker.device)
        self.institutional_retriever = HybridRetriever(
            config, self.embedding_model, self.store, self.reranker,
        )
        self.economic_store = None
        self.economic_retriever = None
        economic_cfg = config.local_documents
        if economic_cfg.retrieval_enabled:
            economic_config = config.model_copy(deep=True)
            economic_config.vectorstore.uri = economic_cfg.vectorstore_uri
            economic_config.vectorstore.table_name = economic_cfg.vectorstore_table
            economic_config.vectorstore.bm25_index_path = economic_cfg.bm25_index_path
            economic_store = VectorStoreManager(economic_config)
            economic_sparse_loaded = economic_store.load_sparse()
            if economic_store.dense.count() > 0 and economic_sparse_loaded:
                self.economic_store = economic_store
                self.economic_retriever = HybridRetriever(
                    economic_config,
                    self.embedding_model,
                    economic_store,
                    self.reranker,
                )
                logger.info(
                    "Index économique séparé chargé : %d chunks.",
                    economic_store.dense.count(),
                )
            else:
                logger.warning(
                    "Routage économique activé mais index indisponible ; "
                    "les questions économiques déclencheront l'abstention."
                )
        self.facebook_store = None
        self.facebook_retriever = None
        facebook_cfg = config.facebook_scraping
        if facebook_cfg.retrieval_enabled:
            routed_config = config.model_copy(deep=True)
            routed_config.vectorstore.uri = facebook_cfg.vectorstore_candidate_uri
            routed_config.vectorstore.table_name = facebook_cfg.vectorstore_candidate_table
            routed_config.vectorstore.bm25_index_path = facebook_cfg.bm25_candidate_path
            candidate_store = VectorStoreManager(routed_config)
            sparse_loaded = candidate_store.load_sparse()
            if candidate_store.dense.count() > 0 and sparse_loaded:
                self.facebook_store = candidate_store
                self.facebook_retriever = HybridRetriever(
                    routed_config, self.embedding_model, candidate_store, self.reranker,
                )
                logger.info(
                    "Index Facebook séparé chargé : %d chunks.",
                    candidate_store.dense.count(),
                )
            else:
                logger.warning(
                    "Routage Facebook activé mais index candidat indisponible ; "
                    "les questions Facebook déclencheront l'abstention."
                )
        self.retriever = RoutedRetriever(
            self.institutional_retriever,
            self.facebook_retriever,
            economic=self.economic_retriever,
            max_combined_results=facebook_cfg.max_combined_results,
            institutional_bias_both=facebook_cfg.institutional_bias_both,
            facebook_page_aliases=facebook_cfg.page_aliases,
        )
        self.generator = OllamaGenerator(
            config.generation, tokenizer=self.embedding_model.tokenizer,
        )
        self.observability = ObservabilityRecorder(config.observability)
        self.last_chunks = []
        self.last_response: RAGResponse | None = None
        self.last_retrieval_query = ""
        self.last_contextualized = False
        self.last_expanded_retrieval = False

    @classmethod
    def from_config(cls, path: str | None = None) -> "RAGPipeline":
        config = load_config(path) if path else load_config()
        return cls(config)

    def close(self) -> None:
        """Libère explicitement les modèles de retrieval lors d'un rechargement."""
        try:
            self.reranker.unload()
        finally:
            self.embedding_model.unload()

    def _prepare_request(
        self,
        question: str,
        conversation_history: list[dict] | None,
    ) -> tuple[str, list[dict], int | None]:
        retrieval_query, contextualized = contextualize_question(
            question, conversation_history,
        )
        explicit_exhaustive = is_exhaustive_request(question)
        expanded = explicit_exhaustive or is_collection_request(question)
        self.last_retrieval_query = retrieval_query
        self.last_contextualized = contextualized
        self.last_expanded_retrieval = expanded
        # Le générateur reçoit toujours une mémoire récente bornée. L'historique
        # n'est pas une source factuelle, mais il doit rester disponible pour
        # préserver le ton, les contraintes et les références implicites d'une
        # vraie conversation. Seule la requête de retrieval est contextualisée
        # de manière sélective afin de ne pas polluer la recherche documentaire.
        relevant_history = generation_history(
            question,
            conversation_history,
            contextualized=contextualized,
        )
        max_results = (
            self.config.retrieval.exhaustive_max_results if expanded else None
        )
        return retrieval_query, relevant_history, max_results

    def ask(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
        *,
        thinking: bool | None = None,
    ) -> RAGResponse:
        started = time.perf_counter()
        run_id = new_run_id("ask")
        chunks = []
        try:
            retrieval_query, relevant_history, max_results = self._prepare_request(
                question, conversation_history,
            )
            chunks = self.retriever.retrieve(
                retrieval_query, max_results=max_results,
            )
            response = self.generator.generate(
                question,
                chunks,
                conversation_history=relevant_history,
                thinking=thinking,
            )
            self.last_chunks = chunks
            self.last_response = response
            self.observability.record(
                event_type="rag",
                question=question,
                source="pipeline.ask",
                status="ok",
                chunks=chunks,
                response=response,
                retrieval_stats=self.retriever.last_timings,
                prompt_stats=self.generator.last_prompt_stats,
                generation_stats=self.generator.last_generation_stats,
                total_seconds=time.perf_counter() - started,
                run_id=run_id,
                metadata={
                    "route": self.retriever.last_decision.route,
                    "route_confidence": self.retriever.last_decision.confidence,
                    "route_reason": self.retriever.last_decision.reason,
                    "contextualized": self.last_contextualized,
                    "expanded_retrieval": self.last_expanded_retrieval,
                    "thinking_enabled": (
                        self.config.generation.thinking
                        if thinking is None else bool(thinking)
                    ),
                },
            )
            return response
        except Exception as error:
            self.observability.record_error(
                question, "pipeline.ask", error, run_id=run_id,
            )
            raise

    def ask_stream(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
        *,
        thinking: bool | None = None,
        event_mode: bool = False,
    ):
        started = time.perf_counter()
        run_id = new_run_id("stream")
        chunks = []
        answer_parts: list[str] = []
        try:
            retrieval_query, relevant_history, max_results = self._prepare_request(
                question, conversation_history,
            )
            chunks = self.retriever.retrieve(
                retrieval_query, max_results=max_results,
            )
            for part in self.generator.generate_stream(
                question,
                chunks,
                conversation_history=relevant_history,
                thinking=thinking,
                event_mode=event_mode,
            ):
                if event_mode:
                    if part.get("type") == "token":
                        answer_parts.append(str(part.get("content", "")))
                else:
                    answer_parts.append(part)
                yield part
            answer = "".join(answer_parts)
            response = RAGResponse(
                answer=answer,
                citations=extract_citations(
                    answer,
                    chunks,
                    prompt_passages=self.generator.prompt_builder.last_passages,
                ),
            )
            self.last_chunks = chunks
            self.last_response = response
            self.observability.record(
                event_type="rag",
                question=question,
                source="pipeline.ask_stream",
                status="ok",
                chunks=chunks,
                response=response,
                retrieval_stats=self.retriever.last_timings,
                prompt_stats=self.generator.last_prompt_stats,
                generation_stats=self.generator.last_generation_stats,
                total_seconds=time.perf_counter() - started,
                run_id=run_id,
                metadata={
                    "route": self.retriever.last_decision.route,
                    "route_confidence": self.retriever.last_decision.confidence,
                    "route_reason": self.retriever.last_decision.reason,
                    "contextualized": self.last_contextualized,
                    "expanded_retrieval": self.last_expanded_retrieval,
                    "thinking_enabled": (
                        self.config.generation.thinking
                        if thinking is None else bool(thinking)
                    ),
                },
            )
        except Exception as error:
            self.observability.record_error(
                question, "pipeline.ask_stream", error, run_id=run_id,
            )
            raise
