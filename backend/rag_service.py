"""Chargement unique du pipeline RAG et adaptation en événements de streaming."""

from __future__ import annotations

import os
import sys
import threading
import time
import json
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

from filelock import FileLock


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Le pipeline utilise exclusivement PyTorch. CUDA reste optionnel. Sur certaines installations
# Windows, Transformers détecte Keras 3 et tente inutilement d'importer son
# backend TensorFlow avant même le chargement du modèle PyTorch.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from config import load_config  # noqa: E402
from conversation import contextualize_question, direct_conversation_reply  # noqa: E402
from source_router import route_question  # noqa: E402

if TYPE_CHECKING:
    from rag_pipeline import RAGPipeline


DEFAULT_GENERATION_MODEL = load_config(
    PROJECT_ROOT / "config.yaml"
).generation.model_name


class RAGService:
    """Singleton applicatif : un seul pipeline et une seule génération GPU à la fois."""

    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._loading = False
        self._last_error: str | None = None
        self._loaded_at: float | None = None
        self._settings_provider: Callable[[], dict[str, Any]] | None = None
        self._effective_profile = "generation_priority_8gb"
        self._maintenance = False
        self._loaded_configuration_revision = ""
        lock_path = PROJECT_ROOT / "data" / "app" / "gpu-engine.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._process_lock = FileLock(str(lock_path))

    def configure(self, settings_provider: Callable[[], dict[str, Any]]) -> None:
        self._settings_provider = settings_provider

    def _load_effective_config(self):
        config = load_config(PROJECT_ROOT / "config.yaml")
        settings = self._settings_provider() if self._settings_provider else {}
        generation_model = settings.get("generation_model")
        if isinstance(generation_model, str) and generation_model.strip():
            allowed = {config.generation.model_name, *config.generation.alternatives}
            if generation_model in allowed:
                config.generation.model_name = generation_model
        profile = str(
            settings.get("performance_profile")
            or config.runtime.performance_profile
        )
        self._effective_profile = profile
        force_cpu = os.getenv("RAG_FORCE_CPU", "0").strip().casefold() in {
            "1", "true", "yes", "on",
        }
        if force_cpu or profile == "cpu_only":
            self._effective_profile = "cpu_only"
            config.embeddings.device = "cpu"
            config.reranker.device = "cpu"
            config.embeddings.precision = "fp32"
            config.reranker.precision = "fp32"
            config.generation.num_gpu = 0
        elif profile == "generation_priority_8gb":
            config.embeddings.device = config.runtime.online_embeddings_device
            config.reranker.device = config.runtime.online_reranker_device
            config.embeddings.precision = (
                "fp16" if config.embeddings.device == "cuda" else "fp32"
            )
            config.reranker.precision = (
                "fp16" if config.reranker.device == "cuda" else "fp32"
            )
        raw_num_gpu = os.getenv("RAG_OLLAMA_NUM_GPU", "").strip()
        if raw_num_gpu:
            try:
                config.generation.num_gpu = int(raw_num_gpu)
            except ValueError:
                pass
        try:
            import torch

            torch.set_num_threads(max(1, config.runtime.torch_cpu_threads))
        except (ImportError, RuntimeError):
            pass
        return config

    def _configuration_revision(self) -> str:
        values = self._settings_provider() if self._settings_provider else {}
        selected = {
            key: values.get(key)
            for key in ("generation_model", "performance_profile", "pipeline_revision")
        }
        return json.dumps(selected, sort_keys=True, ensure_ascii=True)

    def get_pipeline(self) -> Any:
        revision = self._configuration_revision()
        if self._pipeline is not None and revision == self._loaded_configuration_revision:
            return self._pipeline
        with self._load_lock:
            revision = self._configuration_revision()
            if self._pipeline is not None and revision == self._loaded_configuration_revision:
                return self._pipeline
            if self._pipeline is not None:
                self._unload_pipeline()
            self._loading = True
            self._last_error = None
            try:
                config = self._load_effective_config()
                # PyTorch, LanceDB et les modèles ne sont importés qu'au premier
                # message documentaire. Authentification et administration restent
                # légères au démarrage de l'API.
                from rag_pipeline import RAGPipeline

                self._pipeline = RAGPipeline(config)
                self._loaded_at = time.time()
                self._loaded_configuration_revision = revision
                return self._pipeline
            except Exception as error:
                self._last_error = f"{type(error).__name__}: {error}"
                raise
            finally:
                self._loading = False

    def status(self) -> dict[str, Any]:
        pipeline = self._pipeline
        return {
            "loaded": pipeline is not None,
            "loading": self._loading,
            "busy": self._generation_lock.locked(),
            "maintenance": self._maintenance,
            "error": self._last_error,
            "loaded_at": self._loaded_at,
            "institutional_chunks": pipeline.store.dense.count() if pipeline else None,
            "economic_chunks": (
                pipeline.economic_store.dense.count()
                if pipeline and pipeline.economic_store else None
            ),
            "facebook_chunks": (
                pipeline.facebook_store.dense.count()
                if pipeline and pipeline.facebook_store else None
            ),
            "model": (
                pipeline.config.generation.model_name
                if pipeline else DEFAULT_GENERATION_MODEL
            ),
            "performance_profile": self._effective_profile,
            "embedding_device": pipeline.embedding_model.device if pipeline else None,
            "reranker_device": pipeline.reranker.device if pipeline else None,
        }

    def reload(self) -> dict[str, Any]:
        """Recharge proprement modèles et paramètres entre deux requêtes."""
        with self._generation_lock:
            old_pipeline = self._pipeline
            self._pipeline = None
            if old_pipeline is not None:
                old_pipeline.close()
            self._loaded_at = None
            return self.status()

    def _unload_pipeline(self) -> None:
        old_pipeline = self._pipeline
        self._pipeline = None
        if old_pipeline is not None:
            old_pipeline.close()
        self._loaded_at = None

    @contextmanager
    def maintenance(self) -> Iterator[None]:
        """Réserve la machine pour une collecte ou reconstruction d'index."""
        with self._generation_lock, self._process_lock:
            self._maintenance = True
            try:
                model_name = (
                    self._pipeline.config.generation.model_name
                    if self._pipeline else DEFAULT_GENERATION_MODEL
                )
                self._unload_pipeline()
                try:
                    import ollama

                    ollama.generate(model=model_name, prompt="", keep_alive=0)
                except Exception:
                    # Ollama peut être arrêté pendant une tâche d'indexation.
                    pass
                yield
            finally:
                self._maintenance = False

    @contextmanager
    def exclusive_pipeline(self) -> Iterator[Any]:
        """Réserve le GPU et les index pour une génération ou une maintenance."""
        with self._generation_lock, self._process_lock:
            yield self.get_pipeline()

    @staticmethod
    def preview_route(question: str) -> dict[str, Any]:
        decision = route_question(question)
        return {
            "route": decision.route,
            "confidence": decision.confidence,
            "reason": decision.reason,
        }

    @staticmethod
    def is_direct_conversation_turn(message: str) -> bool:
        return direct_conversation_reply(message) is not None

    def stream(
        self,
        question: str,
        conversation_history: list[dict[str, Any]] | None = None,
        *,
        thinking: bool = False,
    ) -> Iterator[dict[str, Any]]:
        direct_reply = direct_conversation_reply(question)
        if direct_reply is not None:
            started = time.perf_counter()
            yield {
                "type": "route",
                "route": "conversation",
                "confidence": 1.0,
                "reason": "tour social sans recherche documentaire",
                "contextualized": bool(conversation_history),
            }
            yield {"type": "token", "content": direct_reply}
            yield {
                "type": "done",
                "answer": direct_reply,
                "route": "conversation",
                "route_reason": "tour social sans recherche documentaire",
                "citations": [],
                "reasoning": None,
                "metrics": {
                    "queue_seconds": 0.0,
                    "time_to_first_token_seconds": 0.0,
                    "retrieval_seconds": 0.0,
                    "generation_seconds": 0.0,
                    "total_seconds": time.perf_counter() - started,
                    "prompt_tokens": 0,
                    "generated_tokens": 0,
                    "tokens_per_second": 0.0,
                    "cache_hit": False,
                    "conversation_turns": len(conversation_history or []),
                    "contextualized": bool(conversation_history),
                    "expanded_retrieval": False,
                    "retrieved_chunks": 0,
                    "thinking_enabled": False,
                    "model_name": None,
                },
            }
            return
        preview_question, contextualized = contextualize_question(
            question, conversation_history,
        )
        preview = self.preview_route(preview_question)
        preview["contextualized"] = contextualized
        yield {"type": "route", **preview}
        queue_started = time.perf_counter()
        with self._generation_lock, self._process_lock:
            queue_seconds = time.perf_counter() - queue_started
            pipeline = self.get_pipeline()
            request_started = time.perf_counter()
            first_token_seconds: float | None = None
            for stream_event in pipeline.ask_stream(
                question,
                conversation_history=conversation_history,
                thinking=thinking,
                event_mode=True,
            ):
                if stream_event.get("type") == "reasoning":
                    yield {
                        "type": "reasoning",
                        "content": stream_event.get("content", ""),
                    }
                    continue
                if stream_event.get("type") != "token":
                    continue
                if first_token_seconds is None:
                    first_token_seconds = time.perf_counter() - request_started
                yield {
                    "type": "token",
                    "content": stream_event.get("content", ""),
                }

            response = pipeline.last_response
            if response is None:
                raise RuntimeError("Le pipeline n'a produit aucune réponse.")
            generation = pipeline.generator.last_generation_stats
            prompt = pipeline.generator.last_prompt_stats
            retrieval = pipeline.retriever.last_timings
            decision = pipeline.retriever.last_decision
            citations = [
                {
                    "index": citation.index,
                    "title": citation.title,
                    "section": citation.section,
                    "url": citation.url,
                    "passage": citation.passage,
                    "rerank_score": citation.rerank_score,
                }
                for citation in response.citations
            ]
            yield {
                "type": "done",
                "answer": response.answer,
                "route": decision.route,
                "route_reason": decision.reason,
                "citations": citations,
                "reasoning": (
                    dict(pipeline.generator.last_reasoning)
                    if thinking and pipeline.generator.last_reasoning
                    else None
                ),
                "metrics": {
                    "queue_seconds": queue_seconds,
                    "time_to_first_token_seconds": (
                        first_token_seconds
                        if first_token_seconds is not None
                        else time.perf_counter() - request_started
                    ),
                    "retrieval_seconds": retrieval.get("total_seconds", 0.0),
                    "generation_seconds": generation.get("total_seconds", 0.0),
                    "total_seconds": time.perf_counter() - request_started,
                    "prompt_tokens": prompt.get("prompt_tokens", 0),
                    "generated_tokens": generation.get("generated_tokens", 0),
                    "tokens_per_second": generation.get("tokens_per_second", 0.0),
                    "cache_hit": generation.get("cache_hit", False),
                    "ollama_num_ctx": prompt.get("ollama_num_ctx"),
                    "conversation_turns": prompt.get("conversation_turns", 0),
                    "membership_validation": prompt.get(
                        "membership_validation", False,
                    ),
                    "coverage_required_chunks": prompt.get(
                        "coverage_required_chunks", 0,
                    ),
                    "repeated_lines_removed": generation.get(
                        "repeated_lines_removed", 0,
                    ),
                    "repetition_guard_triggered": generation.get(
                        "repetition_guard_triggered", False,
                    ),
                    "contextualized": pipeline.last_contextualized,
                    "expanded_retrieval": pipeline.last_expanded_retrieval,
                    "retrieved_chunks": len(pipeline.last_chunks),
                    "thinking_enabled": bool(thinking),
                    "model_name": pipeline.config.generation.model_name,
                    "reasoning_seconds": generation.get("reasoning_seconds", 0.0),
                    "reasoning_characters": generation.get(
                        "reasoning_characters", 0,
                    ),
                    "fallback_answer_used": generation.get(
                        "fallback_answer_used", False,
                    ),
                    "reasoning_truncated": generation.get(
                        "reasoning_truncated", False,
                    ),
                },
            }


rag_service = RAGService()
