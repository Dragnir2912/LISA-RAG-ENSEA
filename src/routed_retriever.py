"""Façade de retrieval routé entre index ENSEA, économique et Facebook."""

from __future__ import annotations

import time
from typing import Any

from retriever import HybridRetriever, RetrievedChunk
from source_router import (
    RouteDecision,
    resolve_facebook_page_scope,
    route_question,
)


class RoutedRetriever:
    """Interroge les corpus indiqués sans transformer le routeur en filtre dur."""

    def __init__(
        self,
        institutional: HybridRetriever,
        facebook: HybridRetriever | None,
        economic: HybridRetriever | None = None,
        *,
        max_combined_results: int = 5,
        institutional_bias_both: float = 0.02,
        facebook_page_aliases: dict[str, list[str]] | None = None,
    ) -> None:
        self.institutional = institutional
        self.facebook = facebook
        self.economic = economic
        self.max_combined_results = max_combined_results
        self.institutional_bias_both = institutional_bias_both
        self.facebook_page_aliases = facebook_page_aliases or {}
        self.last_decision = RouteDecision("institutional", 1.0, "initialisation")
        self.last_timings: dict[str, Any] = {}
        self.last_debug_rows: list[dict[str, Any]] = []

    def _single_source(
        self,
        retriever: HybridRetriever,
        question: str,
        source_index: str,
        max_results: int | None = None,
        category_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        chunks = retriever.retrieve(
            question,
            max_results=max_results,
            category_filter=category_filter,
        )
        self.last_timings = dict(retriever.last_timings)
        self.last_timings["source_route"] = self.last_decision.route
        self.last_timings["route_confidence"] = self.last_decision.confidence
        self.last_timings["route_reason"] = self.last_decision.reason
        self.last_timings["source_index"] = source_index
        self.last_timings["source_scope"] = category_filter or ""
        self.last_debug_rows = [
            {
                **row,
                "source_index": source_index,
                "source_scope": category_filter or "",
            }
            for row in retriever.last_debug_rows
        ]
        return chunks

    def _merge_sources(
        self,
        chunks_by_source: dict[str, list[RetrievedChunk]],
        max_results: int | None = None,
    ) -> list[RetrievedChunk]:
        limit = (
            max_results if max_results is not None
            else self.max_combined_results
        )
        if limit <= 0:
            return []

        source_biases = {
            "institutional": self.institutional_bias_both,
            "economic": 0.0,
            "facebook": 0.0,
        }
        source_priorities = {
            "institutional": 2,
            "economic": 1,
            "facebook": 0,
        }
        candidates_by_source: dict[
            str, list[tuple[float, int, RetrievedChunk]]
        ] = {}
        for source_index, source_chunks in chunks_by_source.items():
            candidates_by_source[source_index] = [
                (
                    chunk.rerank_score + source_biases.get(source_index, 0.0),
                    source_priorities.get(source_index, 0),
                    chunk,
                )
                for chunk in source_chunks
            ]
        candidates = [
            candidate
            for source_candidates in candidates_by_source.values()
            for candidate in source_candidates
        ]
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

        selected: list[RetrievedChunk] = []
        seen_urls: set[str] = set()

        # Les scores de reranking de corpus distincts ne sont pas toujours
        # parfaitement calibrés entre eux. Lorsque la capacité le permet, une
        # place est réservée à chaque corpus ayant fourni un candidat. Ce filet
        # de rappel empêche une mauvaise décision de routage de masquer la bonne
        # source avant la génération.
        non_empty_sources = [
            source_index
            for source_index, source_candidates in candidates_by_source.items()
            if source_candidates
        ]
        if limit >= len(non_empty_sources):
            for source_index in non_empty_sources:
                source_candidates = candidates_by_source[source_index]
                for _, _, chunk in sorted(
                    source_candidates,
                    key=lambda item: (item[0], item[1]),
                    reverse=True,
                ):
                    if chunk.url in seen_urls:
                        continue
                    selected.append(chunk)
                    seen_urls.add(chunk.url)
                    break

        for _, _, chunk in candidates:
            if len(selected) >= limit:
                break
            if chunk.url in seen_urls:
                continue
            selected.append(chunk)
            seen_urls.add(chunk.url)

        adjusted_scores = {
            id(chunk): score for score, _priority, chunk in candidates
        }
        selected.sort(
            key=lambda chunk: adjusted_scores.get(id(chunk), chunk.rerank_score),
            reverse=True,
        )
        return selected

    def _mixed_sources(
        self,
        question: str,
        source_indexes: tuple[str, ...],
        *,
        max_results: int | None,
        facebook_category: str | None,
        started: float,
    ) -> list[RetrievedChunk]:
        retrievers = {
            "institutional": self.institutional,
            "economic": self.economic,
            "facebook": self.facebook,
        }
        chunks_by_source: dict[str, list[RetrievedChunk]] = {}
        timings_by_source: dict[str, dict[str, Any]] = {}
        debug_rows: list[dict[str, Any]] = []

        # Les trois retrievers partagent le même embedder et le même reranker
        # GPU. Les appels restent séquentiels ici pour éviter une concurrence
        # CUDA et une hausse de VRAM sur les machines 8 Go. L'optimisation
        # suivante consistera à mutualiser l'embedding de requête.
        for source_index in source_indexes:
            retriever = retrievers[source_index]
            if retriever is None:
                continue
            category_filter = (
                facebook_category if source_index == "facebook" else None
            )
            source_chunks = retriever.retrieve(
                question,
                max_results=max_results,
                category_filter=category_filter,
            )
            chunks_by_source[source_index] = source_chunks
            timings_by_source[source_index] = dict(retriever.last_timings)
            debug_rows.extend(
                {
                    **row,
                    "source_index": source_index,
                    "source_scope": category_filter or "",
                }
                for row in retriever.last_debug_rows
            )

        chunks = self._merge_sources(
            chunks_by_source, max_results=max_results,
        )
        available_sources = list(chunks_by_source)
        self.last_timings = {
            "source_route": self.last_decision.route,
            "route_confidence": self.last_decision.confidence,
            "route_reason": self.last_decision.reason,
            "source_index": "+".join(available_sources) or "unavailable",
            "source_scope": facebook_category or "",
            "total_seconds": time.perf_counter() - started,
            "returned_chunks": len(chunks),
        }
        for source_index in source_indexes:
            source_timings = timings_by_source.get(source_index, {})
            self.last_timings[f"{source_index}_seconds"] = (
                source_timings.get("total_seconds", 0.0)
            )
            self.last_timings[f"{source_index}_results"] = len(
                chunks_by_source.get(source_index, [])
            )
        self.last_debug_rows = debug_rows
        return chunks

    def retrieve(
        self, question: str, *, max_results: int | None = None,
    ) -> list[RetrievedChunk]:
        started = time.perf_counter()
        self.last_decision = route_question(
            question, page_aliases=self.facebook_page_aliases,
        )
        facebook_page_key = resolve_facebook_page_scope(
            question, self.facebook_page_aliases,
        )
        facebook_category = (
            f"facebook/{facebook_page_key}" if facebook_page_key else None
        )

        if self.last_decision.route == "institutional":
            return self._single_source(
                self.institutional, question, "institutional", max_results,
            )

        if self.last_decision.route == "facebook":
            if self.facebook is None:
                self.last_timings = {
                    "source_route": "facebook",
                    "route_confidence": self.last_decision.confidence,
                    "route_reason": self.last_decision.reason,
                    "source_index": "unavailable",
                    "source_scope": facebook_category or "",
                    "total_seconds": time.perf_counter() - started,
                    "returned_chunks": 0,
                }
                self.last_debug_rows = []
                return []
            return self._single_source(
                self.facebook,
                question,
                "facebook",
                max_results,
                category_filter=facebook_category,
            )

        if self.last_decision.route == "economic":
            if self.economic is None:
                self.last_timings = {
                    "source_route": "economic",
                    "route_confidence": self.last_decision.confidence,
                    "route_reason": self.last_decision.reason,
                    "source_index": "unavailable",
                    "total_seconds": time.perf_counter() - started,
                    "returned_chunks": 0,
                }
                self.last_debug_rows = []
                return []
            return self._single_source(
                self.economic, question, "economic", max_results,
            )

        if self.last_decision.route == "both":
            source_indexes = ("institutional", "facebook")
        else:
            source_indexes = ("institutional", "economic", "facebook")
        return self._mixed_sources(
            question,
            source_indexes,
            max_results=max_results,
            facebook_category=facebook_category,
            started=started,
        )
