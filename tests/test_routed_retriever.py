from __future__ import annotations

import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from retriever import RetrievedChunk  # noqa: E402
except ModuleNotFoundError as exc:  # runtime de validation léger, sans CUDA
    if exc.name != "torch":
        raise
    sys.modules.pop("retriever", None)
    retriever_stub = types.ModuleType("retriever")

    @dataclass
    class RetrievedChunk:  # type: ignore[no-redef]
        chunk_id: str
        url: str
        title: str
        section: str
        doc_type: str
        category: str
        date: str | None
        text: str
        parent_text: str
        dense_score: float
        bm25_score: float
        rerank_score: float
        coverage_reserved: bool = False

    retriever_stub.RetrievedChunk = RetrievedChunk
    retriever_stub.HybridRetriever = object
    sys.modules["retriever"] = retriever_stub
from routed_retriever import RoutedRetriever  # noqa: E402


def _chunk(source: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{source}-1",
        url=f"https://example.test/{source}",
        title=source,
        section=source,
        doc_type="test",
        category=source,
        date=None,
        text=f"preuve {source}",
        parent_text=f"preuve {source}",
        dense_score=score,
        bm25_score=score,
        rerank_score=score,
    )


class _DummyRetriever:
    def __init__(self, source: str, score: float) -> None:
        self.source = source
        self.score = score
        self.calls: list[tuple[str, int | None, str | None]] = []
        self.last_timings = {"total_seconds": 0.01}
        self.last_debug_rows = [{"title": source}]

    def retrieve(
        self,
        query: str,
        *,
        max_results: int | None = None,
        category_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append((query, max_results, category_filter))
        return [_chunk(self.source, self.score)]


class RoutedRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.institutional = _DummyRetriever("institutional", 0.80)
        self.economic = _DummyRetriever("economic", 0.95)
        self.facebook = _DummyRetriever("facebook", 0.85)
        self.retriever = RoutedRetriever(
            self.institutional,  # type: ignore[arg-type]
            self.facebook,  # type: ignore[arg-type]
            economic=self.economic,  # type: ignore[arg-type]
            max_combined_results=5,
        )

    def test_ambiguous_route_searches_all_available_corpora(self) -> None:
        chunks = self.retriever.retrieve(
            "Quel est le palmarès du challenge NOVA sur le campus ?",
            max_results=5,
        )
        self.assertEqual(self.retriever.last_decision.route, "all")
        self.assertEqual(len(self.institutional.calls), 1)
        self.assertEqual(len(self.economic.calls), 1)
        self.assertEqual(len(self.facebook.calls), 1)
        self.assertEqual({chunk.category for chunk in chunks}, {
            "institutional", "economic", "facebook",
        })
        self.assertEqual(
            self.retriever.last_timings["source_index"],
            "institutional+economic+facebook",
        )

    def test_ensea_event_both_route_does_not_query_economic_index(self) -> None:
        self.retriever.retrieve(
            "Qui a remporté le dernier concours scientifique organisé à l'ENSEA en 2026 ?",
            max_results=5,
        )
        self.assertEqual(self.retriever.last_decision.route, "both")
        self.assertEqual(len(self.institutional.calls), 1)
        self.assertEqual(len(self.facebook.calls), 1)
        self.assertEqual(self.economic.calls, [])

    def test_statistical_route_keeps_fast_economic_path(self) -> None:
        chunks = self.retriever.retrieve(
            "Quel était le taux de pauvreté en milieu rural ?",
            max_results=5,
        )
        self.assertEqual(self.retriever.last_decision.route, "economic")
        self.assertEqual(self.institutional.calls, [])
        self.assertEqual(len(self.economic.calls), 1)
        self.assertEqual(self.facebook.calls, [])
        self.assertEqual([chunk.category for chunk in chunks], ["economic"])


if __name__ == "__main__":
    unittest.main()
