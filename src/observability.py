"""Observabilité locale et légère pour le pipeline RAG ENSEA.

Chaque requête produit une ligne JSON autonome. Ce format reste lisible avec
Pandas, ne nécessite aucun serveur externe et permet de diagnostiquer les
régressions de qualité ou de latence directement depuis un notebook.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from config import ObservabilityConfig


def new_run_id(prefix: str = "rag") -> str:
    """Identifiant court permettant de regrouper les événements d'un même run."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def _question_hash(question: str) -> str:
    normalized = " ".join(question.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _json_safe(value: Any) -> Any:
    """Convertit récursivement les valeurs NumPy/Pydantic en JSON standard."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


class ObservabilityRecorder:
    """Enregistre les traces JSONL et applique une rotation bornée des fichiers."""

    _write_lock = threading.Lock()

    def __init__(self, cfg: ObservabilityConfig) -> None:
        self.cfg = cfg
        self.path = cfg.traces_full_path

    def _rotate_if_needed(self) -> None:
        if not self.path.exists():
            return
        if self.path.stat().st_size < self.cfg.max_file_mb * 1024 * 1024:
            return
        for index in range(self.cfg.keep_files, 0, -1):
            source = self.path.with_suffix(self.path.suffix + f".{index}")
            if index == self.cfg.keep_files:
                if source.exists():
                    source.unlink()
                continue
            destination = self.path.with_suffix(self.path.suffix + f".{index + 1}")
            if source.exists():
                source.replace(destination)
        self.path.replace(self.path.with_suffix(self.path.suffix + ".1"))

    @staticmethod
    def _sources(chunks: Iterable[Any]) -> list[dict[str, Any]]:
        return [
            {
                "rank": rank,
                "chunk_id": getattr(chunk, "chunk_id", ""),
                "title": getattr(chunk, "title", ""),
                "url": getattr(chunk, "url", ""),
                "rerank_score": getattr(chunk, "rerank_score", 0.0),
                "dense_score": getattr(chunk, "dense_score", 0.0),
                "bm25_score": getattr(chunk, "bm25_score", 0.0),
            }
            for rank, chunk in enumerate(chunks, start=1)
        ]

    def record(
        self,
        *,
        event_type: str,
        question: str,
        source: str,
        status: str,
        chunks: Iterable[Any] = (),
        response: Any | None = None,
        retrieval_stats: dict[str, Any] | None = None,
        prompt_stats: dict[str, Any] | None = None,
        generation_stats: dict[str, Any] | None = None,
        total_seconds: float | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if not self.cfg.enabled:
            return
        chunk_list = list(chunks)
        answer = getattr(response, "answer", "") if response is not None else ""
        citations = getattr(response, "citations", []) if response is not None else []
        event = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id or new_run_id(source),
            "event_type": event_type,
            "source": source,
            "status": status,
            "question_hash": _question_hash(question),
            "question": question if self.cfg.log_questions else None,
            "answer": answer if self.cfg.log_answers else None,
            "answer_chars": len(answer),
            "citation_count": len(citations),
            "citation_indices": [getattr(citation, "index", None) for citation in citations],
            "retrieved_sources": self._sources(chunk_list),
            "retrieval_stats": retrieval_stats or {},
            "prompt_stats": prompt_stats or {},
            "generation_stats": generation_stats or {},
            "total_seconds": total_seconds,
            "metadata": metadata or {},
            "error": error,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(_json_safe(event), ensure_ascii=False)
        with self._write_lock:
            self._rotate_if_needed()
            with open(self.path, "a", encoding="utf-8") as file:
                file.write(line + "\n")

    def record_error(
        self, question: str, source: str, error: Exception, *, run_id: str | None = None,
    ) -> None:
        self.record(
            event_type="error",
            question=question,
            source=source,
            status="error",
            run_id=run_id,
            error=f"{type(error).__name__}: {error}",
        )


def load_traces(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Charge les traces valides; une dernière ligne incomplète est ignorée."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Une interruption pendant une écriture ne doit pas rendre le
                # tableau d'observabilité entier inutilisable.
                continue
    return rows[-limit:] if limit else rows


def summarize_traces(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Résumé compact exploité par le notebook d'observabilité."""
    if not events:
        return {
            "events": 0,
            "success_rate": 0.0,
            "cache_hit_rate": 0.0,
            "mean_total_seconds": 0.0,
            "p95_total_seconds": 0.0,
            "sources": {},
        }
    durations = sorted(
        float(event["total_seconds"])
        for event in events
        if event.get("total_seconds") is not None
    )
    p95_index = max(0, int(0.95 * len(durations) + 0.999999) - 1) if durations else 0
    generated = [event for event in events if event.get("event_type") == "rag"]
    cache_hits = sum(
        bool(event.get("generation_stats", {}).get("cache_hit")) for event in generated
    )
    return {
        "events": len(events),
        "success_rate": sum(event.get("status") == "ok" for event in events) / len(events),
        "cache_hit_rate": cache_hits / len(generated) if generated else 0.0,
        "mean_total_seconds": sum(durations) / len(durations) if durations else 0.0,
        "p95_total_seconds": durations[p95_index] if durations else 0.0,
        "sources": dict(Counter(event.get("source", "") for event in events)),
    }
