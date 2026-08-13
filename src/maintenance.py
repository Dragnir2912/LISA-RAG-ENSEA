"""Opérations reproductibles utilisées par les notebooks et la console admin."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from chunker import Chunk, HierarchicalChunker
from cleaner import HTMLCleaner
from config import Config, load_config
from embeddings import EmbeddingModel
from facebook_processor import FacebookPostProcessor
from vectorstore import VectorStoreManager


ProgressCallback = Callable[[int, str], None]


def _progress(callback: ProgressCallback | None, value: int, message: str) -> None:
    if callback:
        callback(value, message)
    print(f"PROGRESS {value} {message}", flush=True)


def _read_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        return []
    return [
        Chunk(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_chunks(path: Path, chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.__dict__, ensure_ascii=False) + "\n")


def clean_corpora(
    config: Config, callback: ProgressCallback | None = None
) -> dict[str, int]:
    _progress(callback, 5, "Nettoyage des sources institutionnelles et économiques")
    cleaned = HTMLCleaner(config).clean_all(clear_output=True)
    facebook_count = 0
    if config.facebook_scraping.enabled:
        _progress(callback, 65, "Préparation des publications Facebook")
        facebook_report = FacebookPostProcessor(config).process_all()
        facebook_count = int(facebook_report.get("written_documents", 0))
    _progress(callback, 100, "Nettoyage terminé")
    return {"cleaned_documents": len(cleaned), "facebook_documents": facebook_count}


def chunk_corpora(
    config: Config, callback: ProgressCallback | None = None
) -> dict[str, int]:
    chunker = HierarchicalChunker(config)
    cleaned_dir = config.project.data_path / "cleaned"
    institutional_path = config.project.data_path / "chunks" / "chunks.jsonl"
    combined_path = config.project.data_path / "chunks" / "_combined_admin.jsonl"
    _progress(callback, 10, "Découpage parent-enfant des documents")
    all_chunks = chunker.chunk_all(cleaned_dir, combined_path)
    economic_prefix = config.local_documents.public_route_prefix.rstrip("/") + "/"
    institutional = [c for c in all_chunks if not c.url.startswith(economic_prefix)]
    economic = [c for c in all_chunks if c.url.startswith(economic_prefix)]
    _write_chunks(institutional_path, institutional)
    _write_chunks(config.local_documents.chunks_full_path, economic)
    combined_path.unlink(missing_ok=True)
    _progress(callback, 70, "Découpage des publications Facebook")
    facebook = chunker.chunk_all(
        config.facebook_scraping.cleaned_candidate_full_path,
        config.facebook_scraping.chunks_candidate_full_path,
    )
    _progress(callback, 100, "Chunking terminé")
    return {
        "institutional_chunks": len(institutional),
        "economic_chunks": len(economic),
        "facebook_chunks": len(facebook),
    }


def embed_corpora(
    config: Config, callback: ProgressCallback | None = None
) -> dict[str, int]:
    paths = {
        "institutional": config.project.data_path / "chunks" / "chunks.jsonl",
        "economic": config.local_documents.chunks_full_path,
        "facebook": config.facebook_scraping.chunks_candidate_full_path,
    }
    output_paths = {
        "institutional": config.project.data_path / "embeddings" / "document_vectors.npy",
        "economic": config.local_documents.vectors_full_path,
        "facebook": config.project.data_path / "facebook" / "vectors.npy",
    }
    corpora = {name: _read_chunks(path) for name, path in paths.items()}
    total = sum(len(items) for items in corpora.values())
    if total == 0:
        raise ValueError("Aucun chunk à encoder. Exécutez d'abord le nettoyage et le chunking.")
    model = EmbeddingModel(config.embeddings)
    processed = 0
    try:
        for name, chunks in corpora.items():
            batches: list[np.ndarray] = []
            for start in range(0, len(chunks), config.embeddings.batch_size):
                batch = chunks[start:start + config.embeddings.batch_size]
                batches.append(model.encode([chunk.text for chunk in batch]))
                processed += len(batch)
                percent = 5 + int(90 * processed / total)
                _progress(callback, percent, f"Embeddings {name}: {processed}/{total}")
            vectors = (
                np.concatenate(batches, axis=0)
                if batches else np.zeros((0, model.dim or 0), dtype=np.float32)
            )
            output_paths[name].parent.mkdir(parents=True, exist_ok=True)
            np.save(output_paths[name], vectors)
    finally:
        model.unload()
    _progress(callback, 100, "Embeddings terminés")
    return {f"{name}_vectors": len(chunks) for name, chunks in corpora.items()}


def index_corpora(
    config: Config, callback: ProgressCallback | None = None
) -> dict[str, int]:
    definitions = [
        (
            "institutional",
            config.project.data_path / "chunks" / "chunks.jsonl",
            config.project.data_path / "embeddings" / "document_vectors.npy",
            config,
        ),
        (
            "economic",
            config.local_documents.chunks_full_path,
            config.local_documents.vectors_full_path,
            config.model_copy(deep=True),
        ),
        (
            "facebook",
            config.facebook_scraping.chunks_candidate_full_path,
            config.project.data_path / "facebook" / "vectors.npy",
            config.model_copy(deep=True),
        ),
    ]
    definitions[1][3].vectorstore.uri = config.local_documents.vectorstore_uri
    definitions[1][3].vectorstore.table_name = config.local_documents.vectorstore_table
    definitions[1][3].vectorstore.bm25_index_path = config.local_documents.bm25_index_path
    definitions[2][3].vectorstore.uri = config.facebook_scraping.vectorstore_candidate_uri
    definitions[2][3].vectorstore.table_name = config.facebook_scraping.vectorstore_candidate_table
    definitions[2][3].vectorstore.bm25_index_path = config.facebook_scraping.bm25_candidate_path

    counts: dict[str, int] = {}
    for position, (name, chunks_path, vectors_path, routed_config) in enumerate(definitions):
        chunks = _read_chunks(chunks_path)
        if not chunks:
            counts[f"{name}_indexed"] = 0
            continue
        if not vectors_path.exists():
            raise FileNotFoundError(f"Embeddings absents pour le corpus {name}.")
        vectors = np.load(vectors_path)
        if len(vectors) != len(chunks):
            raise ValueError(
                f"Incohérence {name}: {len(chunks)} chunks pour {len(vectors)} vecteurs."
            )
        _progress(callback, 10 + position * 30, f"Indexation {name}")
        VectorStoreManager(routed_config).index_all(chunks, vectors)
        counts[f"{name}_indexed"] = len(chunks)
    _progress(callback, 100, "Indexation LanceDB et BM25 terminée")
    return counts


def run_stage(stage: str, callback: ProgressCallback | None = None) -> dict[str, int]:
    config = load_config()
    if stage == "clean":
        return clean_corpora(config, callback)
    if stage == "chunk":
        return chunk_corpora(config, callback)
    if stage == "embed":
        return embed_corpora(config, callback)
    if stage == "index":
        return index_corpora(config, callback)
    if stage == "rebuild":
        clean_corpora(config, callback)
        chunk_corpora(config, callback)
        embed_corpora(config, callback)
        return index_corpora(config, callback)
    raise ValueError(f"Étape de maintenance inconnue : {stage}")
