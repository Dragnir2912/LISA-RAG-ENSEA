"""Encodage des chunks en vecteurs denses sur GPU.

Modèle par défaut : BAAI/bge-m3 (voir justification dans le plan d'architecture
et le notebook de comparaison : meilleur compromis qualité française / mémoire
GPU / polyvalence parmi bge-m3, multilingual-e5-large, nomic-embed-text,
Snowflake Arctic Embed et jina-embeddings-v3).

`EmbeddingModel` reste générique : n'importe quel modèle de la liste
`config.embeddings.alternatives` peut être chargé pour la comparaison, via
FlagEmbedding pour la famille BAAI (dense+sparse+colbert) ou sentence-transformers
pour les autres (interface uniforme, un seul type de retour : vecteurs denses
normalisés).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch

from config import Config, EmbeddingsConfig, load_config

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingBenchmark:
    model_name: str
    dim: int
    vram_mb: float
    encode_seconds_per_100_docs: float


class EmbeddingModel:
    """Wrapper GPU unifié pour l'embedding de documents/requêtes.

    Utilise FlagEmbedding.BGEM3FlagModel pour les modèles BAAI bge-* (accès à la
    sortie dense fp16 native), sentence-transformers pour tout le reste — donne
    la même interface `.encode()` quel que soit le modèle sélectionné dans
    `config.yaml`, ce qui permet de comparer les 5 candidats sans dupliquer de code.
    """

    def __init__(self, cfg: EmbeddingsConfig, model_name: str | None = None) -> None:
        self.cfg = cfg
        self.model_name = model_name or cfg.model_name
        self.device = cfg.device if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and cfg.device == "cuda":
            logger.warning("CUDA indisponible : repli sur CPU (sera beaucoup plus lent).")

        self._is_bge_flag = "bge-m3" in self.model_name.lower()
        self._backend = None
        self.dim: int | None = None
        self._load()

    def _load(self) -> None:
        use_fp16 = self.cfg.precision == "fp16" and self.device == "cuda"
        if self._is_bge_flag:
            from FlagEmbedding import BGEM3FlagModel

            self._backend = BGEM3FlagModel(self.model_name, use_fp16=use_fp16, device=self.device)
        else:
            from sentence_transformers import SentenceTransformer

            self._backend = SentenceTransformer(
                self.model_name, device=self.device,
                model_kwargs={"torch_dtype": torch.float16} if use_fp16 else {},
                trust_remote_code=True,  # requis par nomic-embed-text et jina-embeddings-v3
            )
        self.dim = len(self.encode(["dimension probe"])[0])

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Encode une liste de textes en vecteurs denses normalisés (np.ndarray [n, dim])."""
        if not texts:
            return np.zeros((0, self.dim or 0), dtype=np.float32)

        if self._is_bge_flag:
            out = self._backend.encode(
                texts, batch_size=self.cfg.batch_size,
                return_dense=True, return_sparse=False, return_colbert_vecs=False,
            )
            vecs = np.asarray(out["dense_vecs"], dtype=np.float32)
        else:
            vecs = self._backend.encode(
                texts, batch_size=self.cfg.batch_size,
                normalize_embeddings=self.cfg.normalize, show_progress_bar=False,
            )
            vecs = np.asarray(vecs, dtype=np.float32)

        if self.cfg.normalize and self._is_bge_flag:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        return vecs

    @property
    def tokenizer(self):
        """Tokenizer du modèle courant, réutilisable par le chunker token-aware."""
        tokenizer = getattr(self._backend, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                f"Le backend {type(self._backend).__name__} n'expose pas de tokenizer."
            )
        return tokenizer

    def unload(self) -> None:
        """Libère la VRAM — utile pour laisser la place au reranker/LLM au runtime."""
        del self._backend
        self._backend = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def benchmark_model(model_name: str, cfg: EmbeddingsConfig, sample_texts: list[str]) -> EmbeddingBenchmark:
    """Mesure latence et empreinte VRAM d'un modèle sur un échantillon (notebook étape 4)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = EmbeddingModel(cfg, model_name=model_name)
    start = time.perf_counter()
    _ = model.encode(sample_texts)
    elapsed = time.perf_counter() - start
    per_100 = elapsed / max(len(sample_texts), 1) * 100

    vram_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if torch.cuda.is_available() else 0.0
    )
    result = EmbeddingBenchmark(
        model_name=model_name, dim=model.dim or 0,
        vram_mb=vram_mb, encode_seconds_per_100_docs=per_100,
    )
    model.unload()
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    model = EmbeddingModel(config.embeddings)
    logger.info("Modèle chargé : %s (dim=%d, device=%s)", model.model_name, model.dim, model.device)


if __name__ == "__main__":
    main()
