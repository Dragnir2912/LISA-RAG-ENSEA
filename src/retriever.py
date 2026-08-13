"""Retrieval hybride moderne : Dense (LanceDB) + Sparse (BM25) -> fusion RRF
-> MMR (diversité) -> reranking cross-encoder (BAAI/bge-reranker-v2-m3).

Ce pipeline dépasse la simple recherche vectorielle : la fusion RRF combine
deux signaux complémentaires (sémantique dense + mots-clés exacts, utiles
pour sigles/numéros d'articles/noms propres), MMR évite qu'un top-k soit
dominé par des chunks quasi-identiques d'une même page, et le reranker
cross-encoder ré-ordonne finement les meilleurs candidats — au prix d'un
calcul GPU négligeable puisqu'il n'est appliqué qu'à ~15 candidats, jamais à
tout le corpus.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

import numpy as np
import torch

from config import Config
from conversation import filter_unproven_ocr_roles, retrieval_subject_text
from embeddings import EmbeddingModel
from table_facts import is_plausible_table_fact
from vectorstore import (
    VectorStoreManager,
    period_keys,
    preferred_period_keys,
    structured_measure_alignment,
)

logger = logging.getLogger(__name__)

_TITLE_STOPWORDS = {
    "a", "au", "aux", "d", "de", "des", "du", "en", "ensea", "et",
    "in", "l", "la", "le", "les", "of", "the",
}

_QUERY_TITLE_STOPWORDS = _TITLE_STOPWORDS | {
    "comment", "est", "ou", "quel", "quelle", "quelles", "quels", "qui",
    "sont", "trouver", "consulter", "propose", "proposee", "proposes",
}

_GENERIC_SINGLE_TOKEN_TITLES = {
    "accueil", "contact", "faq", "formation", "information", "presentation",
}

_SUCCESSION_RELATION_PREFIXES = {
    "ancien", "apres", "avant", "chronolog", "histor", "preced", "remplac", "succed",
}

_COMPARATIVE_NUMERIC_RE = re.compile(
    r"\b(?:compar(?:e|ee|er|aison)|contre|ecart|difference|"
    r"scenario\s+de\s+reference|situation\s+de\s+reference|par\s+rapport)\b",
    re.IGNORECASE,
)


def _is_comparative_numeric_request(query: str) -> bool:
    normalized = unicodedata.normalize("NFKD", str(query).casefold())
    normalized = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return bool(_COMPARATIVE_NUMERIC_RE.search(normalized))


def _comparison_relation_bonus(query: str, meta: dict) -> float:
    """Favorise la phrase qui exprime réellement la comparaison demandée."""
    if not _is_comparative_numeric_request(query):
        return 0.0
    text = unicodedata.normalize(
        "NFKD",
        " ".join((
            str(meta.get("title", "")), str(meta.get("section", "")),
            str(meta.get("text", "")), str(meta.get("parent_text", "")),
        )).casefold(),
    )
    text = "".join(char for char in text if not unicodedata.combining(char))
    bonus = 0.0
    if re.search(r"\b(?:contre|compare|comparaison|par rapport)\b", text):
        bonus += 0.16
    if re.search(r"\b(?:scenario|situation)\s+de\s+reference\b", text):
        bonus += 0.18
    if meta.get("content_type") == "table_fact":
        bonus -= 0.12
    return bonus

@dataclass
class RetrievedChunk:
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
    content_type: str = "text"
    page: int | None = None
    indicator: str = ""
    period: str = ""
    value: str = ""
    unit: str = ""
    territory: str = ""
    measure_type: str = ""


def _reciprocal_rank_fusion(
    dense_ids_ranked: list[str], bm25_ids_ranked: list[str], k: int,
) -> dict[str, float]:
    """RRF : plus robuste qu'une pondération linéaire car indépendant de l'échelle
    des scores (cosine similarity vs score BM25 ne sont pas comparables directement)."""
    fused: dict[str, float] = {}
    for rank, cid in enumerate(dense_ids_ranked):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, cid in enumerate(bm25_ids_ranked):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return fused


def _mmr_select(
    candidate_ids: list[str], candidate_vectors: np.ndarray, relevance_scores: np.ndarray,
    top_k: int, lambda_mult: float,
) -> list[int]:
    """Maximal Marginal Relevance : sélectionne des candidats pertinents mais
    mutuellement diversifiés (évite la redondance de chunks quasi-identiques).

    `relevance_scores` vient du RRF hybride dense+BM25. Le signal lexical n'est
    donc plus remplacé par une nouvelle similarité dense après la fusion.
    """
    if len(candidate_ids) == 0:
        return []

    selected: list[int] = []
    remaining = list(range(len(candidate_ids)))

    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda i: relevance_scores[i])
        else:
            selected_vecs = candidate_vectors[selected]
            def mmr_score(i: int) -> float:
                redundancy = float(np.max(candidate_vectors[i] @ selected_vecs.T))
                return lambda_mult * relevance_scores[i] - (1 - lambda_mult) * redundancy
            best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)
    return selected


def _deduplicate_candidates(
    candidate_ids: list[str], meta_by_id: dict[str, dict],
    max_per_parent: int, max_per_url: int,
) -> list[str]:
    """Conserve l'ordre RRF tout en limitant les doublons structurels.

    On garde un seul enfant par parent et au plus quelques parents par URL avant
    le reranking. La canonicalisation inter-domaines sera traitée séparément lors
    de l'assainissement du corpus.
    """
    kept: list[str] = []
    parent_counts: dict[str, int] = {}
    url_counts: dict[str, int] = {}

    for chunk_id in candidate_ids:
        meta = meta_by_id[chunk_id]
        parent_id = meta.get("parent_id") or chunk_id
        url = meta.get("url", "")
        if parent_counts.get(parent_id, 0) >= max_per_parent:
            continue
        if url_counts.get(url, 0) >= max_per_url:
            continue
        kept.append(chunk_id)
        parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
        url_counts[url] = url_counts.get(url, 0) + 1
    return kept


def _rerank_passage(meta: dict) -> str:
    """Ajoute le titre et la section au passage évalué par le cross-encoder."""
    parts = []
    if meta.get("title"):
        parts.append(f"Titre : {meta['title']}")
    if meta.get("section"):
        parts.append(f"Section : {meta['section']}")
    parts.append(meta["text"])
    return "\n".join(parts)


def _title_signature(title: str) -> str:
    """Normalise les titres français/anglais pour repérer les pages miroirs."""
    normalized = unicodedata.normalize("NFKD", title.lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    tokens = re.findall(r"[a-z0-9]+", normalized)
    # Le tri rend la signature indépendante de l'ordre des mots. Il rapproche
    # notamment « Procédures d'admissions » et « Admissions procedures ».
    meaningful = {token for token in tokens if token not in _TITLE_STOPWORDS}
    return " ".join(sorted(meaningful))


def _search_tokens(text: str) -> set[str]:
    """Normalise les mots d'une question ou d'un titre pour un rappel par titre."""
    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    tokens = re.findall(r"[a-z0-9]+", normalized)
    result = set()
    for token in tokens:
        if token in _QUERY_TITLE_STOPWORDS or len(token) < 2:
            continue
        # Une normalisation plurielle volontairement prudente suffit pour
        # admissions/admission et seminaires/seminaire sans dépendance NLP.
        if len(token) > 5 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        result.add(token)
    return result


def _query_measure_focus_tokens(query: str) -> set[str]:
    """Repère la mesure explicitement comptée dans « combien de/d' X ».

    Le territoire cité plus loin dans la question reste un périmètre. Il ne
    doit pas battre le groupe demandé dans le classement d'un tableau.
    """
    normalized = unicodedata.normalize("NFKD", query.casefold())
    normalized = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    match = re.search(r"\bcombien\s+d(?:e|['’])?\s*([a-z][a-z-]+)", normalized)
    return _search_tokens(match.group(1)) if match else set()


def _table_fact_bonus(query: str, meta: dict, cfg) -> float:
    """Bonus explicable pour une cellule dont les facettes repondent a la question."""
    if meta.get("content_type") not in {"table_fact", "derived_fact"}:
        return 0.0
    bonus = float(cfg.table_fact_base_boost)
    query_period_facets = period_keys(query)
    fact_period_facets = period_keys(str(meta.get("period", "")))
    explicit_query_periods = {
        key for key in query_period_facets
        if re.fullmatch(r"(?:19|20)\d{2}", key)
        or key.startswith(("month:", "quarter:"))
    }
    matched_periods = explicit_query_periods & fact_period_facets
    period_matches = bool(matched_periods)
    if explicit_query_periods and period_matches:
        bonus += float(cfg.table_fact_period_boost)
    elif explicit_query_periods:
        bonus -= float(cfg.table_fact_period_mismatch_penalty)
    if len(explicit_query_periods) > 1:
        period_coverage = len(matched_periods) / len(explicit_query_periods)
        bonus += float(cfg.table_fact_period_boost) * period_coverage
        if period_coverage < 1.0:
            bonus -= float(cfg.table_fact_period_mismatch_penalty) * (
                1.0 - period_coverage
            )

    preferred_periods = preferred_period_keys(query)
    if preferred_periods and preferred_periods[0] in fact_period_facets:
        bonus += float(getattr(cfg, "table_fact_preferred_period_boost", 0.060))
    elif len(preferred_periods) > 1 and fact_period_facets & set(preferred_periods[1:]):
        bonus -= float(getattr(cfg, "table_fact_secondary_period_penalty", 0.040))

    indicator_tokens = _search_tokens(str(meta.get("indicator", "")))
    query_tokens = _search_tokens(query)
    if indicator_tokens:
        coverage = len(indicator_tokens & query_tokens) / len(indicator_tokens)
        if coverage >= cfg.table_fact_min_indicator_coverage:
            bonus += float(cfg.table_fact_indicator_boost) * min(1.0, coverage)
        focus_tokens = _query_measure_focus_tokens(query)
        if (
            focus_tokens
            and focus_tokens & indicator_tokens
            and (not explicit_query_periods or period_matches)
        ):
            bonus += float(cfg.table_fact_focus_boost)
    # Une bonne correspondance semantique ne doit jamais compenser une periode
    # explicitement fausse. C'est essentiel dans les longues series annuelles,
    # ou le meme indicateur apparait des dizaines de fois.
    if not explicit_query_periods or period_matches:
        bonus += float(getattr(cfg, "table_fact_measure_alignment_weight", 0.100)) * (
            structured_measure_alignment(
                query,
                str(meta.get("indicator", "")),
                " ".join((
                    str(meta.get("measure_type", "")),
                    str(meta.get("unit", "")),
                )),
            )
        )
    return bonus


def _chunk_metadata(chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "parent_id": chunk.parent_id,
        "url": chunk.url,
        "title": chunk.title,
        "section": chunk.section,
        "doc_type": chunk.doc_type,
        "category": chunk.category,
        "date": chunk.date,
        "text": chunk.text,
        "parent_text": chunk.parent_text,
        "content_type": getattr(chunk, "content_type", "text"),
        "page": getattr(chunk, "page", None),
        "table_id": getattr(chunk, "table_id", ""),
        "row_id": getattr(chunk, "row_id", ""),
        "indicator": getattr(chunk, "indicator", ""),
        "period": getattr(chunk, "period", ""),
        "value": getattr(chunk, "value", ""),
        "normalized_value": getattr(chunk, "normalized_value", ""),
        "unit": getattr(chunk, "unit", ""),
        "domain": getattr(chunk, "domain", ""),
        "territory": getattr(chunk, "territory", ""),
        "measure_type": getattr(chunk, "measure_type", ""),
        "qualifier": getattr(chunk, "qualifier", ""),
        "source_method": getattr(chunk, "source_method", ""),
    }


def _decimal_fact_value(value: str) -> Decimal | None:
    normalized = str(value).replace("\u00a0", " ").replace("\u202f", " ").strip()
    match = re.search(r"[-+]?\d(?:[\d ]*\d)?(?:[.,]\d+)?", normalized)
    if not match:
        return None
    numeric = match.group(0).replace(" ", "").replace(",", ".")
    try:
        return Decimal(numeric)
    except InvalidOperation:
        return None


def _derived_ratio_result(
    component_ids: list[str],
    meta_by_id: dict[str, dict],
    *,
    allow_cross_document: bool = False,
    use_absolute_numerator: bool = False,
) -> RetrievedChunk | None:
    """Calcule une part uniquement depuis deux cellules deja appariees."""
    if len(component_ids) != 2 or any(cid not in meta_by_id for cid in component_ids):
        return None
    numerator = meta_by_id[component_ids[0]]
    denominator = meta_by_id[component_ids[1]]
    same_source_page = (
        numerator.get("url") == denominator.get("url")
        and numerator.get("page") == denominator.get("page")
    )
    if not same_source_page and not allow_cross_document:
        return None
    numerator_periods = period_keys(str(numerator.get("period", "")))
    denominator_periods = period_keys(str(denominator.get("period", "")))
    numerator_years = {
        key for key in numerator_periods if re.fullmatch(r"(?:19|20)\d{2}", key)
    }
    denominator_years = {
        key for key in denominator_periods if re.fullmatch(r"(?:19|20)\d{2}", key)
    }
    numerator_specific = {
        key for key in numerator_periods if key.startswith(("month:", "quarter:"))
    }
    denominator_specific = {
        key for key in denominator_periods if key.startswith(("month:", "quarter:"))
    }
    if not (numerator_years & denominator_years):
        return None
    if (
        numerator_specific and denominator_specific
        and not (numerator_specific & denominator_specific)
    ):
        return None
    numerator_unit = _search_tokens(str(numerator.get("unit", "")))
    denominator_unit = _search_tokens(str(denominator.get("unit", "")))
    if not numerator_unit or numerator_unit != denominator_unit:
        return None
    numerator_value = _decimal_fact_value(str(numerator.get("value", "")))
    denominator_value = _decimal_fact_value(str(denominator.get("value", "")))
    if numerator_value is None or denominator_value in {None, Decimal(0)}:
        return None
    ratio_numerator = abs(numerator_value) if use_absolute_numerator else numerator_value
    percentage = (
        ratio_numerator / denominator_value * Decimal(100)
    ).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    percentage_text = format(percentage, "f").replace(".", ",")
    period = str(numerator.get("period", ""))
    page = int(numerator.get("page") or 0) or None
    evidence = (
        "Calcul deterministe a partir de deux faits statistiques officiels "
        "de meme periode et de meme unite.\n"
        f"Numerateur : {numerator.get('indicator', '')} — {period} — "
        f"{numerator.get('value', '')} {numerator.get('unit', '')}.\n"
        f"Denominateur : {denominator.get('indicator', '')} — {period} — "
        f"{denominator.get('value', '')} {denominator.get('unit', '')}.\n"
        f"Part calculee : "
        + ("valeur absolue de " if use_absolute_numerator else "")
        + f"{numerator.get('value', '')} / "
        f"{denominator.get('value', '')} × 100 = {percentage_text} %."
    )
    return RetrievedChunk(
        chunk_id=f"derived_ratio:{component_ids[0]}:{component_ids[1]}",
        url=numerator.get("url", ""),
        title=numerator.get("title", ""),
        section=f"Calcul de part — page {page}" if page else "Calcul de part",
        doc_type=numerator.get("doc_type", "pdf"),
        category=numerator.get("category", ""),
        date=numerator.get("date"),
        text=evidence,
        parent_text=evidence,
        dense_score=0.0,
        bm25_score=0.0,
        rerank_score=1.0,
        content_type="derived_fact",
        page=page,
        indicator=(
            f"Part de {numerator.get('indicator', '')} dans "
            f"{denominator.get('indicator', '')}"
        ),
        period=period,
        value=percentage_text,
        unit="%",
    )


def _title_match_score(query_tokens: set[str], title: str, min_coverage: float) -> float:
    title_tokens = _search_tokens(title)
    if not title_tokens:
        return 0.0
    overlap = query_tokens & title_tokens
    coverage = len(overlap) / len(title_tokens)
    if coverage < min_coverage:
        return 0.0
    if len(title_tokens) == 1:
        token = next(iter(title_tokens))
        if token in _GENERIC_SINGLE_TOKEN_TITLES or len(token) < 5 or token not in query_tokens:
            return 0.0
    elif len(overlap) < 2:
        return 0.0
    return coverage + min(len(overlap), 4) / 10.0


class Reranker:
    """Cross-encoder BAAI/bge-reranker-v2-m3, chargé/déchargé à la demande pour
    respecter le budget de 8 Go de VRAM partagé avec l'embedding model et le LLM."""

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(
                self.model_name, use_fp16=(self.device == "cuda"), device=self.device,
            )

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        self._ensure_loaded()
        pairs = [[query, p] for p in passages]
        scores = self._model.compute_score(pairs, normalize=True)
        return scores if isinstance(scores, list) else [scores]

    def unload(self) -> None:
        del self._model
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class HybridRetriever:
    def __init__(self, config: Config, embedding_model: EmbeddingModel,
                 store: VectorStoreManager, reranker: Reranker) -> None:
        self.config = config
        self.embedding_model = embedding_model
        self.store = store
        self.reranker = reranker
        # Exposé pour le notebook et les benchmarks. Les durées sont en secondes.
        self.last_timings: dict[str, float | int] = {}
        self.last_debug_rows: list[dict] = []
        self._title_entries = self._build_title_entries()
        self._facebook_context_by_url = self._build_facebook_document_contexts()

    def _build_facebook_document_contexts(self) -> dict[str, str]:
        """Reunit la legende et les sections OCR d'une meme publication.

        Le chunking hierarchique conserve utilement ces sections separees pour
        la recherche. Au moment de generer la reponse, elles appartiennent
        toutefois a une meme preuve documentaire et doivent etre transmises
        ensemble, quel que soit le chunk qui a gagne le reranking.
        """
        contexts: dict[str, list[str]] = {}
        signatures: dict[str, set[str]] = {}
        for chunk in self.store.sparse.chunk_lookup.values():
            if chunk.doc_type != "facebook_post" or not chunk.url:
                continue
            passage = (chunk.parent_text or chunk.text).strip()
            if not passage:
                continue
            seen = signatures.setdefault(chunk.url, set())
            signature = re.sub(r"\s+", " ", passage).casefold()
            if signature in seen:
                continue
            seen.add(signature)
            contexts.setdefault(chunk.url, []).append(passage)
        return {
            url: "\n\n".join(passages)
            for url, passages in contexts.items()
        }

    def _generation_parent_text(self, meta: dict, query: str) -> str:
        if meta.get("doc_type") == "facebook_post":
            combined = self._facebook_context_by_url.get(meta.get("url", ""))
            if combined:
                return filter_unproven_ocr_roles(combined, query)
        return meta.get("parent_text", meta.get("text", ""))

    def _build_title_entries(self) -> list[tuple[str, str]]:
        """Un représentant par URL suffit pour proposer une page au reranking."""
        entries: list[tuple[str, str]] = []
        seen_urls: set[str] = set()
        for chunk_id, chunk in self.store.sparse.chunk_lookup.items():
            if not chunk.title or not chunk.url or chunk.url in seen_urls:
                continue
            if not _search_tokens(chunk.title):
                continue
            entries.append((chunk_id, chunk.title))
            seen_urls.add(chunk.url)
        return entries

    def _title_candidate_ids(
        self, query: str, category_filter: str | None = None,
    ) -> list[str]:
        cfg = self.config.retrieval
        if not cfg.title_candidate_enabled or cfg.title_candidate_top_k <= 0:
            return []
        query_tokens = _search_tokens(query)
        ranked = []
        for chunk_id, title in self._title_entries:
            chunk = self.store.sparse.chunk_lookup.get(chunk_id)
            if category_filter and (
                chunk is None or chunk.category != category_filter
            ):
                continue
            score = _title_match_score(
                query_tokens, title, cfg.title_candidate_min_coverage,
            )
            if score > 0:
                ranked.append((chunk_id, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return [chunk_id for chunk_id, _ in ranked[:cfg.title_candidate_top_k]]

    def retrieve(
        self,
        query: str,
        *,
        max_results: int | None = None,
        category_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        total_start = time.perf_counter()
        cfg = self.config.retrieval
        exhaustive_mode = bool(
            max_results is not None and max_results > cfg.rerank_top_k
        )
        dense_top_k = (
            cfg.exhaustive_top_k_dense if exhaustive_mode else cfg.top_k_dense
        )
        bm25_top_k = (
            cfg.exhaustive_top_k_bm25 if exhaustive_mode else cfg.top_k_bm25
        )
        mmr_top_k = (
            cfg.exhaustive_mmr_top_k if exhaustive_mode else cfg.mmr_top_k
        )
        mmr_lambda = (
            cfg.exhaustive_mmr_lambda if exhaustive_mode else cfg.mmr_lambda
        )
        result_similarity_threshold = (
            cfg.exhaustive_max_result_vector_similarity
            if exhaustive_mode else cfg.max_result_vector_similarity
        )
        target_results = max(
            1,
            min(max_results or cfg.rerank_top_k, mmr_top_k),
        )
        self.last_debug_rows = []

        stage_start = time.perf_counter()
        query_vector = self.embedding_model.encode([query], is_query=True)[0]
        query_embedding_seconds = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        dense_results = self.store.dense.search(
            query_vector, dense_top_k, category_filter=category_filter,
        )
        dense_search_seconds = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        bm25_results = self.store.sparse.search(
            query, bm25_top_k, category_filter=category_filter,
        )
        bm25_search_seconds = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        ratio_component_results = self.store.sparse.structured_ratio_components(
            query,
            category_filter=category_filter,
        )
        table_fact_results = self.store.sparse.structured_fact_candidates(
            query,
            cfg.table_fact_candidate_top_k,
            category_filter=category_filter,
        )
        if ratio_component_results:
            combined_table_facts = []
            seen_table_fact_ids = set()
            for candidate in ratio_component_results + table_fact_results:
                if candidate[0] in seen_table_fact_ids:
                    continue
                seen_table_fact_ids.add(candidate[0])
                combined_table_facts.append(candidate)
            table_fact_results = combined_table_facts[:cfg.table_fact_candidate_top_k]
        table_fact_search_seconds = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        dense_ids = [r["chunk_id"] for r in dense_results]
        bm25_ids = [cid for cid, _ in bm25_results]
        table_fact_ids = [cid for cid, _ in table_fact_results]
        ratio_component_ids = [cid for cid, _ in ratio_component_results]
        dense_score_by_id = {r["chunk_id"]: r["similarity_score"] for r in dense_results}
        bm25_score_by_id = dict(bm25_results)

        fused_scores = _reciprocal_rank_fusion(dense_ids, bm25_ids, cfg.rrf_k)
        for rank, chunk_id in enumerate(table_fact_ids, start=1):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
                cfg.table_fact_rrf_weight / (cfg.rrf_k + rank)
            )

        page_rescue_started = time.perf_counter()
        page_rescue_results: list[tuple[str, float]] = []
        if cfg.page_rescue_enabled and fused_scores:
            seed_urls: list[str] = []
            dense_url_by_id = {
                row["chunk_id"]: row.get("url", "") for row in dense_results
            }
            for chunk_id in sorted(fused_scores, key=fused_scores.get, reverse=True):
                url = dense_url_by_id.get(chunk_id, "")
                if not url:
                    chunk = self.store.sparse.chunk_lookup.get(chunk_id)
                    url = chunk.url if chunk is not None else ""
                if url and url not in seed_urls:
                    seed_urls.append(url)
                if len(seed_urls) >= cfg.page_rescue_top_documents:
                    break
            page_rescue_results = self.store.sparse.intra_document_candidates(
                query,
                seed_urls,
                per_document=cfg.page_rescue_per_document,
                category_filter=category_filter,
            )
            for rank, (chunk_id, _score) in enumerate(page_rescue_results, start=1):
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
                    cfg.page_rescue_rrf_weight / (cfg.rrf_k + rank)
                )
        page_rescue_seconds = time.perf_counter() - page_rescue_started
        if not fused_scores:
            self.last_timings = {
                "query_embedding_seconds": query_embedding_seconds,
                "dense_search_seconds": dense_search_seconds,
                "bm25_search_seconds": bm25_search_seconds,
                "table_fact_search_seconds": table_fact_search_seconds,
                "page_rescue_seconds": page_rescue_seconds,
                "fusion_metadata_seconds": time.perf_counter() - stage_start,
                "vector_lookup_mmr_seconds": 0.0,
                "rerank_seconds": 0.0,
                "total_seconds": time.perf_counter() - total_start,
                "dense_candidates": len(dense_results),
                "bm25_candidates": len(bm25_results),
                "structured_fact_candidates": len(table_fact_results),
                "page_rescue_candidates": len(page_rescue_results),
                "fused_candidates": 0,
                "mmr_candidates": 0,
                "returned_chunks": 0,
                "category_filter": category_filter or "",
            }
            return []

        fused_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
        fused_ids_before_dedup = list(fused_ids)

        # Métadonnées des candidats fusionnés : dense_results en a déjà la majorité ;
        # pour les hits BM25-only, on va chercher le Chunk complet dans le lookup sparse.
        meta_by_id = {r["chunk_id"]: r for r in dense_results}
        for cid in fused_ids:
            if cid not in meta_by_id and cid in self.store.sparse.chunk_lookup:
                meta_by_id[cid] = _chunk_metadata(
                    self.store.sparse.chunk_lookup[cid]
                )

        fused_ids = [
            cid for cid in fused_ids
            if cid in meta_by_id
            and (
                meta_by_id[cid].get("content_type") != "table_fact"
                or is_plausible_table_fact(meta_by_id[cid])
            )
        ]
        facet_bonus_by_id = {
            cid: _table_fact_bonus(query, meta_by_id[cid], cfg)
            for cid in fused_ids
        }
        for cid, bonus in facet_bonus_by_id.items():
            fused_scores[cid] += bonus
        fused_ids.sort(key=fused_scores.get, reverse=True)
        fused_ids_before_dedup = list(fused_ids)
        fused_count_before_dedup = len(fused_ids)
        fused_ids = _deduplicate_candidates(
            fused_ids,
            meta_by_id,
            max_per_parent=cfg.max_candidates_per_parent,
            max_per_url=cfg.max_candidates_per_url,
        )
        candidate_texts = [meta_by_id[cid]["text"] for cid in fused_ids]
        fusion_metadata_seconds = time.perf_counter() - stage_start

        # Les embeddings des documents ont déjà été calculés pendant l'indexation.
        # On les récupère depuis le cache RAM de LanceDB au lieu de rappeler BGE-M3.
        stage_start = time.perf_counter()
        candidate_vectors = self.store.dense.get_vectors(fused_ids)
        relevance_scores = np.asarray(
            [fused_scores[cid] for cid in fused_ids], dtype=np.float32,
        )
        candidate_vector_by_id = {
            chunk_id: candidate_vectors[index]
            for index, chunk_id in enumerate(fused_ids)
        }
        max_relevance = float(relevance_scores.max()) if relevance_scores.size else 1.0
        if max_relevance > 0:
            relevance_scores = relevance_scores / max_relevance

        mmr_idx = _mmr_select(
            fused_ids, candidate_vectors, relevance_scores, mmr_top_k, mmr_lambda,
        )
        mmr_ids = [fused_ids[i] for i in mmr_idx]

        # Pour une demande exhaustive, le terme informatif rare de la question
        # sert de rappel lexical général. Les documents concernés rejoignent le
        # lot du reranker ; ils ne sont pas forcés dans la réponse et restent
        # soumis au cross-encoder ainsi qu'aux filtres de source.
        anchor_term: str | None = None
        anchor_candidate_ids: list[str] = []
        if exhaustive_mode:
            anchor_term, anchor_candidate_ids = (
                self.store.sparse.rare_query_term_candidates(
                    query,
                    max_document_frequency=(
                        cfg.exhaustive_anchor_max_document_frequency
                    ),
                    max_candidates=cfg.exhaustive_anchor_max_candidates,
                    category_filter=category_filter,
                )
            )
            for anchor_id in anchor_candidate_ids:
                chunk = self.store.sparse.chunk_lookup.get(anchor_id)
                if chunk is None:
                    continue
                if anchor_id not in meta_by_id:
                    meta_by_id[anchor_id] = _chunk_metadata(chunk)
                if anchor_id not in candidate_vector_by_id:
                    candidate_vector_by_id[anchor_id] = (
                        self.store.dense.get_vectors([anchor_id])[0]
                    )
                if anchor_id not in mmr_ids:
                    mmr_ids.append(anchor_id)

        # Une page dont le titre correspond fortement à la question peut être
        # absente du top dense/BM25 lorsque son corps est très court. On ajoute
        # au plus un tel passage au lot du cross-encoder : il n'est jamais forcé
        # dans les résultats et doit encore être validé par le reranker.
        title_candidate_ids = self._title_candidate_ids(
            query, category_filter=category_filter,
        )
        title_candidates_added = 0
        for title_id in title_candidate_ids:
            if title_id in mmr_ids:
                continue
            chunk = self.store.sparse.chunk_lookup.get(title_id)
            if chunk is None:
                continue
            if title_id not in meta_by_id:
                meta_by_id[title_id] = _chunk_metadata(chunk)
            title_vector = self.store.dense.get_vectors([title_id])[0]
            candidate_vector_by_id[title_id] = title_vector
            mmr_ids.append(title_id)
            title_candidates_added += 1
        mmr_texts = [_rerank_passage(meta_by_id[chunk_id]) for chunk_id in mmr_ids]
        vector_lookup_mmr_seconds = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        rerank_scores = self.reranker.score(query, mmr_texts)
        rerank_seconds = time.perf_counter() - stage_start
        anchor_id_set = set(anchor_candidate_ids)
        # Les libellés techniques de contextualisation (« Question précédente »,
        # « Référence résolue… ») améliorent le rappel, mais ne sont jamais des
        # relations documentaires à exiger des passages. Sans cette séparation,
        # « précédente » était pris pour une succession et une liste exhaustive
        # pouvait ne réserver qu'un seul de ses membres.
        evidence_query = retrieval_subject_text(query)
        query_evidence_tokens = _search_tokens(evidence_query) - (
            {anchor_term} if anchor_term else set()
        )
        succession_tokens = {
            token for token in query_evidence_tokens
            if any(token.startswith(prefix) for prefix in _SUCCESSION_RELATION_PREFIXES)
        }
        required_evidence_tokens = succession_tokens or query_evidence_tokens
        anchor_evidence_id_set: set[str] = set()
        for chunk_id in anchor_id_set:
            meta = meta_by_id[chunk_id]
            document_tokens = _search_tokens("\n".join((
                meta.get("title", ""),
                meta.get("section", ""),
                meta.get("text", ""),
            )))
            if not anchor_term or anchor_term not in document_tokens:
                continue
            # Quand la question contient une relation (« succédé », « associé »,
            # « accrédité »...), la catégorie seule ne suffit pas : le passage
            # doit aussi reprendre au moins une autre notion de la demande.
            if required_evidence_tokens and not (required_evidence_tokens & document_tokens):
                continue
            anchor_evidence_id_set.add(chunk_id)
        table_fact_channel_rank = {
            chunk_id: rank
            for rank, chunk_id in enumerate(table_fact_ids, start=1)
        }
        page_rescue_rank = {
            chunk_id: rank
            for rank, (chunk_id, _score) in enumerate(page_rescue_results, start=1)
        }
        page_rescue_bonus_by_id = {
            chunk_id: float(cfg.page_rescue_rank_weight) / (rank ** 0.5)
            for chunk_id, rank in page_rescue_rank.items()
        }
        direct_structured_request = not _is_comparative_numeric_request(query)
        structured_rank_bonus_by_id = {
            chunk_id: (
                float(getattr(cfg, "table_fact_channel_rank_weight", 0.180))
                / (rank ** 0.5)
                if direct_structured_request else 0.0
            )
            for chunk_id, rank in table_fact_channel_rank.items()
        }
        comparison_bonus_by_id = {
            chunk_id: _comparison_relation_bonus(query, meta_by_id[chunk_id])
            for chunk_id in mmr_ids
        }
        selection_score_by_id = {
            chunk_id: float(score)
            + facet_bonus_by_id.get(chunk_id, 0.0)
            + structured_rank_bonus_by_id.get(chunk_id, 0.0)
            + page_rescue_bonus_by_id.get(chunk_id, 0.0)
            + comparison_bonus_by_id.get(chunk_id, 0.0)
            + (
                cfg.exhaustive_anchor_rerank_bonus
                if exhaustive_mode and chunk_id in anchor_evidence_id_set else 0.0
            )
            for chunk_id, score in zip(mmr_ids, rerank_scores)
        }
        ranked = sorted(
            zip(mmr_ids, mmr_texts, rerank_scores),
            key=lambda item: selection_score_by_id[item[0]],
            reverse=True,
        )
        rerank_score_by_id = {
            chunk_id: float(score)
            for chunk_id, _, score in ranked
        }
        reranked_ids = [chunk_id for chunk_id, _, _ in ranked]
        # Une liste ne doit pas perdre un passage qui contient à la fois le nom
        # de la collection et la relation demandée. Le cross-encoder confirme
        # toujours la pertinence avant qu'une place de couverture soit réservée.
        coverage_reserved_ids = {
            chunk_id for chunk_id in anchor_evidence_id_set
            if rerank_score_by_id.get(chunk_id, 0.0)
            >= cfg.exhaustive_anchor_title_min_rerank_score
        }
        selection_ranked = (
            [item for item in ranked if item[0] in coverage_reserved_ids]
            + [item for item in ranked if item[0] not in coverage_reserved_ids]
        )
        # Dans une demande de liste, le grand bassin de rappel peut contenir
        # plusieurs pages vaguement liées. Les transmettre au générateur dilue
        # les relations utiles et augmente les contresens. On filtre seulement
        # les candidats qui ne prouvent pas la couverture ; toute preuve ayant
        # la catégorie + la relation demandées reste protégée, même avec un
        # score plus bas (cas des annonces courtes et des noms rares).
        best_rerank_score = max(rerank_score_by_id.values(), default=0.0)
        exhaustive_noncoverage_cutoff = max(
            cfg.exhaustive_noncoverage_score_floor,
            best_rerank_score - cfg.exhaustive_noncoverage_relative_margin,
        )
        bm25_rank_by_id = {
            chunk_id: rank for rank, chunk_id in enumerate(bm25_ids, start=1)
        }

        results = []
        result_url_counts: dict[str, int] = {}
        result_url_content_types: dict[str, set[str]] = {}
        selected_vectors: list[np.ndarray] = []
        selected_title_signatures: dict[str, set[str]] = {}
        semantic_duplicates_filtered = 0
        low_relevance_noncoverage_filtered = 0
        for cid, _, rerank_score in selection_ranked:
            if rerank_score < cfg.min_similarity_score:
                continue
            if (
                exhaustive_mode
                and cid not in coverage_reserved_ids
                and rerank_score < exhaustive_noncoverage_cutoff
            ):
                low_relevance_noncoverage_filtered += 1
                continue
            meta = meta_by_id[cid]
            url = meta["url"]
            existing_content_types = result_url_content_types.get(url, set())
            max_for_url = (
                cfg.max_table_fact_results_per_url
                if (
                    meta.get("content_type") == "table_fact"
                    or "table_fact" in existing_content_types
                )
                else cfg.max_results_per_url
            )
            if result_url_counts.get(url, 0) >= max_for_url:
                continue

            title_signature = _title_signature(meta.get("title", ""))
            if (
                title_signature
                and title_signature in selected_title_signatures
                and url not in selected_title_signatures[title_signature]
            ):
                semantic_duplicates_filtered += 1
                continue

            vector = candidate_vector_by_id[cid]
            if selected_vectors:
                max_similarity = max(float(vector @ selected) for selected in selected_vectors)
                if max_similarity >= result_similarity_threshold:
                    semantic_duplicates_filtered += 1
                    continue

            results.append(RetrievedChunk(
                chunk_id=cid, url=url, title=meta["title"], section=meta["section"],
                doc_type=meta["doc_type"], category=meta["category"], date=meta.get("date"),
                text=meta["text"],
                parent_text=self._generation_parent_text(meta, query),
                dense_score=dense_score_by_id.get(cid, 0.0),
                bm25_score=bm25_score_by_id.get(cid, 0.0),
                rerank_score=rerank_score,
                coverage_reserved=cid in coverage_reserved_ids,
                content_type=meta.get("content_type", "text"),
                page=int(meta.get("page") or 0) or None,
                indicator=meta.get("indicator", ""),
                period=meta.get("period", ""),
                value=meta.get("value", ""),
                unit=meta.get("unit", ""),
                territory=meta.get("territory", ""),
                measure_type=meta.get("measure_type", ""),
            ))
            result_url_counts[url] = result_url_counts.get(url, 0) + 1
            result_url_content_types.setdefault(url, set()).add(
                meta.get("content_type", "text")
            )
            selected_vectors.append(vector)
            if title_signature:
                selected_title_signatures.setdefault(title_signature, set()).add(url)
            if len(results) >= target_results:
                break

        # Sauvegarde lexicale : un passage classé tout en haut par BM25 et confirmé
        # par le cross-encoder ne doit pas disparaître juste sous le top-k. On ne
        # touche jamais aux quatre premières places ; au besoin, seul le dernier
        # résultat est remplacé. Cela protège notamment les pages courtes riches en
        # noms propres, sigles, fonctions ou numéros d'article.
        lexical_rescue_applied = False
        current_ids = {result.chunk_id for result in results}
        current_urls = {result.url for result in results}
        current_titles = {_title_signature(result.title) for result in results}
        ranked_bm25_candidates = sorted(
            (
                chunk_id for chunk_id in reranked_ids
                if chunk_id in bm25_rank_by_id
                and bm25_rank_by_id[chunk_id] <= cfg.lexical_rescue_max_bm25_rank
            ),
            key=bm25_rank_by_id.get,
        )
        for rescue_id in ranked_bm25_candidates:
            rescue_score = rerank_score_by_id[rescue_id]
            if rescue_id in current_ids:
                break
            if rescue_score < cfg.lexical_rescue_min_rerank_score:
                continue
            rescue_meta = meta_by_id[rescue_id]
            rescue_title = _title_signature(rescue_meta.get("title", ""))
            same_url_indices = [
                index for index, result in enumerate(results)
                if result.url == rescue_meta["url"]
            ]
            same_url_replacement_index: int | None = None
            if same_url_indices:
                rescue_bonus = facet_bonus_by_id.get(rescue_id, 0.0)
                existing_bonuses = [
                    facet_bonus_by_id.get(results[index].chunk_id, 0.0)
                    for index in same_url_indices
                ]
                if (
                    rescue_meta.get("content_type") != "table_fact"
                    or rescue_bonus <= max(existing_bonuses, default=0.0)
                ):
                    continue
                same_url_replacement_index = same_url_indices[
                    existing_bonuses.index(min(existing_bonuses))
                ]
            if (
                same_url_replacement_index is None
                and rescue_title
                and rescue_title in current_titles
            ):
                continue

            if same_url_replacement_index is not None:
                comparison_results = [
                    result for index, result in enumerate(results)
                    if index != same_url_replacement_index
                ]
            else:
                comparison_results = (
                    results[:-1] if len(results) >= target_results else results
                )
            rescue_vector = candidate_vector_by_id[rescue_id]
            comparison_vectors = [
                candidate_vector_by_id[result.chunk_id]
                for result in comparison_results
            ]
            if comparison_vectors and max(
                float(rescue_vector @ vector) for vector in comparison_vectors
            ) >= result_similarity_threshold:
                continue

            rescue_result = RetrievedChunk(
                chunk_id=rescue_id,
                url=rescue_meta["url"],
                title=rescue_meta["title"],
                section=rescue_meta["section"],
                doc_type=rescue_meta["doc_type"],
                category=rescue_meta["category"],
                date=rescue_meta.get("date"),
                text=rescue_meta["text"],
                parent_text=self._generation_parent_text(rescue_meta, query),
                dense_score=dense_score_by_id.get(rescue_id, 0.0),
                bm25_score=bm25_score_by_id.get(rescue_id, 0.0),
                rerank_score=rescue_score,
                coverage_reserved=rescue_id in coverage_reserved_ids,
                content_type=rescue_meta.get("content_type", "text"),
                page=int(rescue_meta.get("page") or 0) or None,
                indicator=rescue_meta.get("indicator", ""),
                period=rescue_meta.get("period", ""),
                value=rescue_meta.get("value", ""),
                unit=rescue_meta.get("unit", ""),
                territory=rescue_meta.get("territory", ""),
                measure_type=rescue_meta.get("measure_type", ""),
            )
            if same_url_replacement_index is not None:
                results[same_url_replacement_index] = rescue_result
            elif len(results) >= target_results:
                results[-1] = rescue_result
            else:
                results.append(rescue_result)
            lexical_rescue_applied = True
            break

        normalized_ratio_query = unicodedata.normalize("NFKD", query.casefold())
        normalized_ratio_query = "".join(
            char for char in normalized_ratio_query
            if not unicodedata.combining(char)
        )
        cross_document_ratio_requested = bool(re.search(
            r"\ben\s+rapportant\b", normalized_ratio_query,
        ))
        derived_ratio = _derived_ratio_result(
            ratio_component_ids,
            meta_by_id,
            allow_cross_document=cross_document_ratio_requested,
            use_absolute_numerator=(
                cross_document_ratio_requested
                and "deficit" in normalized_ratio_query
            ),
        )
        if derived_ratio is not None:
            if cross_document_ratio_requested:
                component_results: list[RetrievedChunk] = []
                existing_by_id = {result.chunk_id: result for result in results}
                for component_id in ratio_component_ids:
                    existing = existing_by_id.get(component_id)
                    if existing is not None:
                        component_results.append(existing)
                        continue
                    component = meta_by_id[component_id]
                    component_results.append(RetrievedChunk(
                        chunk_id=component_id,
                        url=component.get("url", ""),
                        title=component.get("title", ""),
                        section=component.get("section", ""),
                        doc_type=component.get("doc_type", "pdf"),
                        category=component.get("category", ""),
                        date=component.get("date"),
                        text=component.get("text", ""),
                        parent_text=component.get("parent_text", ""),
                        dense_score=dense_score_by_id.get(component_id, 0.0),
                        bm25_score=bm25_score_by_id.get(component_id, 0.0),
                        rerank_score=rerank_score_by_id.get(component_id, 1.0),
                        content_type=component.get("content_type", "table_fact"),
                        page=int(component.get("page") or 0) or None,
                        indicator=component.get("indicator", ""),
                        period=component.get("period", ""),
                        value=component.get("value", ""),
                        unit=component.get("unit", ""),
                        territory=component.get("territory", ""),
                        measure_type=component.get("measure_type", ""),
                    ))
                component_id_set = set(ratio_component_ids)
                results = [derived_ratio] + component_results + [
                    result for result in results
                    if result.chunk_id not in component_id_set
                ]
            else:
                results = [derived_ratio] + [
                    result for result in results if result.url != derived_ratio.url
                ]
            results = results[:target_results]

        dense_rank = {chunk_id: rank for rank, chunk_id in enumerate(dense_ids, start=1)}
        bm25_rank = bm25_rank_by_id
        table_fact_rank = {
            chunk_id: rank
            for rank, chunk_id in enumerate(table_fact_ids, start=1)
        }
        rrf_rank = {
            chunk_id: rank
            for rank, chunk_id in enumerate(fused_ids_before_dedup, start=1)
        }
        dedup_rank = {chunk_id: rank for rank, chunk_id in enumerate(fused_ids, start=1)}
        mmr_rank = {chunk_id: rank for rank, chunk_id in enumerate(mmr_ids, start=1)}
        rerank_rank = {
            chunk_id: rank for rank, chunk_id in enumerate(reranked_ids, start=1)
        }
        final_rank = {
            chunk.chunk_id: rank for rank, chunk in enumerate(results, start=1)
        }
        self.last_debug_rows = [
            {
                "chunk_id": chunk_id,
                "title": meta_by_id[chunk_id].get("title", ""),
                "section": meta_by_id[chunk_id].get("section", ""),
                "url": meta_by_id[chunk_id].get("url", ""),
                "content_type": meta_by_id[chunk_id].get("content_type", "text"),
                "page": meta_by_id[chunk_id].get("page") or None,
                "indicator": meta_by_id[chunk_id].get("indicator", ""),
                "period": meta_by_id[chunk_id].get("period", ""),
                "value": meta_by_id[chunk_id].get("value", ""),
                "unit": meta_by_id[chunk_id].get("unit", ""),
                "territory": meta_by_id[chunk_id].get("territory", ""),
                "measure_type": meta_by_id[chunk_id].get("measure_type", ""),
                "facet_bonus": facet_bonus_by_id.get(chunk_id, 0.0),
                "structured_rank_bonus": structured_rank_bonus_by_id.get(chunk_id, 0.0),
                "page_rescue_bonus": page_rescue_bonus_by_id.get(chunk_id, 0.0),
                "comparison_bonus": comparison_bonus_by_id.get(chunk_id, 0.0),
                "dense_rank": dense_rank.get(chunk_id),
                "bm25_rank": bm25_rank.get(chunk_id),
                "table_fact_rank": table_fact_rank.get(chunk_id),
                "page_rescue_rank": page_rescue_rank.get(chunk_id),
                "rrf_rank": rrf_rank.get(chunk_id),
                "dedup_rank": dedup_rank.get(chunk_id),
                "mmr_rank": mmr_rank.get(chunk_id),
                "rerank_rank": rerank_rank.get(chunk_id),
                "rerank_score": rerank_score_by_id.get(chunk_id),
                "selection_score": selection_score_by_id.get(chunk_id),
                "anchor_candidate": chunk_id in anchor_id_set,
                "anchor_evidence_candidate": chunk_id in anchor_evidence_id_set,
                # Conservé pour les anciens notebooks de diagnostic.
                "anchor_title_candidate": chunk_id in anchor_evidence_id_set,
                "coverage_reserved": chunk_id in coverage_reserved_ids,
                "final_rank": final_rank.get(chunk_id),
            }
            for chunk_id in dict.fromkeys(
                fused_ids_before_dedup + title_candidate_ids + anchor_candidate_ids
            )
            if chunk_id in meta_by_id
        ]

        self.last_timings = {
            "query_embedding_seconds": query_embedding_seconds,
            "dense_search_seconds": dense_search_seconds,
            "bm25_search_seconds": bm25_search_seconds,
            "table_fact_search_seconds": table_fact_search_seconds,
            "page_rescue_seconds": page_rescue_seconds,
            "fusion_metadata_seconds": fusion_metadata_seconds,
            "vector_lookup_mmr_seconds": vector_lookup_mmr_seconds,
            "rerank_seconds": rerank_seconds,
            "total_seconds": time.perf_counter() - total_start,
            "dense_candidates": len(dense_results),
            "bm25_candidates": len(bm25_results),
            "structured_fact_candidates": len(table_fact_results),
            "page_rescue_candidates": len(page_rescue_results),
            "ratio_component_candidates": len(ratio_component_ids),
            "fused_candidates": fused_count_before_dedup,
            "deduplicated_candidates": len(fused_ids),
            "mmr_candidates": len(mmr_ids),
            "semantic_duplicates_filtered": semantic_duplicates_filtered,
            "exhaustive_mode": exhaustive_mode,
            "effective_dense_top_k": dense_top_k,
            "effective_bm25_top_k": bm25_top_k,
            "effective_mmr_top_k": mmr_top_k,
            "effective_mmr_lambda": mmr_lambda,
            "anchor_term": anchor_term or "",
            "anchor_candidates_added": len(anchor_id_set),
            "anchor_evidence_candidates": len(anchor_evidence_id_set),
            "anchor_title_candidates": len(anchor_evidence_id_set),
            "coverage_reserved_candidates": len(coverage_reserved_ids),
            "exhaustive_noncoverage_cutoff": (
                exhaustive_noncoverage_cutoff if exhaustive_mode else 0.0
            ),
            "low_relevance_noncoverage_filtered": low_relevance_noncoverage_filtered,
            "lexical_rescue_applied": lexical_rescue_applied,
            "title_candidates_added": title_candidates_added,
            "table_fact_candidates": sum(
                meta_by_id[chunk_id].get("content_type") == "table_fact"
                for chunk_id in fused_ids
            ),
            "table_fact_results": sum(
                chunk.content_type == "table_fact" for chunk in results
            ),
            "derived_ratio_result": derived_ratio is not None,
            "returned_chunks": len(results),
            "category_filter": category_filter or "",
        }
        logger.info(
            "Retrieval %.3fs | query=%.3fs dense=%.3fs bm25=%.3fs "
            "fusion=%.3fs vectors+MMR=%.3fs rerank=%.3fs | %d résultat(s)",
            self.last_timings["total_seconds"],
            query_embedding_seconds,
            dense_search_seconds,
            bm25_search_seconds,
            fusion_metadata_seconds,
            vector_lookup_mmr_seconds,
            rerank_seconds,
            len(results),
        )
        return results
