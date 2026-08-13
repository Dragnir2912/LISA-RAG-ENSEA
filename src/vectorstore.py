"""Stockage vectoriel (LanceDB, embarqué) + index BM25 (rank_bm25), avec
réindexation incrémentale par URL.

LanceDB est choisi pour un usage 100% local sans serveur (fichiers sur disque,
comme SQLite/DuckDB), avec support natif du filtrage par métadonnées et de
l'ajout/suppression incrémentale de vecteurs — voir justification complète
dans le plan d'architecture (comparaison avec FAISS/ChromaDB/Qdrant).

L'index BM25 est volontairement reconstruit en entier à chaque réindexation
plutôt que mis à jour ligne à ligne : rank_bm25 ne supporte pas la mise à jour
incrémentale nativement, et reconstruire l'index complet ne prend que quelques
secondes pour un corpus de quelques milliers de chunks — un compromis simple
et robuste plutôt qu'une mise à jour partielle complexe à maintenir.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import pickle
import re
import time
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import lancedb
import numpy as np
import pyarrow as pa
from rank_bm25 import BM25Okapi

from chunker import Chunk
from config import Config, load_config
from conversation import exhaustive_category_tokens
from table_facts import is_plausible_table_fact

if TYPE_CHECKING:
    from embeddings import EmbeddingModel

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

_STRUCTURED_ACRONYM_EXPANSIONS = {
    "pib": {"produit", "interieur", "brut"},
    "ippi": {"indice", "prix", "production", "industrie"},
    "ica": {"indice", "chiffre", "affaires"},
    "fbcf": {"formation", "brute", "capital", "fixe", "investissement"},
}

_STRUCTURED_SYNONYM_EXPANSIONS = {
    "deficit": {"solde"},
    "excedent": {"solde"},
    "solde": {"deficit", "excedent"},
    "annuel": {"annee", "total"},
    "annee": {"annuel", "total"},
}

_RATE_INTENT_TOKENS = {
    "croissance", "evolution", "part", "pourcentage", "proportion", "ratio",
    "taux", "variation",
}
_MEASURE_GENERIC_TOKENS = _RATE_INTENT_TOKENS | {
    "indice", "niveau", "nombre", "total", "valeur",
}
_MODALITY_TOKENS = {
    "attendu", "estime", "estimation", "observe", "prevision", "prevu",
    "projete", "projection", "realise",
}
_UNIT_TOKENS = {
    "fcfa", "milliard", "million", "millier", "md", "mds", "tonne", "usd", "xof",
}
_SEX_SPECIFIC_TOKENS = {
    "femme", "feminin", "fille", "garcon", "homme", "masculin",
}
_BM25_TOKENIZER_VERSION = 8

_BM25_STOPWORDS = {
    # Mots-outils français : ils sont très fréquents et faussent fortement le
    # classement des questions naturelles (« qui est le… », « quelles sont… »).
    "a", "ai", "au", "aux", "avec", "ce", "ces", "cette", "combien", "comment",
    "d", "dans", "de", "des", "du", "elle", "en", "ensea", "est", "et",
    "etaient", "etait", "il", "ils", "l", "la", "le", "les", "leur", "leurs", "mais", "ne",
    "nous", "on", "ou", "par", "pas", "pour", "qu", "que", "quel",
    "quelle", "quelles", "quels", "qui", "se", "ses", "son", "sont",
    "sur", "un", "une", "vous",
    # Équivalents anglais présents dans les pages traduites.
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "what", "which", "who", "with",
}

# Ces mots peuvent apparaître dans la contextualisation technique d'une relance
# ou dans la formulation d'une demande de liste. Ils ne désignent jamais, à eux
# seuls, la catégorie documentaire à couvrir de manière exhaustive.
_EXHAUSTIVE_ANCHOR_STOPWORDS = {
    "annee", "complet", "complete", "exhaustif", "exhaustive", "initiale",
    "je", "liste", "precedent", "precedente", "question", "suivi", "veux",
}

_MONTH_NUMBERS = {
    "janv": 1, "janvier": 1, "january": 1,
    "fev": 2, "fevr": 2, "fevrier": 2, "feb": 2, "february": 2,
    "mars": 3, "march": 3,
    "avr": 4, "avril": 4, "apr": 4, "april": 4,
    "mai": 5, "may": 5,
    "juin": 6, "june": 6,
    "juil": 7, "juillet": 7, "july": 7,
    "aout": 8, "aug": 8, "august": 8,
    "sept": 9, "septembre": 9, "sep": 9, "september": 9,
    "oct": 10, "octobre": 10, "october": 10,
    "nov": 11, "novembre": 11, "november": 11,
    "dec": 12, "decembre": 12, "december": 12,
}


def _tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    tokens = []
    for token in _TOKEN_RE.findall(normalized):
        if token in _BM25_STOPWORDS:
            continue
        # Normalisation légère et générale du pluriel français/anglais. Elle
        # rapproche notamment finalistes/finaliste, admissions/admission et
        # laboratoires/laboratoire sans dépendance NLP supplémentaire.
        if len(token) > 5 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        if token and token not in _BM25_STOPWORDS:
            tokens.append(token)
    return tokens


def _expanded_structured_tokens(text: str) -> set[str]:
    tokens = set(_tokenize(text))
    for acronym, expansion in _STRUCTURED_ACRONYM_EXPANSIONS.items():
        if acronym in tokens:
            tokens.update(expansion)
    for token, expansion in _STRUCTURED_SYNONYM_EXPANSIONS.items():
        if token in tokens:
            tokens.update(expansion)
    return tokens


@lru_cache(maxsize=32768)
def _structured_measure_profile(
    text: str,
) -> tuple[frozenset[str], str, bool, frozenset[str], frozenset[str]]:
    """Profil lexical immuable reutilise par toutes les lignes statistiques."""
    tokens = frozenset(
        token for token in _expanded_structured_tokens(text)
        if token not in _UNIT_TOKENS
    )
    normalized = " ".join(
        token for token in _tokenize(text) if token not in _UNIT_TOKENS
    )
    has_rate_intent = bool(tokens & _RATE_INTENT_TOKENS) or "%" in text
    subject = frozenset(
        token for token in tokens
        if token not in _MEASURE_GENERIC_TOKENS
        and token not in _MODALITY_TOKENS
        and not re.fullmatch(r"(?:19|20)\d{2}", token)
    )
    return (
        tokens,
        normalized,
        has_rate_intent,
        subject,
        frozenset(tokens & _SEX_SPECIFIC_TOKENS),
    )


def preferred_period_keys(query: str) -> list[str]:
    """Retourne les periodes explicites dans leur ordre metier de preference.

    Une annee scolaire ``2023-2024`` est representee dans les brochures par la
    colonne 2024. Hors contexte scolaire, aucune extremite d'un intervalle n'est
    privilegiee arbitrairement.
    """
    normalized = unicodedata.normalize("NFKD", str(query).casefold())
    normalized = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    academic = bool(re.search(
        r"\b(?:academique|annee scolaire|eleve|etudiant|inscrit|primaire|secondaire)\w*\b",
        normalized,
    ))
    year_range = re.search(
        r"\b((?:19|20)\d{2})\s*[-/–—]\s*((?:19|20)\d{2})\b",
        normalized,
    )
    if academic and year_range:
        return [year_range.group(2), year_range.group(1)]

    facets = period_keys(query)
    specific = sorted(
        key for key in facets if key.startswith(("month:", "quarter:"))
    )
    years = re.findall(r"\b(?:19|20)\d{2}\b", normalized)
    return specific + list(dict.fromkeys(years))


def structured_measure_alignment(
    query: str,
    indicator: str,
    measure_type: str = "",
) -> float:
    """Mesure la compatibilite de la grandeur demandee avec une ligne statistique."""
    (
        query_tokens,
        normalized_query,
        query_has_rate_intent,
        query_subject,
        query_sex,
    ) = _structured_measure_profile(str(query))
    (
        indicator_tokens,
        normalized_indicator,
        indicator_has_rate_intent,
        indicator_subject,
        indicator_sex,
    ) = _structured_measure_profile(str(indicator))
    measure_tokens = set(_tokenize(measure_type))
    indicator_has_rate_intent = indicator_has_rate_intent or (
        "%" in measure_type or bool(measure_tokens & _RATE_INTENT_TOKENS)
    )
    if not query_tokens or not indicator_tokens:
        return 0.0
    overlap = len(query_tokens & indicator_tokens)
    score = overlap / max(1, len(indicator_tokens))

    # Un libelle tres generique (par exemple ``Taux de croissance``) ne doit pas
    # battre ``Secteur primaire`` lorsque la question porte precisement sur ce
    # secteur. On mesure donc aussi la couverture du *sujet* demande, apres avoir
    # retire les mots de grandeur, de modalite et les annees. Ce signal reste
    # general : il vaut aussi pour inflation, consommation, population, etc.
    if query_subject:
        subject_overlap = len(query_subject & indicator_subject)
        subject_coverage = subject_overlap / len(query_subject)
        score += 0.90 * subject_coverage
        if subject_overlap == 0 and indicator_has_rate_intent:
            score -= 0.45

    # « le PIB » vise le niveau absolu ; « taux/part ... du PIB » vise une mesure
    # relative. Cette distinction evite qu'une ligne « En % du PIB » batte le PIB.
    if "pib" in query_tokens and not query_has_rate_intent:
        if {"produit", "interieur", "brut"} <= indicator_tokens:
            score += 0.75
        if indicator_has_rate_intent:
            score -= 1.25
    elif query_has_rate_intent and indicator_has_rate_intent:
        score += 0.35
    elif indicator_has_rate_intent and not query_has_rate_intent:
        # Une demande d'effectif ou de niveau absolu ne doit pas etre remplacee
        # par un ratio, une proportion ou une variation de la meme population.
        score -= 0.75

    # Une question sur l'effectif total ne doit pas etre remplacee par une ligne
    # hommes/femmes ou garcons/filles lorsque ce sous-groupe n'est pas demande.
    if indicator_sex and not query_sex:
        score -= 0.8
    elif query_sex & indicator_sex:
        score += 0.5

    # Les libelles exacts et compacts sont plus fiables que les phrases aplaties.
    if normalized_indicator and normalized_indicator in normalized_query:
        score += 0.4
    if len(indicator) > 100:
        score -= 0.5

    # Les comptes nationaux publient souvent la meme ligne en volume, en
    # valeur nominale, brute ou CVS sur des pages voisines. Ces facettes sont
    # des contraintes de sens, pas de simples mots bonus.
    normalized_measure = " ".join(_tokenize(measure_type))
    combined_measure = f"{normalized_indicator} {normalized_measure}".strip()
    query_wants_cvs = bool(re.search(r"\bcvs\b|corrige\w* variation saison", normalized_query))
    query_wants_volume = bool(re.search(r"\bvolume\b|prix chaine|prix constant", normalized_query))
    query_wants_nominal = bool(re.search(r"\bnominal\w*\b|\bvaleur\b|prix courant", normalized_query))
    has_cvs = "cvs" in combined_measure
    has_volume = "volume" in combined_measure or "chaine" in combined_measure
    has_nominal = "nominal" in combined_measure or "valeur" in combined_measure
    for requested, present in (
        (query_wants_cvs, has_cvs),
        (query_wants_volume, has_volume),
        (query_wants_nominal, has_nominal),
    ):
        if requested:
            score += 0.70 if present else -0.95
    if query_wants_nominal and has_volume:
        score -= 0.85
    if query_wants_volume and has_nominal and not has_volume:
        score -= 0.85

    base_years = re.findall(
        r"(?:prix|base|reference|chaine\w*)[^0-9]{0,20}((?:19|20)\d{2})",
        normalized_query,
    )
    if base_years:
        score += 0.45 if any(year in normalized_measure for year in base_years) else -0.45
    return max(-1.5, min(3.0, score))


def period_keys(text: str) -> set[str]:
    """Normalise année, trimestre et mois pour rapprocher 2025 de déc-25."""
    normalized = unicodedata.normalize("NFKD", str(text).casefold())
    normalized = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
    if not normalized:
        return set()

    keys = {normalized.replace(" ", "-")}
    years = re.findall(r"\b(?:19|20)\d{2}\b", normalized)
    short_years = re.findall(r"(?<!\d)(\d{2})(?!\d)", normalized)
    if not years and short_years:
        short = int(short_years[-1])
        years = [str(2000 + short if short <= 49 else 1900 + short)]
    keys.update(years)

    month = next(
        (
            number for alias, number in _MONTH_NUMBERS.items()
            if re.search(rf"\b{re.escape(alias)}\w*\b", normalized)
        ),
        None,
    )
    quarter_matches = list(re.finditer(r"\b[qt]\s*([1-4])\b", normalized))
    quarter_words = {
        "premier": "1", "premiere": "1", "1er": "1", "1ere": "1",
        "deuxieme": "2", "second": "2", "seconde": "2", "2e": "2",
        "troisieme": "3", "3e": "3",
        "quatrieme": "4", "4e": "4",
    }
    if re.search(r"\btrimestre\w*\b", normalized):
        for word, number in quarter_words.items():
            if re.search(rf"\b{word}\b", normalized):
                keys.add(
                    f"quarter:{number}:{years[-1]}"
                    if years else f"quarter:{number}"
                )
    if month and years:
        keys.add(f"month:{month:02d}:{years[-1]}")
    if quarter_matches:
        for quarter in quarter_matches:
            # Pour une expression T3 2025 à T4 2025, l'année la plus proche à
            # droite de chaque trimestre est privilégiée.
            following = normalized[quarter.end():]
            following_year = re.search(r"\b(?:19|20)\d{2}\b", following)
            year = following_year.group(0) if following_year else (years[-1] if years else "")
            keys.add(
                f"quarter:{quarter.group(1)}:{year}"
                if year else f"quarter:{quarter.group(1)}"
            )
    return keys


def _structured_query_focus_tokens(text: str) -> set[str]:
    """Extrait le groupe mesure dans « combien de X » ou « part des X ».

    Le reste de la question contient souvent le territoire (« Côte d'Ivoire »),
    qui décrit le périmètre et non l'indicateur recherché. Le premier nom après
    ``combien de`` ou ``part des`` constitue un signal plus discriminant.
    """
    normalized = unicodedata.normalize("NFKD", str(text).casefold())
    normalized = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    patterns = (
        r"\bcombien\s+d(?:e|['’])?\s*([a-z][a-z-]+)",
        r"\b(?:part|proportion|ratio)\s+d(?:e|es|['’])?\s*([a-z][a-z-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return set(_tokenize(match.group(1)))
    return set()


def _ratio_measure_phrases(text: str) -> tuple[str, str] | None:
    """Extrait numerateur et denominateur d'une demande de part explicite."""
    normalized = unicodedata.normalize("NFKD", str(text).casefold())
    normalized = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    match = re.search(
        r"\b(?:part|proportion)\s+d(?:e|es|['’])?\s*(.+?)\s+"
        r"(?:dans|sur|par rapport a)\s+(?:le |la |les |l['’])?(.+?)"
        r"(?=\s+(?:en|pour|au)\s+(?:19|20)\d{2}\b|[?.!,;]|$)",
        normalized,
    )
    if not match:
        match = re.search(
            r"\ben\s+rapportant\s+(?:le |la |les |l['’])?(.+?)\s+"
            r"(?:au|a la|aux|a l['’])\s+(.+?)"
            r"(?=\s*,?\s*(?:quelle|quel|combien|en|pour|au)\b|[?.!,;]|$)",
            normalized,
        )
    if not match:
        return None
    numerator = match.group(1).strip(" -:;,.?")
    denominator = match.group(2).strip(" -:;,.?")
    return (numerator, denominator) if numerator and denominator else None


def _measure_phrase_score(phrase: str, indicator: str) -> float:
    phrase_tokens = set(_structured_measure_profile(phrase)[0])
    indicator_tokens = set(_structured_measure_profile(indicator)[0])
    if not phrase_tokens or not indicator_tokens:
        return -10.0
    overlap = len(phrase_tokens & indicator_tokens)
    coverage = overlap / len(phrase_tokens)
    precision = overlap / len(indicator_tokens)
    score = coverage + precision + structured_measure_alignment(phrase, indicator)
    if phrase_tokens == indicator_tokens:
        score += 2.0
    return score


def _unit_signature(unit: str) -> frozenset[str]:
    return frozenset(
        token for token in _tokenize(unit)
        if token not in {"de", "des", "du", "en", "par"}
    )


def _bm25_document_text(chunk: Chunk) -> str:
    """Texte indexé par BM25 : métadonnées éditoriales puis contenu du passage."""
    structured = "\n".join(
        value for value in (
            chunk.content_type,
            chunk.domain,
            chunk.indicator,
            chunk.period,
            chunk.value,
            chunk.normalized_value,
            chunk.unit,
            chunk.territory,
            chunk.measure_type,
        ) if value
    )
    return f"{chunk.title}\n{chunk.section}\n{structured}\n{chunk.text}"


def _chunk_is_retrievable(chunk: Chunk) -> bool:
    """Ecarte uniquement les cellules tabulaires manifestement corrompues.

    Le passage textuel d'origine reste indexe : ce filtre ne supprime donc pas
    le document, il empeche seulement une mauvaise reconstruction de cellule de
    prendre la place d'un candidat sain.
    """
    return (
        chunk.content_type != "table_fact"
        or is_plausible_table_fact({
            "indicator": chunk.indicator,
            "period": chunk.period,
            "value": chunk.value,
            "unit": chunk.unit,
        })
    )


@dataclass(frozen=True)
class IncrementalIndexReport:
    changed_urls: int
    embedded_chunks: int
    reused_chunks: int
    removed_chunks: int
    total_chunks: int
    embedding_seconds: float


def _content_fingerprint(values: dict) -> str:
    """Empreinte de tout ce qui influence le retrieval ou le contexte génératif."""
    payload = {
        field: values.get(field) or ""
        for field in LanceDBStore.SCHEMA_FIELDS
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class LanceDBStore:
    """Index vectoriel dense, embarqué sur disque, avec suppression/ajout par URL."""

    SCHEMA_FIELDS = [
        "chunk_id", "parent_id", "url", "title", "section",
        "doc_type", "depth", "date", "category", "text", "parent_text",
        "content_type", "page", "table_id", "row_id", "indicator",
        "period", "value", "normalized_value", "unit", "domain",
        "territory", "measure_type", "qualifier", "source_method",
    ]

    def __init__(self, config: Config) -> None:
        self.config = config
        self.db = lancedb.connect(str(config.vectorstore.uri_path))
        self.table_name = config.vectorstore.table_name
        self._table = None
        # Cache léger utilisé par MMR au runtime. Pour le corpus actuel :
        # 4 576 vecteurs x 1 024 dimensions x 4 octets ~= 18 Mo en RAM.
        # Cela évite de ré-encoder 30 à 60 textes avec BGE-M3 à chaque requête.
        self._vector_cache: dict[str, np.ndarray] | None = None

    @property
    def table(self):
        if self._table is None:
            if self.table_name in self.db.table_names():
                self._table = self.db.open_table(self.table_name)
            else:
                self._table = None
        return self._table

    def _rows(self, chunks: list[Chunk], vectors: np.ndarray) -> list[dict]:
        rows = []
        for chunk, vector in zip(chunks, vectors):
            row = {k: v for k, v in asdict(chunk).items() if k in self.SCHEMA_FIELDS}
            row["date"] = row["date"] or ""
            row["page"] = int(row.get("page") or 0)
            row["vector"] = vector.astype(np.float32).tolist()
            rows.append(row)
        return rows

    def _selectable_schema_fields(self) -> list[str]:
        """Reste compatible en lecture avec un index construit avant ce schema."""
        if self.table is None:
            return []
        available = set(self.table.schema.names)
        return [field for field in self.SCHEMA_FIELDS if field in available]

    def schema_is_current(self) -> bool:
        if self.table is None:
            return True
        return set(self.SCHEMA_FIELDS) <= set(self.table.schema.names)

    @staticmethod
    def _with_schema_defaults(row: dict) -> dict:
        defaults = {
            "content_type": "text",
            "page": 0,
            "table_id": "",
            "row_id": "",
            "indicator": "",
            "period": "",
            "value": "",
            "normalized_value": "",
            "unit": "",
            "domain": "",
            "territory": "",
            "measure_type": "",
            "qualifier": "",
            "source_method": "",
        }
        return {**defaults, **row}

    def create(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        rows = self._rows(chunks, vectors)
        self._table = self.db.create_table(self.table_name, data=rows, mode="overwrite")
        self._invalidate_vector_cache()
        logger.info("Table LanceDB '%s' créée avec %d vecteurs.", self.table_name, len(rows))

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        rows = self._rows(chunks, vectors)
        if self.table is None:
            self.create(chunks, vectors)
            return
        self.table.add(rows)
        self._invalidate_vector_cache()

    def delete_by_urls(self, urls: list[str]) -> None:
        if self.table is None or not urls:
            return
        quoted = ", ".join(f"'{u.replace(chr(39), chr(39) * 2)}'" for u in urls)
        self.table.delete(f"url IN ({quoted})")
        self._invalidate_vector_cache()

    def metadata_rows(self) -> list[dict]:
        """Retourne l'inventaire sans charger les vecteurs de 1024 dimensions."""
        if self.table is None:
            return []
        row_count = self.table.count_rows()
        if row_count == 0:
            return []
        rows = (
            self.table.search()
            .select(self._selectable_schema_fields())
            .limit(row_count)
            .to_list()
        )
        return [self._with_schema_defaults(row) for row in rows]

    def _invalidate_vector_cache(self) -> None:
        """Invalide le cache après toute modification de la table dense."""
        self._vector_cache = None

    def _ensure_vector_cache(self) -> None:
        """Charge une fois en RAM les vecteurs déjà calculés dans LanceDB.

        Seules les colonnes `chunk_id` et `vector` sont lues. Les textes et les
        métadonnées volumineuses ne sont donc pas dupliqués dans ce cache.
        """
        if self._vector_cache is not None:
            return
        if self.table is None:
            self._vector_cache = {}
            return

        row_count = self.table.count_rows()
        arrow_table = (
            self.table.search()
            .select(["chunk_id", "vector"])
            .limit(row_count)
            .to_arrow()
        )
        chunk_ids = arrow_table.column("chunk_id").to_pylist()
        vectors = arrow_table.column("vector").to_pylist()
        self._vector_cache = {
            chunk_id: np.asarray(vector, dtype=np.float32)
            for chunk_id, vector in zip(chunk_ids, vectors)
        }
        logger.info(
            "Cache RAM des vecteurs chargé : %d vecteurs (%.1f Mo).",
            len(self._vector_cache),
            sum(vector.nbytes for vector in self._vector_cache.values()) / (1024 ** 2),
        )

    def get_vectors(self, chunk_ids: list[str]) -> np.ndarray:
        """Retourne les vecteurs de chunks dans le même ordre que `chunk_ids`."""
        self._ensure_vector_cache()
        cache = self._vector_cache or {}
        missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in cache]
        if missing:
            preview = ", ".join(missing[:3])
            raise KeyError(
                f"{len(missing)} chunk(s) absent(s) du cache vectoriel : {preview}"
            )
        if not chunk_ids:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([cache[chunk_id] for chunk_id in chunk_ids])

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        *,
        category_filter: str | None = None,
    ) -> list[dict]:
        if self.table is None:
            return []
        search = self.table.search(query_vector.astype(np.float32).tolist())
        if category_filter:
            escaped = category_filter.replace("'", "''")
            search = search.where(f"category = '{escaped}'")
        results = (
            search.select(self._selectable_schema_fields())
            .limit(max(top_k, top_k * 2))
            .to_list()
        )
        for r in results:
            defaults = self._with_schema_defaults(r)
            r.clear()
            r.update(defaults)
            # LanceDB retourne une _distance (L2 ou cosine selon l'index) ; on la
            # convertit en score de similarité croissant (1 = identique).
            r["similarity_score"] = 1.0 / (1.0 + r.get("_distance", 0.0))
        return [
            row for row in results
            if row.get("content_type") != "table_fact"
            or is_plausible_table_fact(row)
        ][:top_k]

    def count(self) -> int:
        return self.table.count_rows() if self.table is not None else 0


class BM25Store:
    """Index sparse BM25 (mots-clés) — capte sigles, numéros d'articles, noms propres
    que le dense retrieval seul peut manquer."""

    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []
        self.chunk_lookup: dict[str, Chunk] = {}
        self._fact_bm25: BM25Okapi | None = None
        self._fact_row_ids: list[str] = []
        self._fact_rows: dict[str, list[Chunk]] = {}
        self._fact_row_tokens: list[set[str]] = []
        self._fact_groups_by_period: dict[
            str, dict[tuple[str, int | None, str], list[Chunk]]
        ] = {}
        self._chunk_index_by_id: dict[str, int] = {}
        self._chunk_ids_by_url: dict[str, list[str]] = {}

    def _build_fact_index(self) -> None:
        """Un document BM25 par ligne de tableau, pas par cellule annuelle."""
        rows: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in self.chunk_lookup.values():
            if (
                chunk.content_type in {"table_fact", "derived_fact"}
                and chunk.parent_id
                and _chunk_is_retrievable(chunk)
            ):
                rows[chunk.parent_id].append(chunk)
        self._fact_rows = dict(rows)
        self._fact_row_ids = list(self._fact_rows)
        self._fact_row_tokens = []
        period_groups: dict[
            str, dict[tuple[str, int | None, str], list[Chunk]]
        ] = defaultdict(lambda: defaultdict(list))
        for row in self._fact_rows.values():
            for chunk in row:
                group_key = (chunk.url, chunk.page, chunk.period)
                for period_key in period_keys(chunk.period):
                    period_groups[period_key][group_key].append(chunk)
        self._fact_groups_by_period = {
            period: dict(groups) for period, groups in period_groups.items()
        }
        if not self._fact_row_ids:
            self._fact_bm25 = None
            return
        corpus = []
        for row_id in self._fact_row_ids:
            representative = self._fact_rows[row_id][0]
            indicator_profile = _structured_measure_profile(
                representative.indicator,
            )
            tokens = list(
                set(indicator_profile[0])
                | set(_tokenize(representative.domain))
                | set(_tokenize(representative.unit))
                | set(_tokenize(representative.territory))
                | set(_tokenize(representative.measure_type))
            )
            corpus.append(tokens)
            self._fact_row_tokens.append(set(tokens))
        self._fact_bm25 = BM25Okapi(corpus)

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            self.bm25 = None
            self.chunk_ids = []
            self.chunk_lookup = {}
            self._chunk_index_by_id = {}
            self._chunk_ids_by_url = {}
            self._build_fact_index()
            logger.info("Index BM25 vidé : aucun chunk.")
            return
        corpus = [_tokenize(_bm25_document_text(c)) for c in chunks]
        self.bm25 = BM25Okapi(corpus)
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.chunk_lookup = {c.chunk_id: c for c in chunks}
        self._chunk_index_by_id = {
            chunk_id: index for index, chunk_id in enumerate(self.chunk_ids)
        }
        by_url: dict[str, list[str]] = defaultdict(list)
        for chunk in chunks:
            by_url[chunk.url].append(chunk.chunk_id)
        self._chunk_ids_by_url = dict(by_url)
        self._build_fact_index()
        logger.info("Index BM25 construit sur %d chunks.", len(chunks))

    def search(
        self,
        query: str,
        top_k: int,
        *,
        category_filter: str | None = None,
    ) -> list[tuple[str, float]]:
        if self.bm25 is None:
            return []
        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        eligible = [
            index for index, chunk_id in enumerate(self.chunk_ids)
            if _chunk_is_retrievable(self.chunk_lookup[chunk_id])
            and (
                not category_filter
                or self.chunk_lookup[chunk_id].category == category_filter
            )
        ]
        top_idx = sorted(
            eligible, key=lambda index: float(scores[index]), reverse=True,
        )[:top_k]
        results: list[tuple[str, float]] = []
        for index in top_idx:
            lexical_overlap = sum(
                1 for token in query_tokens if token in self.bm25.doc_freqs[index]
            )
            if lexical_overlap <= 0:
                continue
            # Sur un très petit corpus, rank_bm25 peut attribuer un IDF nul ou
            # négatif à tous les termes partagés. Une correspondance lexicale
            # réelle ne doit pas disparaître pour cette seule raison.
            score = max(float(scores[index]), lexical_overlap * 1e-9)
            results.append((self.chunk_ids[index], score))
        return results

    def structured_fact_candidates(
        self,
        query: str,
        top_k: int,
        *,
        category_filter: str | None = None,
    ) -> list[tuple[str, float]]:
        """Rappelle les lignes pertinentes puis choisit leur cellule temporelle.

        Le canal n'est activé que si la question contient une période explicite.
        Il évite que les milliers de cellules annuelles d'un corpus statistique
        saturent les top-k dense et BM25 avant que la bonne ligne soit examinée.
        """
        if self._fact_bm25 is None or top_k <= 0:
            return []
        query_periods = period_keys(query)
        query_years = {
            key for key in query_periods if re.fullmatch(r"(?:19|20)\d{2}", key)
        }
        query_specific_periods = {
            key for key in query_periods
            if key.startswith(("month:", "quarter:"))
        }
        if not query_years and not query_specific_periods:
            return []

        query_tokens = _expanded_structured_tokens(query)
        focus_tokens = _structured_query_focus_tokens(query)
        normalized_query = unicodedata.normalize("NFKD", str(query).casefold())
        normalized_query = "".join(
            char for char in normalized_query if not unicodedata.combining(char)
        )
        wants_change = bool(re.search(
            r"\b(?:ecart|difference|evolu\w*|vari\w*)\b", normalized_query,
        ))
        wants_annual_total = bool(re.search(
            r"\b(?:cumul\w*|total|totale|annuel|annuelle)\b|"
            r"\bensemble\s+de\s+l['’ ]annee\b|\bannee\s+entiere\b",
            normalized_query,
        )) and not wants_change
        scores = self._fact_bm25.get_scores(list(query_tokens))
        row_scores = []
        for index, tokens in enumerate(self._fact_row_tokens):
            lexical_overlap = len(query_tokens & tokens)
            representative = self._fact_rows[self._fact_row_ids[index]][0]
            measure_alignment = structured_measure_alignment(
                query,
                representative.indicator,
                f"{representative.measure_type} {representative.unit}",
            )
            if focus_tokens & tokens:
                measure_alignment += 0.25
            row_scores.append((
                measure_alignment,
                max(0.0, float(scores[index])) + lexical_overlap * 1e-3,
                lexical_overlap,
            ))
        ranked_rows = sorted(
            range(len(self._fact_row_ids)),
            key=lambda index: row_scores[index],
            reverse=True,
        )
        candidates: list[tuple[float, int, Chunk]] = []
        for index in ranked_rows:
            measure_alignment, score, lexical_overlap = row_scores[index]
            if lexical_overlap <= 0:
                continue
            row = self._fact_rows[self._fact_row_ids[index]]
            if category_filter and row[0].category != category_filter:
                continue
            preferred = preferred_period_keys(query)
            for chunk in row:
                fact_periods = period_keys(chunk.period)
                specific_match = bool(query_specific_periods & fact_periods)
                year_match = bool(query_years & fact_periods)
                priority = len(preferred)
                if query_specific_periods:
                    if specific_match:
                        priority = min(
                            (preferred.index(key) for key in preferred if key in fact_periods),
                            default=len(preferred),
                        )
                    else:
                        continue
                elif year_match:
                    priority = min(
                        (preferred.index(key) for key in preferred if key in fact_periods),
                        default=len(preferred),
                    )
                else:
                    continue

                operation_bonus = 0.0
                if wants_change:
                    if chunk.source_method == "derived_quarter_change":
                        operation_bonus += 8.0
                    elif chunk.source_method == "derived_annual_sum":
                        operation_bonus -= 3.0
                elif wants_annual_total:
                    if chunk.source_method == "derived_annual_sum":
                        operation_bonus += 8.0
                    elif chunk.source_method == "derived_quarter_change":
                        operation_bonus -= 3.0

                chunk_alignment = structured_measure_alignment(
                    query,
                    chunk.indicator,
                    f"{chunk.measure_type} {chunk.unit}",
                )
                requested_periods = query_specific_periods or query_years
                period_coverage = (
                    len(requested_periods & fact_periods) / len(requested_periods)
                    if requested_periods else 0.0
                )
                candidate_score = (
                    score
                    + max(0.0, measure_alignment)
                    + max(0.0, chunk_alignment)
                    + operation_bonus
                    + 4.0 * period_coverage
                )
                candidates.append((candidate_score, priority, chunk))

        candidates.sort(key=lambda item: (-item[0], item[1], item[2].chunk_id))
        results: list[tuple[str, float]] = []
        seen: set[str] = set()
        for candidate_score, _priority, chunk in candidates:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            results.append((chunk.chunk_id, candidate_score))
            if len(results) >= top_k:
                break
        return results

    def intra_document_candidates(
        self,
        query: str,
        seed_urls: list[str],
        *,
        per_document: int = 2,
        category_filter: str | None = None,
    ) -> list[tuple[str, float]]:
        """Cherche la meilleure page dans chaque document deja identifie.

        Dense/BM25 trouvent souvent le bon rapport mais une page generique. Ce
        second passage reste local et peu couteux : il reclasse uniquement les
        chunks appartenant aux quelques URL semences, avec les contraintes de
        periode et de type de mesure des faits structures.
        """
        if self.bm25 is None or per_document <= 0:
            return []
        query_tokens = _expanded_structured_tokens(query)
        scores = self.bm25.get_scores(_tokenize(query))
        requested_periods = {
            key for key in period_keys(query)
            if re.fullmatch(r"(?:19|20)\d{2}", key)
            or key.startswith(("month:", "quarter:"))
        }
        candidates: list[tuple[str, float]] = []
        for url in dict.fromkeys(seed_urls):
            ranked: list[tuple[str, float]] = []
            for chunk_id in self._chunk_ids_by_url.get(url, []):
                chunk = self.chunk_lookup[chunk_id]
                if category_filter and chunk.category != category_filter:
                    continue
                if not _chunk_is_retrievable(chunk):
                    continue
                index = self._chunk_index_by_id.get(chunk_id)
                if index is None:
                    continue
                lexical = max(0.0, float(scores[index]))
                overlap = len(query_tokens & set(_tokenize(_bm25_document_text(chunk))))
                score = lexical + overlap * 0.02
                if chunk.content_type in {"table_fact", "derived_fact"}:
                    alignment = structured_measure_alignment(
                        query,
                        chunk.indicator,
                        f"{chunk.measure_type} {chunk.unit}",
                    )
                    score += 1.8 * alignment
                    fact_periods = period_keys(chunk.period)
                    if requested_periods:
                        period_coverage = len(
                            requested_periods & fact_periods
                        ) / len(requested_periods)
                        score += 3.0 * period_coverage
                        score -= 3.0 * (1.0 - period_coverage)
                if score > 0:
                    ranked.append((chunk_id, score))
            ranked.sort(key=lambda item: item[1], reverse=True)
            candidates.extend(ranked[:per_document])
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates

    def structured_ratio_components(
        self,
        query: str,
        *,
        category_filter: str | None = None,
    ) -> list[tuple[str, float]]:
        """Trouve deux cellules coherentes pour une part explicitement demandee.

        Par défaut, les deux grandeurs doivent venir du même tableau. Un
        rapprochement entre documents n'est autorisé que pour la construction
        explicite « en rapportant X à Y », avec période, territoire et unité
        compatibles. Le calcul reste déterministe dans le retriever.
        """
        phrases = _ratio_measure_phrases(query)
        preferred = preferred_period_keys(query)
        if not phrases or not preferred:
            return []
        numerator_phrase, denominator_phrase = phrases
        target_period = preferred[0]
        groups = self._fact_groups_by_period.get(target_period, {})
        normalized_query = unicodedata.normalize("NFKD", str(query).casefold())
        normalized_query = "".join(
            char for char in normalized_query
            if not unicodedata.combining(char)
        )
        allow_cross_document = bool(re.search(r"\ben rapportant\b", normalized_query))
        candidate_groups = (
            [[chunk for chunks in groups.values() for chunk in chunks]]
            if allow_cross_document else list(groups.values())
        )

        best: tuple[float, Chunk, Chunk] | None = None
        for chunks in candidate_groups:
            if category_filter and chunks[0].category != category_filter:
                continue
            numerator_ranked = sorted(
                (
                    (_measure_phrase_score(numerator_phrase, chunk.indicator), chunk)
                    for chunk in chunks
                ),
                key=lambda item: item[0],
                reverse=True,
            )[:4]
            denominator_ranked = sorted(
                (
                    (_measure_phrase_score(denominator_phrase, chunk.indicator), chunk)
                    for chunk in chunks
                ),
                key=lambda item: item[0],
                reverse=True,
            )[:4]
            for numerator_score, numerator in numerator_ranked:
                for denominator_score, denominator in denominator_ranked:
                    if numerator.chunk_id == denominator.chunk_id:
                        continue
                    numerator_unit = _unit_signature(numerator.unit)
                    denominator_unit = _unit_signature(denominator.unit)
                    if not numerator_unit or numerator_unit != denominator_unit:
                        continue
                    pair_score = numerator_score + denominator_score
                    if numerator_score < 0.35 or denominator_score < 0.35:
                        continue
                    if numerator.page == denominator.page:
                        pair_score += 2.0
                    if numerator.url == denominator.url:
                        pair_score += 1.0
                    elif not allow_cross_document:
                        continue
                    if (
                        numerator.territory and denominator.territory
                        and numerator.territory != denominator.territory
                    ):
                        continue
                    if best is None or pair_score > best[0]:
                        best = (pair_score, numerator, denominator)
        if best is None:
            return []
        score, numerator, denominator = best
        return [
            (numerator.chunk_id, score),
            (denominator.chunk_id, score - 1e-6),
        ]

    def rare_query_term_candidates(
        self,
        query: str,
        *,
        max_document_frequency: int,
        max_candidates: int,
        category_filter: str | None = None,
    ) -> tuple[str | None, list[str]]:
        """Rappelle les documents partageant le terme informatif le plus rare.

        Ce mécanisme est destiné aux listes exhaustives : un élément pertinent
        peut être rédigé différemment des autres et rester loin dans le classement
        global, tout en contenant le nom exact de la catégorie demandée.
        """
        if self.bm25 is None or max_candidates <= 0:
            return None, []

        query_tokens = list(dict.fromkeys(_tokenize(query)))
        preferred_tokens = exhaustive_category_tokens(query)
        candidates: list[tuple[int, float, str]] = []
        for token in query_tokens:
            if (
                len(token) < 4
                or token.isdigit()
                or token in _EXHAUSTIVE_ANCHOR_STOPWORDS
                or token not in self.bm25.idf
            ):
                continue
            document_frequency = sum(
                1
                for index, frequencies in enumerate(self.bm25.doc_freqs)
                if token in frequencies
                and (
                    not category_filter
                    or self.chunk_lookup[self.chunk_ids[index]].category
                    == category_filter
                )
            )
            preferred = token in preferred_tokens
            if document_frequency < 2:
                continue
            # Le nom de la collection reste le meilleur ancrage même lorsqu'il
            # est fréquent. Sinon un mot incident de l'historique (« tour »,
            # « tête »...) peut être choisi et forcer des pages hors sujet.
            if not preferred and document_frequency > max_document_frequency:
                continue
            # Le facteur log(df) évite de choisir un mot accidentel présent
            # dans seulement deux pages. On cherche un terme à la fois
            # informatif (IDF élevé) et suffisamment partagé pour décrire une
            # collection. Un nom extrait du patron interrogatif reste prioritaire.
            coverage_score = float(self.bm25.idf[token]) * math.log1p(
                document_frequency
            )
            candidates.append((1 if preferred else 0, coverage_score, token))
        if not candidates:
            return None, []

        _preferred, _coverage_score, anchor = max(candidates)
        query_scores = self.bm25.get_scores(_tokenize(query))
        matching_indices = [
            index for index, frequencies in enumerate(self.bm25.doc_freqs)
            if anchor in frequencies
            and (
                not category_filter
                or self.chunk_lookup[self.chunk_ids[index]].category
                == category_filter
            )
        ]
        matching_indices.sort(
            key=lambda index: float(query_scores[index]), reverse=True,
        )
        matching_ids = [self.chunk_ids[index] for index in matching_indices]
        return anchor, matching_ids[:max_candidates]

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25, "chunk_ids": self.chunk_ids, "chunk_lookup": self.chunk_lookup,
                "tokenizer_version": _BM25_TOKENIZER_VERSION,
            }, f)

    def load(self) -> bool:
        if not self.index_path.exists():
            return False
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
        if data.get("tokenizer_version") != _BM25_TOKENIZER_VERSION:
            logger.info(
                "Version BM25 obsolète détectée ; reconstruction automatique avec le tokenizer v%d.",
                _BM25_TOKENIZER_VERSION,
            )
            self.build(list(data.get("chunk_lookup", {}).values()))
            self.save()
            return True
        self.bm25, self.chunk_ids, self.chunk_lookup = data["bm25"], data["chunk_ids"], data["chunk_lookup"]
        self._build_fact_index()
        return True


class VectorStoreManager:
    """Façade orchestrant LanceDB + BM25, avec réindexation incrémentale par URL."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.dense = LanceDBStore(config)
        self.sparse = BM25Store(config.vectorstore.bm25_index_full_path)

    def index_all(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """(Ré)indexation complète — utilisée pour le premier build du corpus."""
        self.dense.create(chunks, vectors)
        self.sparse.build(chunks)
        self.sparse.save()

    def reindex_incremental(self, changed_urls: list[str], new_chunks: list[Chunk],
                             new_vectors: np.ndarray, all_chunks: list[Chunk]) -> None:
        """Supprime les chunks des URLs modifiées puis réinsère leurs nouveaux chunks.

        `all_chunks` (corpus complet à jour) est nécessaire pour reconstruire l'index
        BM25 en entier (voir note de conception en tête de fichier).
        """
        self.dense.delete_by_urls(changed_urls)
        self.dense.add(new_chunks, new_vectors)
        self.sparse.build(all_chunks)
        self.sparse.save()

    @staticmethod
    def _fingerprints_by_url(rows: list[dict]) -> dict[str, tuple[str, ...]]:
        fingerprints: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            fingerprints[row.get("url", "")].append(_content_fingerprint(row))
        return {
            url: tuple(sorted(values))
            for url, values in fingerprints.items()
        }

    def sync_incremental(
        self,
        all_chunks: list[Chunk],
        embedding_model: "EmbeddingModel",
    ) -> IncrementalIndexReport:
        """Synchronise LanceDB avec le corpus courant en ne réencodant que les URL modifiées.

        La comparaison porte sur le texte enfant, le parent et toutes les métadonnées
        utilisées par le retrieval. Une modification après les 64 premiers caractères
        est donc détectée même si l'identifiant historique du chunk reste identique.
        """
        existing_rows = self.dense.metadata_rows()
        schema_upgrade = not self.dense.schema_is_current()
        current_rows = [asdict(chunk) for chunk in all_chunks]
        existing_by_url = self._fingerprints_by_url(existing_rows)
        current_by_url = self._fingerprints_by_url(current_rows)
        changed_urls = sorted(
            url
            for url in set(existing_by_url) | set(current_by_url)
            if existing_by_url.get(url) != current_by_url.get(url)
        )
        if schema_upgrade:
            changed_urls = sorted(current_by_url)

        if not changed_urls:
            if not self.sparse.load():
                self.sparse.build(all_chunks)
                self.sparse.save()
            return IncrementalIndexReport(
                changed_urls=0,
                embedded_chunks=0,
                reused_chunks=len(all_chunks),
                removed_chunks=0,
                total_chunks=len(all_chunks),
                embedding_seconds=0.0,
            )

        changed_set = set(changed_urls)
        new_chunks = [chunk for chunk in all_chunks if chunk.url in changed_set]
        removed_chunks = sum(
            1 for row in existing_rows if row.get("url", "") in changed_set
        )

        embedding_started = time.perf_counter()
        vector_batches = []
        batch_size = self.config.embeddings.batch_size
        for start in range(0, len(new_chunks), batch_size):
            texts = [chunk.text for chunk in new_chunks[start:start + batch_size]]
            vector_batches.append(embedding_model.encode(texts))
        embedding_seconds = time.perf_counter() - embedding_started
        vectors = (
            np.concatenate(vector_batches, axis=0)
            if vector_batches
            else np.empty((0, embedding_model.dim or 0), dtype=np.float32)
        )

        # Les embeddings sont tous calculés avant la première écriture : un échec
        # d'encodage ne peut donc pas laisser l'index dense partiellement supprimé.
        if schema_upgrade:
            self.dense.create(new_chunks, vectors)
        else:
            self.dense.delete_by_urls(changed_urls)
            if new_chunks:
                self.dense.add(new_chunks, vectors)
        self.sparse.build(all_chunks)
        self.sparse.save()

        dense_count = self.dense.count()
        if dense_count != len(all_chunks):
            raise RuntimeError(
                "Synchronisation incohérente : "
                f"LanceDB contient {dense_count} chunks au lieu de {len(all_chunks)}."
            )

        report = IncrementalIndexReport(
            changed_urls=len(changed_urls),
            embedded_chunks=len(new_chunks),
            reused_chunks=len(all_chunks) - len(new_chunks),
            removed_chunks=removed_chunks,
            total_chunks=len(all_chunks),
            embedding_seconds=embedding_seconds,
        )
        logger.info("Synchronisation incrémentale terminée : %s", report)
        return report

    def load_sparse(self) -> bool:
        return self.sparse.load()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    manager = VectorStoreManager(config)
    logger.info("Vecteurs indexés : %d", manager.dense.count())


if __name__ == "__main__":
    main()
