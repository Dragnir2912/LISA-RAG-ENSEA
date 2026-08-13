"""Chunking hiérarchique moderne des documents nettoyés.

Stratégie retenue (voir notebook de comparaison pour la justification chiffrée) :
    Markdown Header Split (ancrage sur la structure H1>H2>H3 du site)
        -> Recursive Character Split en filet de sécurité (sections trop longues)
        -> Parent/Child : un "parent" par section (contexte complet, ~1800 tokens)
           découpé en "children" plus petits (~400 tokens, chevauchement ~15%)
           utilisés pour le retrieval dense/sparse ; le parent est renvoyé au LLM.

Ce module expose aussi des implémentations de référence pour les 3 autres
stratégies comparées à l'étape 3 (recursive seule, markdown header seule,
semantic chunking) — utilisées uniquement dans le notebook de comparaison,
jamais dans le pipeline de production.

Sortie : data/chunks/chunks.jsonl (1 ligne JSON par chunk enfant).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

import frontmatter
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from config import Config, load_config
from table_facts import (
    group_facts_by_row,
    is_plausible_table_fact,
    is_plausible_table_row,
    normalize_numeric_value,
)
from text_quality import has_substantive_body, strip_placeholder_lines

logger = logging.getLogger(__name__)

# Approximation caractères -> tokens retenue après comparaison sur le golden set.
# À 400/1 800 tokens stricts, le corpus passait de 4 459 à 4 942 chunks et le
# Hit@5 chutait de 0,60 à 0,50. Le découpage par caractères préserve mieux les
# unités sémantiques du corpus ENSEA ; les tailles réelles restent mesurées dans
# le notebook pour surveiller les limites du modèle.
CHARS_PER_TOKEN = 4


def _chunk_id(doc_id: str, index: int, text: str) -> str:
    digest = hashlib.sha256(f"{doc_id}:{index}:{text}".encode("utf-8")).hexdigest()[:16]
    return f"{doc_id}_{digest}"


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str
    url: str
    title: str
    section: str
    doc_type: str
    depth: int
    date: str | None
    category: str
    text: str          # texte du chunk enfant (utilisé pour l'embedding/BM25)
    parent_text: str   # texte complet du parent (renvoyé au LLM comme contexte)

    content_type: str = "text"
    page: int | None = None
    table_id: str = ""
    row_id: str = ""
    indicator: str = ""
    period: str = ""
    value: str = ""
    normalized_value: str = ""
    unit: str = ""
    domain: str = ""
    territory: str = ""
    measure_type: str = ""
    qualifier: str = ""
    source_method: str = ""


class HierarchicalChunker:
    """Markdown Header Split -> Recursive fallback -> Parent/Child avec overlap."""

    def __init__(self, config: Config) -> None:
        self.config = config
        cfg = config.chunking
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[(sym, name) for sym, name in cfg.headers_to_split_on],
            strip_headers=False,
        )
        parent_chars = cfg.parent_chunk_size_tokens * CHARS_PER_TOKEN
        child_chars = cfg.child_chunk_size_tokens * CHARS_PER_TOKEN
        child_overlap_chars = int(child_chars * cfg.chunk_overlap_ratio)

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chars, chunk_overlap=0,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chars, chunk_overlap=child_overlap_chars,
        )

    def _section_path(self, metadata: dict) -> str:
        parts = [metadata[key] for key, _ in self._header_keys() if key in metadata]
        return " > ".join(parts) if parts else ""

    def _header_keys(self):
        return [(name, sym) for sym, name in self.config.chunking.headers_to_split_on]

    @staticmethod
    def _page_from_section(section_path: str) -> int | None:
        match = re.search(r"\bPage\s+(\d+)\b", section_path, flags=re.IGNORECASE)
        return int(match.group(1)) if match else None

    @staticmethod
    def _period_sort_key(period: str) -> tuple[int, str]:
        match = re.search(r"\b(?:19|20)\d{2}\b", period or "")
        return (int(match.group(0)) if match else 0, period or "")

    @staticmethod
    def _quarter_and_year(period: str) -> tuple[int, str] | None:
        normalized = str(period).casefold().replace("–", "-").replace("—", "-")
        quarter = re.search(r"\b[qt]\s*([1-4])\b", normalized)
        year = re.search(r"\b(?:19|20)\d{2}\b", normalized)
        if not quarter or not year:
            return None
        return int(quarter.group(1)), year.group(0)

    @staticmethod
    def _decimal_value(fact: dict) -> Decimal | None:
        normalized = str(
            fact.get("normalized_value")
            or normalize_numeric_value(str(fact.get("value", "")))
        ).strip()
        if not normalized:
            return None
        try:
            return Decimal(normalized)
        except InvalidOperation:
            return None

    @staticmethod
    def _format_decimal(value: Decimal, decimals: int = 1) -> str:
        quantum = Decimal(1).scaleb(-decimals)
        rounded = value.quantize(quantum, rounding=ROUND_HALF_UP)
        return format(rounded, "f").replace(".", ",")

    @staticmethod
    def _is_level_series(indicator: str, unit: str) -> bool:
        """Refuse de sommer ou re-deriver des taux, parts et pourcentages."""
        normalized = f"{indicator} {unit}".casefold()
        return not (
            "%" in normalized
            or re.search(
                r"\b(?:indice|part|pourcentage|proportion|ratio|taux|variation)\b",
                normalized,
            )
        )

    @staticmethod
    def _is_additive_series(
        indicator: str,
        unit: str,
        measure_type: str,
        source_title: str = "",
    ) -> bool:
        if not HierarchicalChunker._is_level_series(indicator, unit):
            return False
        normalized_indicator = indicator.casefold()
        if re.search(
            r"\b(?:encours|effectif|emploi|indice|population|solde|stock)\w*\b",
            normalized_indicator,
        ):
            return False
        normalized = f"{unit} {measure_type}".casefold()
        # Une somme annuelle n'a de sens que pour un flux ou un effectif. Une
        # unité reconnue est exigée afin de ne pas additionner arbitrairement
        # des niveaux sans métadonnée exploitable.
        unit_is_additive = bool(re.search(
            r"\b(?:fcfa|xof|usd|euro|milliard|million|millier|nombre|personne|"
            r"habitant|tonne|kg|kilogramme|litre|hectare|volume|valeur|nominal)\w*\b",
            normalized,
        ))
        if not unit_is_additive:
            return False
        # Les comptes nationaux trimestriels sont des flux par construction.
        # Ailleurs, un libellé de flux explicite est exigé pour éviter de sommer
        # quatre photographies successives d'une population ou d'un stock.
        national_accounts = bool(re.search(
            r"\bcomptes?\s+nationaux|\bcomptabilite\s+nationale\b",
            source_title.casefold(),
        ))
        flow_indicator = bool(re.search(
            r"\b(?:agriculture|chiffre d.affaires|consommation|depense|export|"
            r"import|investissement|pib|production|recette|revenu|valeur ajoutee)\w*\b",
            normalized_indicator,
        ))
        return national_accounts or flow_indicator

    def _derived_table_fact_chunks(
        self,
        doc_id: str,
        meta: dict,
        row_id: str,
        ordered: list[dict],
        group_index: int,
    ) -> list[Chunk]:
        """Construit des faits calculés traçables depuis une série trimestrielle.

        Deux opérations générales sont autorisées : le total annuel d'un flux
        additif et la variation en pourcentage entre deux trimestres successifs.
        Chaque fait conserve la page, l'unité, le territoire, le type de mesure
        et la formule complète utilisée.
        """
        if not ordered:
            return []
        first = ordered[0]
        indicator = str(first.get("indicator", "")).strip()
        unit = str(first.get("unit", "")).strip()
        measure_type = str(first.get("measure_type", "")).strip()
        territory = str(first.get("territory", "")).strip()
        domain = str(first.get("domain", "")).strip()
        page = int(first.get("page") or 0) or None
        table_id = str(first.get("table_id", ""))
        by_year: dict[str, dict[int, tuple[dict, Decimal]]] = {}
        for fact in ordered:
            quarter_year = self._quarter_and_year(str(fact.get("period", "")))
            numeric = self._decimal_value(fact)
            if quarter_year is None or numeric is None:
                continue
            quarter, year = quarter_year
            by_year.setdefault(year, {})[quarter] = (fact, numeric)

        chunks: list[Chunk] = []
        derived_index = 0
        for year, quarterly in sorted(by_year.items()):
            if len(quarterly) < 2:
                continue

            def append_derived(
                *, operation: str, derived_indicator: str, period: str,
                value: str, derived_unit: str, formula: str,
            ) -> None:
                nonlocal derived_index
                parent_id = _chunk_id(
                    doc_id,
                    1_100_000 + group_index * 16 + derived_index,
                    f"{row_id}:{operation}:{period}",
                )
                section = meta.get("title", "")
                if page:
                    section += f" > Page {page}"
                section += f" > {derived_indicator}"
                facets = [
                    "Fait statistique calculé et vérifiable",
                    f"Indicateur : {derived_indicator}",
                    f"Période : {period}",
                    f"Valeur : {value}" + (f" {derived_unit}" if derived_unit else ""),
                    f"Formule : {formula}",
                ]
                if domain:
                    facets.append(f"Domaine : {domain}")
                if territory:
                    facets.append(f"Territoire : {territory}")
                if measure_type:
                    facets.append(f"Type de mesure : {measure_type}")
                evidence = "\n".join(facets)
                chunks.append(Chunk(
                    chunk_id=_chunk_id(parent_id, 0, f"{period}:{value}"),
                    parent_id=parent_id,
                    url=meta["url"], title=meta.get("title", ""),
                    section=section, doc_type=meta.get("doc_type", "pdf"),
                    depth=meta.get("depth", 0), date=meta.get("date"),
                    category=meta.get("category", ""), text=" | ".join(facets),
                    parent_text=evidence, content_type="derived_fact", page=page,
                    table_id=table_id, row_id=f"{row_id}:{operation}:{period}",
                    indicator=derived_indicator, period=period, value=value,
                    normalized_value=value.replace(",", "."), unit=derived_unit,
                    domain=domain, territory=territory, measure_type=measure_type,
                    qualifier="calcul déterministe", source_method=operation,
                ))
                derived_index += 1

            if set(quarterly) == {1, 2, 3, 4} and self._is_additive_series(
                indicator, unit, measure_type, str(meta.get("title", "")),
            ):
                total = sum((quarterly[q][1] for q in range(1, 5)), Decimal(0))
                values = [str(quarterly[q][0].get("value", "")) for q in range(1, 5)]
                append_derived(
                    operation="derived_annual_sum",
                    derived_indicator=f"Total annuel de {indicator}",
                    period=year,
                    value=self._format_decimal(total),
                    derived_unit=unit,
                    formula=" + ".join(values) + f" = {self._format_decimal(total)}",
                )

            if self._is_level_series(indicator, unit):
                for previous, current in ((1, 2), (2, 3), (3, 4)):
                    if previous not in quarterly or current not in quarterly:
                        continue
                    old = quarterly[previous][1]
                    new = quarterly[current][1]
                    if old == 0:
                        continue
                    variation = (new - old) / old * Decimal(100)
                    old_text = str(quarterly[previous][0].get("value", ""))
                    new_text = str(quarterly[current][0].get("value", ""))
                    variation_text = self._format_decimal(variation)
                    append_derived(
                        operation="derived_quarter_change",
                        derived_indicator=(
                            f"Variation de {indicator} entre T{previous} {year} "
                            f"et T{current} {year}"
                        ),
                        period=f"T{previous} {year} à T{current} {year}",
                        value=variation_text,
                        derived_unit="%",
                        formula=(
                            f"({new_text} - {old_text}) / {old_text} × 100 "
                            f"= {variation_text} %"
                        ),
                    )
        return chunks

    def _table_fact_chunks(self, doc_id: str, meta: dict) -> list[Chunk]:
        """Cree un enfant par cellule et un parent par ligne statistique."""
        facts = [
            fact for fact in meta.get("table_facts", [])
            if (
                fact.get("indicator") and fact.get("period") and fact.get("value")
                and is_plausible_table_fact(fact)
            )
        ]
        chunks: list[Chunk] = []
        for group_index, (row_id, row_facts) in enumerate(
            sorted(group_facts_by_row(facts).items()),
        ):
            if not is_plausible_table_row(row_facts):
                logger.debug("Ligne tabulaire mal formee ignoree : %s", row_id)
                continue
            ordered = sorted(
                row_facts,
                key=lambda fact: self._period_sort_key(str(fact.get("period", ""))),
            )
            first = ordered[0]
            indicator = str(first.get("indicator", "")).strip()
            unit = str(first.get("unit", "")).strip()
            domain = str(first.get("domain", "")).strip()
            territory = str(first.get("territory", "")).strip()
            measure_type = str(first.get("measure_type", "")).strip()
            page = int(first.get("page") or 0) or None
            table_id = str(first.get("table_id", ""))
            parent_id = _chunk_id(
                doc_id,
                900_000 + group_index,
                f"{row_id}:{indicator}",
            )
            observations = "\n".join(
                f"- Periode {fact.get('period')}: {fact.get('value')}"
                + (f" {unit}" if unit else "")
                + (f" {fact.get('qualifier')}" if fact.get("qualifier") else "")
                for fact in ordered
            )
            parent_parts = [
                "Donnee tabulaire structuree.",
                f"Indicateur : {indicator}",
            ]
            if domain:
                parent_parts.append(f"Domaine : {domain}")
            if territory:
                parent_parts.append(f"Territoire : {territory}")
            if measure_type:
                parent_parts.append(f"Type de mesure : {measure_type}")
            if unit:
                parent_parts.append(f"Unite : {unit}")
            if page:
                parent_parts.append(f"Page : {page}")
            parent_parts.append("Observations :\n" + observations)
            parent_text = "\n".join(parent_parts)

            for fact_index, fact in enumerate(ordered):
                period = str(fact.get("period", "")).strip()
                value = str(fact.get("value", "")).strip()
                normalized_value = str(fact.get("normalized_value", "")).strip()
                text_parts = [
                    "Fait statistique exact",
                    f"Indicateur : {indicator}",
                    f"Periode : {period}",
                    f"Valeur : {value}" + (f" {unit}" if unit else ""),
                ]
                if normalized_value and normalized_value != value:
                    text_parts.append(f"Valeur normalisee : {normalized_value}")
                if domain:
                    text_parts.append(f"Domaine : {domain}")
                if territory:
                    text_parts.append(f"Territoire : {territory}")
                if measure_type:
                    text_parts.append(f"Type de mesure : {measure_type}")
                section = meta.get("title", "")
                if page:
                    section += f" > Page {page}"
                section += f" > {indicator}"
                chunks.append(Chunk(
                    chunk_id=_chunk_id(
                        parent_id,
                        fact_index,
                        f"{period}:{normalized_value or value}",
                    ),
                    parent_id=parent_id,
                    url=meta["url"],
                    title=meta.get("title", ""),
                    section=section,
                    doc_type=meta.get("doc_type", "pdf"),
                    depth=meta.get("depth", 0),
                    date=meta.get("date"),
                    category=meta.get("category", ""),
                    text=" | ".join(text_parts),
                    parent_text=parent_text,
                    content_type="table_fact",
                    page=page,
                    table_id=table_id,
                    row_id=row_id,
                    indicator=indicator,
                    period=period,
                    value=value,
                    normalized_value=normalized_value,
                    unit=unit,
                    domain=domain,
                    territory=territory,
                    measure_type=measure_type,
                    qualifier=str(fact.get("qualifier", "")),
                    source_method=str(fact.get("source_method", "")),
                ))
            chunks.extend(self._derived_table_fact_chunks(
                doc_id, meta, row_id, ordered, group_index,
            ))
        return chunks

    def chunk_document(self, doc_id: str, markdown: str, meta: dict) -> list[Chunk]:
        sections = self.header_splitter.split_text(markdown)
        if not sections:
            sections = [type("D", (), {"page_content": markdown, "metadata": {}})()]

        chunks: list[Chunk] = []
        for section_idx, section in enumerate(sections):
            section_text = strip_placeholder_lines(section.page_content)
            if not section_text or not has_substantive_body(section_text):
                continue
            section_path = self._section_path(section.metadata) or meta.get("title", "")

            # Filet de sécurité : une section trop longue est encore découpée (parent).
            parent_texts = self.parent_splitter.split_text(section_text) or [section_text]

            for parent_idx, parent_text in enumerate(parent_texts):
                parent_id = _chunk_id(doc_id, section_idx * 1000 + parent_idx, parent_text[:64])
                child_texts = self.child_splitter.split_text(parent_text) or [parent_text]

                for child_idx, child_text in enumerate(child_texts):
                    chunks.append(Chunk(
                        chunk_id=_chunk_id(parent_id, child_idx, child_text[:64]),
                        parent_id=parent_id,
                        url=meta["url"],
                        title=meta.get("title", ""),
                        section=section_path,
                        doc_type=meta.get("doc_type", "page"),
                        depth=meta.get("depth", 0),
                        date=meta.get("date"),
                        category=meta.get("category", ""),
                        text=child_text.strip(),
                        parent_text=parent_text.strip(),
                        content_type="text",
                        page=self._page_from_section(section_path),
                    ))
        chunks.extend(self._table_fact_chunks(doc_id, meta))
        return chunks

    def chunk_all(
        self,
        cleaned_dir: Path | None = None,
        output_path: Path | None = None,
    ) -> list[Chunk]:
        """Découpe un corpus, avec chemins alternatifs pour les expériences sûres.

        Sans argument, le comportement historique reste strictement identique.
        Les notebooks peuvent fournir un dossier nettoyé et un fichier de sortie
        isolés afin de mesurer une nouvelle stratégie avant promotion.
        """
        cleaned_dir = cleaned_dir or (self.config.project.data_path / "cleaned")
        output_path = output_path or (
            self.config.project.data_path / "chunks" / "chunks.jsonl"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_chunks: list[Chunk] = []
        for path in sorted(cleaned_dir.glob("*.md")):
            post = frontmatter.load(path)
            meta = dict(post.metadata)
            doc_chunks = self.chunk_document(path.stem, post.content, meta)
            all_chunks.extend(doc_chunks)

        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk.__dict__, ensure_ascii=False) + "\n")

        logger.info("%d chunks écrits dans %s (à partir de %d documents).",
                     len(all_chunks), output_path, len(list(cleaned_dir.glob('*.md'))))
        logger.info(
            "Types de chunks : texte=%d, faits tabulaires=%d, faits calcules=%d.",
            sum(chunk.content_type == "text" for chunk in all_chunks),
            sum(chunk.content_type == "table_fact" for chunk in all_chunks),
            sum(chunk.content_type == "derived_fact" for chunk in all_chunks),
        )
        return all_chunks


# ---------------------------------------------------------------------------
# Stratégies de référence pour la comparaison (notebook uniquement, non utilisées
# dans le pipeline de production — voir HierarchicalChunker ci-dessus).
# ---------------------------------------------------------------------------

def chunk_recursive_only(markdown: str, chunk_size_tokens: int = 400, overlap_ratio: float = 0.15) -> list[str]:
    """Recursive character splitting pur, sans respect de la structure sémantique."""
    chars = chunk_size_tokens * CHARS_PER_TOKEN
    splitter = RecursiveCharacterTextSplitter(chunk_size=chars, chunk_overlap=int(chars * overlap_ratio))
    return splitter.split_text(markdown)


def chunk_markdown_header_only(markdown: str, headers_to_split_on: list[list[str]]) -> list[str]:
    """Markdown Header splitting pur, sans étape recursive ni parent/child."""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[(sym, name) for sym, name in headers_to_split_on],
        strip_headers=False,
    )
    return [s.page_content for s in splitter.split_text(markdown)]


def chunk_semantic(markdown: str, embed_fn, breakpoint_percentile: float = 95.0) -> list[str]:
    """Semantic chunking : découpe aux ruptures de similarité entre phrases consécutives.

    `embed_fn` : callable(list[str]) -> np.ndarray, injecté pour éviter de coupler ce
    module de référence à un modèle d'embedding précis (comparaison notebook only).
    """
    import re

    import numpy as np

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", markdown) if s.strip()]
    if len(sentences) < 3:
        return [markdown]

    embeddings = embed_fn(sentences)
    sims = [
        float(np.dot(embeddings[i], embeddings[i + 1]) /
              (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-8))
        for i in range(len(embeddings) - 1)
    ]
    distances = [1 - s for s in sims]
    threshold = float(np.percentile(distances, breakpoint_percentile))

    chunks, current = [], [sentences[0]]
    for i, dist in enumerate(distances):
        if dist >= threshold:
            chunks.append(" ".join(current))
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])
    if current:
        chunks.append(" ".join(current))
    return chunks


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    chunker = HierarchicalChunker(config)
    chunker.chunk_all()


if __name__ == "__main__":
    main()
