"""Nettoyage des pages scrapées : suppression du menu/header/footer/navigation
et du texte répétitif, pour ne conserver que le contenu éditorial utile.

Deux passes :
1. Nettoyage par page : trafilatura (extraction du contenu principal à partir
   du HTML brut) complété par un filtrage de lignes de bruit connues du thème
   WordPress d'ENSEA (Avada) qui échappent parfois à trafilatura (ex: "About
   the Author", "Share This Post!", widgets de sidebar).
2. Déduplication inter-pages : toute ligne de texte identique retrouvée sur
   plus de `boilerplate_dedup_threshold` des pages (mentions légales, liens de
   réseaux sociaux répétés, etc.) est considérée comme du boilerplate global
   et supprimée de tous les documents.

Sortie : data/cleaned/<hash>.md, un Markdown avec front-matter YAML de
métadonnées (url, title, section, doc_type, depth, date, category).
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import frontmatter
import trafilatura

from config import Config, load_config
from scraper import _is_enqueueable, url_hash
from table_facts import deduplicate_table_facts, extract_matrix_table_facts
from text_quality import is_placeholder_line

logger = logging.getLogger(__name__)

# Lignes de bruit récurrentes dans le thème WordPress (Avada) d'ENSEA que
# trafilatura laisse parfois passer (widgets de sidebar, méta d'auteur, etc.).
NOISE_LINE_PATTERNS = [
    re.compile(r"^about the author\s*:", re.IGNORECASE),
    re.compile(r"^share this post", re.IGNORECASE),
    re.compile(r"^search$", re.IGNORECASE),
    re.compile(r"^recent posts?$", re.IGNORECASE),
    re.compile(r"^articles? similaires?$", re.IGNORECASE),
    re.compile(r"^laisser un commentaire", re.IGNORECASE),
    re.compile(r"^\d+\s+commentaires?$", re.IGNORECASE),
    re.compile(r"^tags?\s*:", re.IGNORECASE),
    re.compile(r"^catégories?\s*:", re.IGNORECASE),
    re.compile(r"^ce site est enregistré sur wpml\.org", re.IGNORECASE),
    re.compile(r"^[a-z][a-z0-9_-]*\.[a-z][a-z0-9_.-]*$", re.IGNORECASE),
    re.compile(r"^\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\+00:00|z)?$", re.IGNORECASE),
]


@dataclass
class CleanedDocument:
    doc_id: str
    url: str
    title: str
    markdown: str
    doc_type: str
    depth: int
    category: str
    date: str | None
    section_titles: list[str]
    extraction_method: str = "unknown"
    extraction_diagnostics: dict | None = None
    table_facts: list[dict] | None = None


def _derive_category(url: str, breadcrumbs: list[str]) -> str:
    if breadcrumbs:
        return breadcrumbs[0].strip().lower().replace(" ", "-")
    path = urlparse(url).path.strip("/")
    if not path:
        return "accueil"
    return path.split("/")[0]


def _derive_depth(url: str) -> int:
    path = urlparse(url).path.strip("/")
    return len([seg for seg in path.split("/") if seg]) if path else 0


def _strip_noise_lines(text: str) -> str:
    kept = []
    previous_nonempty = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if any(pattern.match(stripped) for pattern in NOISE_LINE_PATTERNS):
            continue
        if is_placeholder_line(stripped):
            continue
        # Les cartes WordPress répètent fréquemment le nom/titre dans plusieurs
        # sous-balises imbriquées. Une répétition immédiatement consécutive ne
        # porte aucune information supplémentaire et dégrade BM25.
        if stripped == previous_nonempty:
            continue
        kept.append(line)
        previous_nonempty = stripped
    return "\n".join(kept)


def _normalized_length(text: str) -> int:
    """Longueur informative, indépendante des espaces ajoutés par le HTML."""
    return len(re.sub(r"\s+", " ", text).strip())


def _is_archive_like(record: dict) -> bool:
    """Repère les listes de navigation qui dupliquent les documents finaux."""
    path = urlparse(record.get("url", "")).path.casefold().rstrip("/")
    title = (record.get("title") or "").casefold().strip()
    if re.search(r"/page/\d+$", path):
        return True
    segments = {segment for segment in path.split("/") if segment}
    if segments & {
        "author", "user", "tag", "category", "document-category",
        "sandbox", "element_category",
    }:
        return True
    archive_title_prefixes = (
        "archives des ", "archive des ", "portfolio archive", "faqs archive",
        "actualité - ensea", "news - ensea", "etudes et recherche - page ",
        "studies and research - page ",
        "sandbox - ensea",
    )
    return title.startswith(archive_title_prefixes)


def _choose_html_extraction(record: dict) -> tuple[str, str, dict]:
    """Choisit l'extraction la plus informative sans réintroduire tout le thème.

    Trafilatura est excellent pour les articles rédigés, mais il ignore parfois
    les grilles de cartes WordPress (doctorants, partenaires, équipes). Le texte
    ``record['text']`` provient déjà du seul ``main/#content`` après suppression
    du header, footer, formulaires et scripts. Il constitue donc un repli sûr si
    son gain d'information est très net.
    """
    url = record.get("url", "")
    precise = trafilatura.extract(
        record.get("raw_html", ""),
        url=url,
        output_format="markdown",
        include_tables=True,
        include_formatting=True,
        include_links=False,
        favor_precision=True,
    ) or ""
    precise = _strip_noise_lines(precise).strip()
    structured = _strip_noise_lines(record.get("text", "")).strip()
    structured = _restore_structured_lists(
        structured, record.get("lists", []),
    )
    precise_chars = _normalized_length(precise)
    structured_chars = _normalized_length(structured)

    # Le seuil absolu empêche de choisir le texte structuré pour quelques mots
    # supplémentaires ; le ratio protège les articles déjà bien extraits.
    archive_like = _is_archive_like(record)
    structured_is_materially_richer = (
        not archive_like
        and
        structured_chars >= 800
        and structured_chars >= precise_chars + 600
        and structured_chars >= max(1, precise_chars) * 1.8
    )
    if not precise and structured and not archive_like:
        method = "structured_main_fallback"
        selected = structured
    elif structured_is_materially_richer:
        method = "structured_main_richer"
        selected = structured
    else:
        method = "trafilatura_precision"
        selected = precise or (structured if not archive_like else "")

    diagnostics = {
        "trafilatura_chars": precise_chars,
        "structured_main_chars": structured_chars,
        "selected_chars": _normalized_length(selected),
        "archive_like": archive_like,
    }
    return selected, method, diagnostics


def _restore_structured_lists(
    text: str,
    extracted_lists: list[list[str]],
) -> str:
    """Rétablit les marqueurs Markdown perdus lors de l'extraction HTML.

    Le scraper conserve déjà chaque ``ul``/``ol`` comme une liste distincte.
    Sans marqueurs, deux listes voisines deviennent un bloc ambigu et le LLM
    peut confondre leurs types. Le remplacement reste général et ne modifie
    que les suites d'items retrouvées exactement dans le texte principal.
    """
    restored = text
    for items in extracted_lists:
        cleaned_items = [
            " ".join(str(item).split()) for item in items
            if str(item).strip()
        ]
        if len(cleaned_items) < 2:
            continue
        plain_block = "\n".join(cleaned_items)
        if plain_block not in restored:
            continue
        markdown_block = "\n".join(f"- {item}" for item in cleaned_items)
        restored = restored.replace(
            plain_block, f"\n{markdown_block}\n", 1,
        )
    return re.sub(r"\n{3,}", "\n\n", restored).strip()


class HTMLCleaner:
    """Nettoie l'ensemble des pages de data/raw/ vers data/cleaned/."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.raw_dir = config.project.data_path / "raw"
        self.cleaned_dir = config.project.data_path / "cleaned"
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)

    def _load_raw_records(self) -> list[dict]:
        records = []
        for path in sorted(self.raw_dir.glob("*.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError as exc:
                logger.warning("JSON illisible ignoré : %s (%s)", path, exc)
        return records

    def _clean_single(self, record: dict) -> CleanedDocument | None:
        url = record["url"]
        source_doc_type = record.get("metadata", {}).get("doc_type", "page")
        is_local_document = source_doc_type in {"pdf", "spreadsheet"}
        # Une archive duplique des extraits de documents déjà indexés et peut
        # artificiellement dominer BM25/reranking. Les pages finales restent
        # conservées ; seules les listes de navigation sont écartées.
        if (not is_local_document and not _is_enqueueable(url)) or _is_archive_like(record):
            return None

        if is_local_document:
            body_markdown = record["text"].strip()
            extraction_method = (
                "pdf_text" if source_doc_type == "pdf" else "spreadsheet_cells"
            )
            extraction_diagnostics = {"selected_chars": _normalized_length(body_markdown)}
        else:
            body_markdown, extraction_method, extraction_diagnostics = (
                _choose_html_extraction(record)
            )

        body_markdown = re.sub(r"\n{3,}", "\n\n", body_markdown).strip()
        if len(body_markdown) < self.config.cleaning.min_content_length:
            return None

        title = record.get("title") or record.get("metadata", {}).get("page_title") or url
        section_titles = [h["text"] for h in record.get("headings", [])]
        breadcrumbs = record.get("breadcrumbs", [])
        table_facts = list(record.get("table_facts", []))
        for table_number, table in enumerate(record.get("tables", []), start=1):
            # Les pages HTML disposent deja de matrices de cellules. Leur
            # conversion utilise exactement le meme contrat que les PDF.
            if isinstance(table, list):
                table_facts.extend(extract_matrix_table_facts(
                    table,
                    page_number=0,
                    table_number=table_number,
                ))
        table_facts = deduplicate_table_facts(table_facts)
        extraction_diagnostics = dict(extraction_diagnostics or {})
        extraction_diagnostics["table_facts"] = len(table_facts)

        return CleanedDocument(
            doc_id=url_hash(url),
            url=url,
            title=title,
            markdown=body_markdown,
            doc_type=source_doc_type if is_local_document else "page",
            depth=_derive_depth(url),
            category=_derive_category(url, breadcrumbs),
            date=record.get("modified_date") or record.get("published_date"),
            section_titles=section_titles,
            extraction_method=extraction_method,
            extraction_diagnostics=extraction_diagnostics,
            table_facts=table_facts,
        )

    @staticmethod
    def _compute_global_boilerplate(docs: list[CleanedDocument], threshold: float) -> set[str]:
        """Détecte les lignes présentes sur une proportion trop élevée de pages."""
        line_page_counter: Counter[str] = Counter()
        for doc in docs:
            unique_lines = {line.strip() for line in doc.markdown.splitlines() if line.strip()}
            for line in unique_lines:
                if len(line) > 15:  # les lignes très courtes (titres génériques) ne comptent pas
                    line_page_counter[line] += 1

        n_docs = max(len(docs), 1)
        return {
            line for line, count in line_page_counter.items()
            if count / n_docs >= threshold
        }

    @staticmethod
    def _remove_lines(markdown: str, lines_to_remove: set[str]) -> str:
        kept = [line for line in markdown.splitlines() if line.strip() not in lines_to_remove]
        return "\n".join(kept).strip()

    def clean_all(self, *, clear_output: bool = False) -> list[CleanedDocument]:
        # Le dossier peut être redirigé par un notebook d'expérience après
        # l'initialisation du cleaner. Sa création appartient donc à l'opération
        # de nettoyage elle-même, pas uniquement au constructeur.
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        if clear_output:
            for path in self.cleaned_dir.glob("*.md"):
                path.unlink()
        records = self._load_raw_records()
        logger.info("%d documents bruts à nettoyer.", len(records))

        docs = [d for d in (self._clean_single(r) for r in records) if d is not None]
        logger.info("%d documents conservés après filtrage de longueur minimale.", len(docs))

        boilerplate = self._compute_global_boilerplate(
            docs, self.config.cleaning.boilerplate_dedup_threshold
        )
        logger.info("%d lignes de boilerplate global détectées et supprimées.", len(boilerplate))

        final_docs = []
        for doc in docs:
            doc.markdown = self._remove_lines(doc.markdown, boilerplate)
            if len(doc.markdown) < self.config.cleaning.min_content_length:
                continue
            self._save(doc)
            final_docs.append(doc)

        method_counts = Counter(doc.extraction_method for doc in final_docs)
        logger.info("Méthodes d'extraction retenues : %s", dict(method_counts))
        logger.info("%d documents nettoyés écrits dans %s.", len(final_docs), self.cleaned_dir)
        return final_docs

    def _save(self, doc: CleanedDocument) -> None:
        post = frontmatter.Post(doc.markdown)
        post["url"] = doc.url
        post["title"] = doc.title
        post["doc_type"] = doc.doc_type
        post["depth"] = doc.depth
        post["category"] = doc.category
        post["date"] = doc.date
        post["section_titles"] = doc.section_titles
        post["extraction_method"] = doc.extraction_method
        post["extraction_diagnostics"] = doc.extraction_diagnostics or {}
        post["table_facts"] = doc.table_facts or []

        out_path = self.cleaned_dir / f"{doc.doc_id}.md"
        out_path.write_text(frontmatter.dumps(post), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    cleaner = HTMLCleaner(config)
    cleaner.clean_all()


if __name__ == "__main__":
    main()
