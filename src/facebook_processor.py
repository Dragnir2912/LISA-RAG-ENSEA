"""Préparation isolée des posts Facebook pour une expérience RAG.

Les fichiers Apify normalisés sont transformés en un document Markdown par
publication. Les textes sont normalisés Unicode, les répétitions exactes sont
réduites à leur occurrence la plus récente et les messages trop courts sont
écartés. Le corpus institutionnel de production n'est jamais modifié.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import frontmatter

from config import Config, load_config
from facebook_scraper import extract_media_metadata

_KEYCAP_DIGIT_RE = re.compile(r"([0-9#*])\ufe0f?\u20e3")


def _parse_date(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _normalize_text(text: str, normalize_unicode: bool = True) -> str:
    # Facebook emploie souvent les chiffres-émojis (par ex. 2️⃣0️⃣). Sans cette
    # conversion, BM25 obtient les tokens séparés « 2 », « 0 » tandis que la
    # question contient « 20 », ce qui rend des séries comme Dépêche 19/20
    # pratiquement indiscernables.
    text = _KEYCAP_DIGIT_RE.sub(r"\1", text)
    if normalize_unicode:
        text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalized_hash(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).casefold().strip()
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()


def _headline(text: str, fallback: str) -> str:
    first_line = next((line.strip(" #") for line in text.splitlines() if line.strip()), fallback)
    return first_line[:140].rstrip()


class FacebookPostProcessor:
    """Construit un corpus Markdown candidat à partir des snapshots Facebook."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.settings = config.facebook_scraping

    def _load_posts(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.settings.raw_full_path.glob("*_page.jsonl")):
            raw_by_id: dict[str, dict[str, Any]] = {}
            raw_path = path.with_name(f"{path.stem}_raw.json")
            if raw_path.exists():
                raw_items = json.loads(raw_path.read_text(encoding="utf-8"))
                for item in raw_items if isinstance(raw_items, list) else []:
                    raw_id = str(
                        item.get("postId") or item.get("postID")
                        or item.get("id") or item.get("feedbackId") or ""
                    )
                    if raw_id:
                        raw_by_id[raw_id] = item
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL invalide : {path}:{line_number}") from exc
                # Compatibilite avec les collectes anterieures : le snapshot
                # brut contient souvent l'OCR des affiches que l'ancien JSONL
                # normalise avait omis.
                if not row.get("media_ocr"):
                    raw_item = raw_by_id.get(str(row.get("post_id") or ""))
                    if raw_item:
                        media_ocr, media_urls = extract_media_metadata(raw_item)
                        row["media_ocr"] = media_ocr
                        row["media_urls"] = media_urls
                row["_input_file"] = path.name
                rows.append(row)
        return rows

    def _select_posts(self, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter]:
        counters: Counter = Counter(raw_posts=len(rows))
        eligible: list[dict[str, Any]] = []
        for row in rows:
            if row.get("media_urls"):
                counters["posts_with_media"] += 1
            text = _normalize_text(
                str(row.get("text") or ""),
                normalize_unicode=self.settings.normalize_unicode,
            )
            media_ocr = [
                _normalize_text(str(item), self.settings.normalize_unicode)
                for item in (row.get("media_ocr") or [])
                if str(item).strip()
            ]
            media_ocr = list(dict.fromkeys(
                item for item in media_ocr if item.casefold() not in text.casefold()
            ))
            if media_ocr:
                counters["posts_with_usable_media_ocr"] += 1
                counters["usable_media_ocr_fragments"] += len(media_ocr)
            elif row.get("media_urls"):
                counters["media_posts_without_usable_ocr"] += 1
            if media_ocr:
                text = (
                    f"{text}\n\n## Texte detecte dans les images\n"
                    + "\n".join(f"- {item}" for item in media_ocr)
                )
            if len(text) < self.settings.min_rag_characters:
                counters["removed_too_short"] += 1
                continue
            if not row.get("post_url") or not row.get("published_at"):
                counters["removed_missing_provenance"] += 1
                continue
            row = dict(row)
            row["clean_text"] = text
            row["media_ocr_count"] = len(media_ocr)
            row["normalized_content_hash"] = _normalized_hash(text)
            eligible.append(row)

        if self.settings.deduplicate_content:
            # Le tri décroissant garantit que la première occurrence conservée
            # est la republication la plus récente.
            eligible.sort(key=lambda row: _parse_date(row.get("published_at")), reverse=True)
            unique: dict[str, dict[str, Any]] = {}
            for row in eligible:
                key = row["normalized_content_hash"]
                if key in unique:
                    counters["removed_duplicate_content"] += 1
                else:
                    unique[key] = row
            eligible = list(unique.values())

        eligible.sort(key=lambda row: (row.get("page_key", ""), _parse_date(row.get("published_at"))), reverse=True)
        counters["selected_posts"] = len(eligible)
        return eligible, counters

    def _write_document(self, row: dict[str, Any], output_dir: Path) -> Path:
        page_key = str(row.get("page_key") or "facebook")
        author = str(row.get("author") or page_key)
        text = row["clean_text"]
        title = f"{author} — {_headline(text, 'Publication Facebook')}"
        post = frontmatter.Post(f"# {title}\n\n{text}")
        post["url"] = row["post_url"]
        post["title"] = title
        post["doc_type"] = "facebook_post"
        post["depth"] = 0
        post["category"] = f"facebook/{page_key}"
        post["date"] = row["published_at"]
        post["source"] = "facebook"
        post["page_key"] = page_key
        post["author"] = author
        post["content_hash"] = row["normalized_content_hash"]
        post["likes"] = int(row.get("likes") or 0)
        post["comments"] = int(row.get("comments") or 0)
        post["shares"] = int(row.get("shares") or 0)
        post["media_ocr_count"] = int(row.get("media_ocr_count") or 0)
        post["extraction_method"] = "apify_facebook_posts_and_media_ocr"
        filename = f"facebook_{row['post_id']}.md"
        path = output_dir / filename
        path.write_text(frontmatter.dumps(post), encoding="utf-8")
        return path

    def process_all(self) -> dict[str, Any]:
        rows = self._load_posts()
        selected, counters = self._select_posts(rows)
        output_dir = self.settings.cleaned_candidate_full_path.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Ce répertoire est exclusivement un artefact candidat reproductible à
        # partir des JSONL ; retirer ses anciens Markdown évite les fichiers périmés.
        for old_path in output_dir.glob("facebook_*.md"):
            old_path.unlink()

        paths = [self._write_document(row, output_dir) for row in selected]
        per_page = Counter(row.get("page_key", "unknown") for row in selected)
        report = {
            **dict(counters),
            "per_page": dict(sorted(per_page.items())),
            "cleaned_candidate_dir": str(output_dir),
            "written_documents": len(paths),
            "production_cleaned_dir": str((self.config.project.data_path / "cleaned").resolve()),
        }
        report_path = output_dir.parent / "facebook_processing_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report


def main() -> None:
    report = FacebookPostProcessor(load_config()).process_all()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
