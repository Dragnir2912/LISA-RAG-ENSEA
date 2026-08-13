"""Collecte Apify des publications Facebook des pages ENSEA.

Le jeton est reçu par le constructeur ou lu depuis ``APIFY_TOKEN``. Il n'est
jamais enregistré dans les fichiers de sortie. Les données restent sous
``data/facebook/raw`` jusqu'à leur validation avant toute intégration au RAG.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from config import Config, load_config

logger = logging.getLogger(__name__)
SPONSORED_LABELS = ("sponsorisé", "sponsored")


@dataclass(frozen=True)
class FacebookPost:
    source: str
    page_key: str
    page_url: str
    post_url: str | None
    post_id: str
    author: str
    text: str
    published_at: str | None
    likes: int
    comments: int
    shares: int
    media_ocr: list[str]
    media_urls: list[str]
    scraped_at: str
    content_hash: str


@dataclass(frozen=True)
class FacebookPageReport:
    page_key: str
    page_url: str
    requested_posts: int
    raw_items: int
    unique_posts: int
    skipped_empty_or_short: int
    actor_run_id: str | None
    dataset_id: str | None
    normalized_path: str | None
    raw_path: str | None
    warning: str | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _first(item: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = item.get(key)
        if value is not None and value != "":
            return value
    return default


def _run_field(run: Any, camel_name: str, snake_name: str) -> Any:
    """Lit un champ des anciennes réponses dict et des modèles Apify récents."""
    if isinstance(run, dict):
        return run.get(camel_name) or run.get(snake_name)
    value = getattr(run, snake_name, None) or getattr(run, camel_name, None)
    if value is not None:
        return value
    model_dump = getattr(run, "model_dump", None)
    if callable(model_dump):
        payload = model_dump(by_alias=True)
        return payload.get(camel_name) or payload.get(snake_name)
    return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(_first(value, "name", "title", "text", default="")).strip()
    return str(value).strip()


def _as_count(value: Any) -> int:
    try:
        return max(0, int(float(value or 0)))
    except (TypeError, ValueError):
        return 0


def _as_iso_date(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().isdigit()):
        timestamp = float(value)
        if timestamp > 10_000_000_000:  # certains exports utilisent les millisecondes
            timestamp /= 1000
        try:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        except (OSError, OverflowError, ValueError):
            return None
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
    except ValueError:
        return text or None


def _looks_sponsored(text: str) -> bool:
    prefix = text.casefold()[:250]
    return any(label in prefix for label in SPONSORED_LABELS)


def _content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _walk_nested(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_nested(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_nested(child)


def _clean_media_ocr(value: Any) -> str | None:
    text = re.sub(r"\s+", " ", _as_text(value)).strip()
    if not text:
        return None
    quoted = re.search(
        r"(?:text|texte)\s+(?:that\s+)?says?\s+['\u2019\"](.+?)['\u2019\"]\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if quoted:
        text = quoted.group(1).strip()
    normalized = text.casefold().rstrip(".")
    generic_descriptions = (
        "no photo description available",
        "may be an image of",
        "may be a graphic of",
        "peut etre une image de",
        "peut \u00eatre une image de",
    )
    if any(normalized.startswith(prefix) for prefix in generic_descriptions):
        return None
    return text if len(text) >= 3 else None


def extract_media_metadata(item: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Extrait le texte OCR et les images d'un item Apify sans dependre d'un Actor.

    Cette fonction publique est aussi utilisee par le processeur pour enrichir
    les anciens JSONL a partir des snapshots ``*_raw.json`` deja presents.
    """
    ocr_texts: list[str] = []
    media_urls: list[str] = []
    seen_ocr: set[str] = set()
    seen_urls: set[str] = set()
    for node in _walk_nested(item.get("media") or []):
        for key in ("ocrText", "ocr_text", "altText", "alt_text"):
            cleaned = _clean_media_ocr(node.get(key))
            signature = cleaned.casefold() if cleaned else ""
            if cleaned and signature not in seen_ocr:
                seen_ocr.add(signature)
                ocr_texts.append(cleaned)
        for key in ("thumbnail", "uri", "imageUrl", "image_url"):
            url = _as_text(node.get(key))
            if url.startswith(("http://", "https://")) and url not in seen_urls:
                seen_urls.add(url)
                media_urls.append(url)
    return ocr_texts, media_urls


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _save_jsonl(path: Path, posts: list[FacebookPost]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(asdict(post), ensure_ascii=False) + "\n" for post in posts)
    path.write_text(body, encoding="utf-8")


class FacebookScraper:
    """Lance l'Actor Apify page par page et normalise son jeu de données."""

    def __init__(self, config: Config, api_token: str | None = None) -> None:
        self.config = config
        self.settings = config.facebook_scraping
        self.api_token = (api_token or os.getenv("APIFY_TOKEN") or "").strip()
        if not self.api_token:
            raise ValueError(
                "Jeton Apify absent. Saisissez-le avec getpass dans le notebook "
                "ou définissez la variable d'environnement APIFY_TOKEN."
            )
        try:
            from apify_client import ApifyClient
        except ImportError as exc:
            raise RuntimeError(
                "apify-client n'est pas installé. Exécutez `%pip install apify-client>=1.8`, "
                "puis redémarrez le kernel."
            ) from exc
        self.client = ApifyClient(self.api_token)
        self.settings.raw_full_path.mkdir(parents=True, exist_ok=True)

    def _normalize_item(self, item: dict[str, Any], page_key: str, page_url: str) -> FacebookPost | None:
        text = _as_text(_first(item, "text", "message", "description", "caption"))
        media_ocr, media_urls = extract_media_metadata(item)
        searchable_content = "\n".join([text, *media_ocr]).strip()
        if (
            len(searchable_content) < self.settings.min_post_characters
            or _looks_sponsored(text)
        ):
            return None

        post_url = _as_text(_first(item, "postUrl", "postURL", "url", "facebookUrl")) or None
        raw_id = _as_text(_first(item, "postId", "postID", "id", "feedbackId"))
        post_id = raw_id or hashlib.sha256(
            f"{page_key}\n{post_url or text}".encode("utf-8")
        ).hexdigest()[:20]
        author = _as_text(_first(item, "pageName", "authorName", "author", "user", default=page_key))
        published_at = _as_iso_date(_first(item, "timestamp", "time", "date", "publishedAt"))
        return FacebookPost(
            source="facebook",
            page_key=page_key,
            page_url=page_url,
            post_url=post_url,
            post_id=post_id,
            author=author or page_key,
            text=text,
            published_at=published_at,
            likes=_as_count(_first(item, "likes", "likesCount", "reactions", "reactionsCount")),
            comments=_as_count(_first(item, "comments", "commentsCount")),
            shares=_as_count(_first(item, "shares", "sharesCount")),
            media_ocr=media_ocr,
            media_urls=media_urls,
            scraped_at=_utc_now(),
            content_hash=_content_hash(searchable_content),
        )

    def scrape_page(
        self,
        page_key: str,
        *,
        max_posts: int | None = None,
        reuse_run_id: str | None = None,
    ) -> FacebookPageReport:
        if page_key not in self.settings.pages:
            raise KeyError(f"Page Facebook non autorisée : {page_key}")
        requested = int(
            self.settings.max_posts_per_page if max_posts is None else max_posts
        )
        if not 1 <= requested <= 100:
            raise ValueError(
                "Le nombre de publications doit être compris entre 1 et 100 par page."
            )
        page_url = self.settings.pages[page_key]
        run_input = {
            "startUrls": [{"url": page_url}],
            "resultsLimit": requested,
        }
        logger.info("Apify Facebook : %s (%d posts maximum)", page_key, requested)
        try:
            if reuse_run_id:
                logger.info("Réutilisation de l'exécution Apify %s", reuse_run_id)
                run = self.client.run(reuse_run_id).get()
            else:
                run = self.client.actor(self.settings.actor_id).call(run_input=run_input)
            if not run:
                raise RuntimeError("L'Actor n'a renvoyé aucune information d'exécution.")
            dataset_id = _run_field(run, "defaultDatasetId", "default_dataset_id")
            if not dataset_id:
                raise RuntimeError("Aucun dataset Apify n'est associé à cette exécution.")
            raw_items = list(self.client.dataset(dataset_id).list_items(limit=requested).items)
        except Exception as exc:
            raise RuntimeError(
                f"Échec Apify pour {page_key}. Vérifiez le jeton, les crédits du compte "
                f"et l'accès à l'Actor {self.settings.actor_id}. Détail : {exc}"
            ) from exc

        raw_path = self.settings.raw_full_path / f"{page_key}_raw.json"
        normalized_path = self.settings.raw_full_path / f"{page_key}.jsonl"
        _save_json(raw_path, raw_items)

        posts_by_id: dict[str, FacebookPost] = {}
        skipped = 0
        for item in raw_items:
            post = self._normalize_item(item, page_key, page_url)
            if post is None:
                skipped += 1
                continue
            posts_by_id.setdefault(post.post_id, post)
        posts = list(posts_by_id.values())[:requested]
        _save_jsonl(normalized_path, posts)

        warning = None
        if len(posts) < requested:
            warning = f"{len(posts)} posts textuels uniques obtenus sur {requested} demandés."
        return FacebookPageReport(
            page_key=page_key,
            page_url=page_url,
            requested_posts=requested,
            raw_items=len(raw_items),
            unique_posts=len(posts),
            skipped_empty_or_short=skipped,
            actor_run_id=_run_field(run, "id", "id"),
            dataset_id=dataset_id,
            normalized_path=str(normalized_path),
            raw_path=str(raw_path),
            warning=warning,
        )

    def run(
        self,
        *,
        max_posts_per_page: int | None = None,
        page_keys: list[str] | None = None,
        progress_callback: Callable[[int, int, str, str], None] | None = None,
    ) -> list[FacebookPageReport]:
        selected_pages = list(self.settings.pages if page_keys is None else page_keys)
        if not selected_pages:
            raise ValueError("Aucune page Facebook sélectionnée.")
        unknown_pages = [key for key in selected_pages if key not in self.settings.pages]
        if unknown_pages:
            raise KeyError("Pages Facebook non autorisées : " + ", ".join(unknown_pages))

        reports: list[FacebookPageReport] = []
        total = len(selected_pages)
        for index, page_key in enumerate(selected_pages, start=1):
            if progress_callback:
                progress_callback(index - 1, total, page_key, "starting")
            reports.append(
                self.scrape_page(page_key, max_posts=max_posts_per_page)
            )
            if progress_callback:
                progress_callback(index, total, page_key, "completed")
        report_path = self.settings.raw_full_path / "collection_report.json"
        _save_json(report_path, {
            "created_at": _utc_now(),
            "actor_id": self.settings.actor_id,
            "selected_pages": selected_pages,
            "pages": [asdict(report) for report in reports],
            "total_unique_posts": sum(report.unique_posts for report in reports),
        })
        return reports


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    scraper = FacebookScraper(load_config())
    for report in scraper.run():
        print(asdict(report))


if __name__ == "__main__":
    main()
