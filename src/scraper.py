"""Scraping exhaustif du site https://ensea.ed.ci/.

Utilise Crawlee (la librairie open-source d'Apify) en mode 100% local : pas de
compte Apify Cloud requis. Le crawl part du sitemap officiel du site (déjà
confirmé accessible) et complète par un parcours récursif des liens internes,
ce qui couvre aussi les pages non listées dans le sitemap.

Pour chaque page HTML, on extrait uniquement le contenu utile (texte, titres,
listes, tableaux, liens, métadonnées, breadcrumbs, date) et on ignore images,
JS, CSS et éléments publicitaires/tracking. Les PDF rencontrés sont
téléchargés tels quels puis convertis en texte par `PDFExtractor`.

Sortie :
    data/raw/<hash>.json      un fichier par page HTML scrapée
    data/pdf/<hash>.pdf       PDFs téléchargés tels quels
    data/raw/pdf_<hash>.json  texte extrait par PDF + métadonnées
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

import httpx
from bs4 import BeautifulSoup, Tag
from crawlee import ConcurrencySettings, Request
from crawlee.crawlers import BeautifulSoupCrawler, BeautifulSoupCrawlingContext

from config import Config, load_config

logger = logging.getLogger(__name__)

# Balises qui ne contiennent jamais de contenu éditorial utile : on les retire
# avant toute extraction pour ne pas polluer texte/titres/listes/tableaux.
NOISE_TAGS = ["script", "style", "noscript", "iframe", "svg", "form", "img", "picture"]
NOISE_SELECTORS = [
    "header", "footer", "nav",
    ".site-header", ".site-footer", ".widget-area", ".nav-menu",
    "#cookie-notice", ".cookie-banner", ".advertisement", ".ads",
]

# Extensions binaires à ne jamais enqueuer comme pages (images, archives, médias) :
# le site expose de nombreux liens directs vers des images de galerie qui ne sont
# pas des pages HTML et n'ont rien à faire dans le crawl (cf. consigne "exclure
# les images").
_NON_PAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp", ".ico",
    ".zip", ".rar", ".mp4", ".mp3", ".avi", ".mov",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
)


def _is_enqueueable(url: str) -> bool:
    """PDFs sont gérés séparément (téléchargés puis convertis) ; les autres
    ressources binaires ne sont jamais des pages à parser."""
    lowered = url.lower().split("?")[0]
    if lowered.endswith(".pdf"):
        return True
    return not lowered.endswith(_NON_PAGE_EXTENSIONS)


def url_hash(url: str) -> str:
    """Identifiant court et stable pour nommer les fichiers par URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


@dataclass
class ScrapedPage:
    """Représentation structurée d'une page HTML scrapée, prête à sérialiser en JSON."""

    url: str
    title: str
    headings: list[dict]      # [{"level": "h2", "text": "..."}]
    text: str                 # texte brut concaténé (sera nettoyé à l'étape suivante)
    raw_html: str              # HTML original complet, pour un nettoyage trafilatura en aval
    lists: list[list[str]]    # chaque liste = liste d'items texte
    tables: list[list[list[str]]]  # chaque table = liste de lignes = liste de cellules
    links: list[str]
    breadcrumbs: list[str]
    metadata: dict
    published_date: str | None
    modified_date: str | None
    scraped_at: str
    content_hash: str


class SitemapDiscoverer:
    """Découvre l'ensemble des URLs du site à partir du sitemap index WordPress/Yoast."""

    def __init__(self, sitemap_url: str, timeout: float = 20.0) -> None:
        self.sitemap_url = sitemap_url
        self.timeout = timeout

    def discover(self) -> list[str]:
        urls: list[str] = []
        try:
            sub_sitemaps = self._fetch_sitemap_locs(self.sitemap_url)
        except Exception as exc:  # noqa: BLE001 - on log et on retombe sur le crawl récursif
            logger.warning("Impossible de lire le sitemap index (%s) : %s", self.sitemap_url, exc)
            return urls

        # Un sitemap_index pointe vers des sous-sitemaps (post-sitemap.xml, page-sitemap.xml...)
        for loc in sub_sitemaps:
            if loc.endswith(".xml"):
                try:
                    urls.extend(self._fetch_sitemap_locs(loc))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Sous-sitemap illisible (%s) : %s", loc, exc)
            else:
                urls.append(loc)
        return sorted(set(urls))

    def _fetch_sitemap_locs(self, url: str) -> list[str]:
        resp = httpx.get(url, timeout=self.timeout, follow_redirects=True)
        resp.raise_for_status()
        content = resp.content.lstrip(b"\xef\xbb\xbf").lstrip()
        root = ElementTree.fromstring(content)
        # Espace de noms standard des sitemaps ; on l'ignore pour rester tolérant.
        return [
            el.text.strip()
            for el in root.iter()
            if el.tag.endswith("loc") and el.text
        ]


class PDFExtractor:
    """Télécharge un PDF et en extrait le texte, page par page."""

    def __init__(self, pdf_dir: Path, timeout: float = 30.0) -> None:
        self.pdf_dir = pdf_dir
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def download_and_extract(self, pdf_url: str) -> dict | None:
        h = url_hash(pdf_url)
        pdf_path = self.pdf_dir / f"{h}.pdf"
        try:
            if not pdf_path.exists():
                resp = httpx.get(pdf_url, timeout=self.timeout, follow_redirects=True)
                resp.raise_for_status()
                pdf_path.write_bytes(resp.content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Échec du téléchargement du PDF %s : %s", pdf_url, exc)
            return None

        text = self._extract_with_pypdf(pdf_path)
        if not text.strip():
            text = self._extract_with_pdfplumber(pdf_path)
        if not text.strip():
            logger.warning("Aucun texte extractible pour %s (PDF probablement scanné)", pdf_url)
            return None

        return {
            "url": pdf_url,
            "title": pdf_path.stem,
            "headings": [],
            "text": text,
            "raw_html": "",
            "lists": [],
            "tables": [],
            "links": [],
            "breadcrumbs": [],
            "metadata": {"doc_type": "pdf", "local_path": str(pdf_path)},
            "published_date": None,
            "modified_date": None,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }

    @staticmethod
    def _extract_with_pypdf(pdf_path: Path) -> str:
        from pypdf import PdfReader

        try:
            reader = PdfReader(str(pdf_path))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("pypdf a échoué sur %s : %s", pdf_path, exc)
            return ""

    @staticmethod
    def _extract_with_pdfplumber(pdf_path: Path) -> str:
        import pdfplumber

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                return "\n\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("pdfplumber a échoué sur %s : %s", pdf_path, exc)
            return ""


def _extract_breadcrumbs(soup: BeautifulSoup) -> list[str]:
    """Cherche des breadcrumbs en JSON-LD (BreadcrumbList) puis en fallback CSS."""
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        candidates = data if isinstance(data, list) else [data]
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("@type") == "BreadcrumbList":
                items = sorted(candidate.get("itemListElement", []), key=lambda x: x.get("position", 0))
                names = [item.get("name") or item.get("item", {}).get("name") for item in items]
                names = [n for n in names if n]
                if names:
                    return names

    crumb_container = soup.select_one(".breadcrumb, .breadcrumbs, [class*='breadcrumb']")
    if crumb_container:
        return [a.get_text(strip=True) for a in crumb_container.find_all("a")]
    return []


def _extract_dates(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    published = None
    modified = None
    for meta in soup.find_all("meta"):
        prop = meta.get("property") or meta.get("name")
        if prop == "article:published_time":
            published = meta.get("content")
        elif prop == "article:modified_time":
            modified = meta.get("content")
    return published, modified


def _extract_metadata(soup: BeautifulSoup) -> dict:
    metadata: dict = {}
    if soup.title:
        metadata["page_title"] = soup.title.get_text(strip=True)
    for meta in soup.find_all("meta"):
        name = meta.get("name") or meta.get("property")
        if name in {"description", "og:title", "og:description", "og:type"}:
            metadata[name] = meta.get("content")
    return metadata


def _clean_soup_for_extraction(soup: BeautifulSoup) -> BeautifulSoup:
    for tag_name in NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    for selector in NOISE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()
    return soup


def _extract_headings(content_root: Tag) -> list[dict]:
    headings = []
    for level in range(1, 7):
        for tag in content_root.find_all(f"h{level}"):
            text = tag.get_text(strip=True)
            if text:
                headings.append({"level": f"h{level}", "text": text})
    return headings


def _extract_lists(content_root: Tag) -> list[list[str]]:
    lists = []
    for list_tag in content_root.find_all(["ul", "ol"]):
        items = [li.get_text(strip=True) for li in list_tag.find_all("li", recursive=False)]
        items = [i for i in items if i]
        if items:
            lists.append(items)
    return lists


def _extract_tables(content_root: Tag) -> list[list[list[str]]]:
    tables = []
    for table_tag in content_root.find_all("table"):
        rows = []
        for row in table_tag.find_all("tr"):
            cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)
    return tables


def _extract_links(content_root: Tag, base_url: str) -> list[str]:
    links = set()
    for a in content_root.find_all("a", href=True):
        href = a["href"].strip()
        if href and not href.startswith(("#", "javascript:", "mailto:", "tel:")):
            links.add(urljoin(base_url, href))
    return sorted(links)


def parse_page(html: str, url: str) -> ScrapedPage:
    """Transforme le HTML brut d'une page en ScrapedPage structurée."""
    soup = BeautifulSoup(html, "lxml")
    breadcrumbs = _extract_breadcrumbs(soup)
    published_date, modified_date = _extract_dates(soup)
    metadata = _extract_metadata(soup)
    title = soup.title.get_text(strip=True) if soup.title else ""

    content_root = soup.find("main") or soup.find(id="content") or soup.body or soup
    content_root = _clean_soup_for_extraction(content_root)

    headings = _extract_headings(content_root)
    lists_ = _extract_lists(content_root)
    tables = _extract_tables(content_root)
    links = _extract_links(content_root, url)
    text = content_root.get_text(separator="\n", strip=True)

    return ScrapedPage(
        url=url,
        title=title,
        headings=headings,
        text=text,
        raw_html=html,
        lists=lists_,
        tables=tables,
        links=links,
        breadcrumbs=breadcrumbs,
        metadata=metadata,
        published_date=published_date,
        modified_date=modified_date,
        scraped_at=datetime.now(timezone.utc).isoformat(),
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


class EnseaScraper:
    """Orchestrateur du crawl complet : découverte + extraction HTML + PDFs."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.raw_dir = config.project.data_path / "raw"
        self.pdf_dir = config.project.data_path / "pdf"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_extractor = PDFExtractor(self.pdf_dir)
        self.domain = urlparse(config.project.base_url).netloc

    def _save_page(self, page: ScrapedPage) -> None:
        path = self.raw_dir / f"{url_hash(page.url)}.json"
        path.write_text(json.dumps(page.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_pdf_record(self, record: dict) -> None:
        path = self.raw_dir / f"pdf_{url_hash(record['url'])}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    async def run(
        self,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> None:
        seed_urls = SitemapDiscoverer(self.config.project.sitemap_url).discover()
        if not seed_urls:
            seed_urls = [self.config.project.base_url]
        logger.info("Sitemap : %d URLs découvertes en amorce du crawl.", len(seed_urls))

        max_requests = self.config.scraping.max_pages or None
        progress_total = min(len(seed_urls), max_requests) if max_requests else len(seed_urls)
        processed = 0
        crawler = BeautifulSoupCrawler(
            respect_robots_txt_file=True,
            max_requests_per_crawl=max_requests,
            concurrency_settings=ConcurrencySettings(
                max_concurrency=self.config.scraping.max_concurrency,
                desired_concurrency=min(10, self.config.scraping.max_concurrency),
            ),
        )

        @crawler.router.default_handler
        async def handler(context: BeautifulSoupCrawlingContext) -> None:
            nonlocal processed, progress_total
            url = context.request.url
            content_type = context.http_response.headers.get("content-type", "")

            if url.lower().endswith(".pdf") or "application/pdf" in content_type:
                if self.config.scraping.download_pdfs:
                    record = self.pdf_extractor.download_and_extract(url)
                    if record:
                        self._save_pdf_record(record)
                        await context.push_data(record)
                processed += 1
                progress_total = max(progress_total, processed)
                if progress_callback:
                    progress_callback(processed, progress_total, url)
                return

            page = parse_page(str(context.soup), url)
            self._save_page(page)
            await context.push_data(page.__dict__)

            def _skip_non_pages(request_options):
                return "skip" if not _is_enqueueable(request_options["url"]) else "unchanged"

            await context.enqueue_links(
                strategy="same-domain", transform_request_function=_skip_non_pages,
            )
            processed += 1
            progress_total = max(progress_total, processed)
            if progress_callback:
                progress_callback(processed, progress_total, url)

        requests = [Request.from_url(u) for u in seed_urls]
        await crawler.run(requests)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    scraper = EnseaScraper(config)
    asyncio.run(scraper.run())


if __name__ == "__main__":
    main()
