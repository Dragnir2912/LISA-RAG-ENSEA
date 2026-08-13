"""Point d'entrée sûr des travaux lancés depuis la console d'administration."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import load_config  # noqa: E402
from facebook_scraper import FacebookScraper  # noqa: E402
from local_documents import LocalPDFImporter  # noqa: E402
from maintenance import run_stage  # noqa: E402
from scraper import EnseaScraper  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "collect_institutional", "collect_facebook", "import_local_documents",
            "clean", "chunk", "embed", "index", "rebuild", "full_refresh",
        ),
    )
    parser.add_argument(
        "--facebook-page",
        action="append",
        dest="facebook_pages",
        help="Clé d'une page Facebook configurée (option répétable).",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=None,
        help="Nombre maximal de publications à récupérer par page.",
    )
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--skip-institutional-pdfs", action="store_true")
    parser.add_argument("--continue-on-pdf-errors", action="store_true")
    args = parser.parse_args()
    config = load_config()
    if args.stage == "collect_institutional":
        if args.max_pages is not None:
            config.scraping.max_pages = args.max_pages
        if args.max_concurrency is not None:
            config.scraping.max_concurrency = args.max_concurrency
        if args.skip_institutional_pdfs:
            config.scraping.download_pdfs = False

        def institutional_progress(current: int, total: int, url: str) -> None:
            percent = 5 + round(90 * current / max(total, 1))
            host_path = url.split("//", 1)[-1][:90]
            print(
                f"PROGRESS {percent} Site ENSEA · {current}/{total} · {host_path}",
                flush=True,
            )

        print("PROGRESS 2 Préparation de la collecte institutionnelle", flush=True)
        asyncio.run(
            EnseaScraper(config).run(progress_callback=institutional_progress)
        )
        result = {"institutional_collection": "completed"}
    elif args.stage == "collect_facebook":
        def facebook_progress(current: int, total: int, page: str, state: str) -> None:
            if state == "starting":
                percent = 5 + round((current / max(total, 1)) * 90)
                message = f"Facebook · collecte de {page} ({current + 1}/{total})"
            else:
                percent = 5 + round((current / max(total, 1)) * 90)
                message = f"Facebook · {page} terminé ({current}/{total})"
            print(f"PROGRESS {percent} {message}", flush=True)

        reports = FacebookScraper(config).run(
            page_keys=args.facebook_pages,
            max_posts_per_page=args.max_posts,
            progress_callback=facebook_progress,
        )
        result = {
            "facebook_pages": len(reports),
            "facebook_posts": sum(report.unique_posts for report in reports),
        }
    elif args.stage == "import_local_documents":
        print("PROGRESS 3 Analyse de la bibliothèque PDF locale", flush=True)
        report = LocalPDFImporter(config).import_all(show_progress=True)
        result = {
            "discovered_pdfs": report.discovered_pdfs,
            "imported_pdfs": report.imported_pdfs,
            "reused_pdfs": report.reused_pdfs,
            "failed_pdfs": report.failed_pdfs,
        }
        print(
            f"PROGRESS 95 Import PDF terminé · {report.imported_pdfs} importés, "
            f"{report.reused_pdfs} réutilisés, {report.failed_pdfs} à vérifier",
            flush=True,
        )
        if report.failed_pdfs and not args.continue_on_pdf_errors:
            raise RuntimeError(
                f"{report.failed_pdfs} PDF sans texte exploitable ; OCR requis"
            )
    elif args.stage == "full_refresh":
        asyncio.run(EnseaScraper(config).run())
        FacebookScraper(config).run(
            page_keys=args.facebook_pages,
            max_posts_per_page=args.max_posts,
        )
        LocalPDFImporter(config).import_all(show_progress=True)
        result = run_stage("rebuild")
    else:
        result = run_stage(args.stage)
    print("RESULT " + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Ligne structurée consommée par la console d'administration. Le détail
        # utile est affiché sans imposer une trace Python à l'administrateur.
        print(f"ERROR {type(error).__name__}: {error}", flush=True)
        sys.exit(1)
