from __future__ import annotations

import re
import unittest
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
BACKEND = ROOT / "backend"


class _IdParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.append(attributes["id"])


class StaticContractTests(unittest.TestCase):
    def test_javascript_id_references_exist_and_html_ids_are_unique(self) -> None:
        html = (FRONTEND / "index.html").read_text(encoding="utf-8")
        javascript = (FRONTEND / "app.js").read_text(encoding="utf-8")
        parser = _IdParser()
        parser.feed(html)
        counts = Counter(parser.ids)
        duplicates = sorted(identifier for identifier, count in counts.items() if count > 1)
        self.assertEqual(duplicates, [])
        referenced = set(re.findall(r'\$\("#([A-Za-z0-9_-]+)"\)', javascript))
        missing = sorted(referenced - set(parser.ids))
        self.assertEqual(missing, [])

    def test_reasoning_trace_is_streamed_collapsible_and_animated(self) -> None:
        javascript = (FRONTEND / "app.js").read_text(
            encoding="utf-8"
        )
        stylesheet = (FRONTEND / "styles.css").read_text(
            encoding="utf-8"
        )
        self.assertIn('event.type === "reasoning"', javascript)
        self.assertIn('class="reasoning-toggle"', javascript)
        self.assertIn("reasoning-transcript", javascript)
        self.assertIn(".reasoning-summary.shimmer", stylesheet)
        self.assertIn(".reasoning-trace.expanded", stylesheet)

    def test_facebook_collection_is_configurable_and_secret_is_not_job_payload(self) -> None:
        html = (FRONTEND / "index.html").read_text(encoding="utf-8")
        javascript = (FRONTEND / "app.js").read_text(encoding="utf-8")
        admin_service = (BACKEND / "admin_service.py").read_text(encoding="utf-8")
        self.assertIn('id="facebook-page-options"', html)
        self.assertIn('id="facebook-max-posts"', html)
        self.assertIn('id="facebook-token" type="password"', html)
        self.assertIn("page_keys: pageKeys", javascript)
        self.assertIn("max_posts_per_page: maxPosts", javascript)
        self.assertIn("_job_apify_tokens", admin_service)
        self.assertIn('environment["APIFY_TOKEN"]', admin_service)

    def test_interface_uses_segoe_ui_variable_and_responsive_facebook_cards(self) -> None:
        stylesheet = (FRONTEND / "styles.css").read_text(
            encoding="utf-8"
        )
        self.assertIn('"Segoe UI Variable Text"', stylesheet)
        self.assertIn(".parameter-box", stylesheet)
        self.assertIn(".facebook-page-grid", stylesheet)
        self.assertIn(
            ".operation-grid,.facebook-page-grid,.schedule-facebook-config{grid-template-columns:1fr}",
            stylesheet,
        )

    def test_every_admin_operation_uses_a_real_configuration_panel(self) -> None:
        html = (FRONTEND / "index.html").read_text(encoding="utf-8")
        javascript = (FRONTEND / "app.js").read_text(encoding="utf-8")
        admin_service = (BACKEND / "admin_service.py").read_text(encoding="utf-8")
        pipeline = (ROOT / "scripts" / "admin_pipeline.py").read_text(encoding="utf-8")
        scraper = (ROOT / "src" / "scraper.py").read_text(encoding="utf-8")
        self.assertIn('id="operation-configurator"', html)
        self.assertIn('id="operation-job-status"', html)
        for kind in (
            "collect_institutional", "import_local_documents", "clean",
            "chunk", "embed", "index", "rebuild",
        ):
            self.assertIn(f"{kind}:", javascript)
        self.assertIn("openOperationConfigurator(kind)", javascript)
        self.assertNotIn("window.confirm(`Lancer", javascript)
        self.assertIn('command.extend(["--max-pages"', admin_service)
        self.assertIn('parser.add_argument("--max-concurrency"', pipeline)
        self.assertIn("if self.config.scraping.download_pdfs:", scraper)

    def test_admin_completion_refreshes_tools_without_unloading_on_chunking(self) -> None:
        javascript = (FRONTEND / "app.js").read_text(encoding="utf-8")
        admin_service = (BACKEND / "admin_service.py").read_text(encoding="utf-8")
        self.assertIn("synchronizeAdminJobCompletion", javascript)
        self.assertIn("if (completedNow) await updateHealth()", javascript)
        self.assertIn("if (launch && state.user) updateHealth()", javascript)
        self.assertIn("PIPELINE_RESTORE_TASKS = GPU_EXCLUSIVE_TASKS", admin_service)
        self.assertIn("self.rag_service.get_pipeline()", admin_service)
        self.assertIn("index actif conservé", admin_service)

    def test_lisa_and_ensea_brand_assets_are_integrated(self) -> None:
        html = (FRONTEND / "index.html").read_text(encoding="utf-8")
        stylesheet = (FRONTEND / "styles.css").read_text(encoding="utf-8")
        assets = FRONTEND / "assets"
        self.assertTrue((assets / "lisa-logo.png").is_file())
        self.assertTrue((assets / "lisa-mark.png").is_file())
        self.assertTrue((assets / "ensea-logo.png").is_file())
        self.assertIn("LISA · ENSEA", html)
        self.assertIn("/assets/assets/lisa-mark.png", html)
        self.assertIn("/assets/assets/ensea-logo.png", html)
        self.assertIn("body { font-size: 15px; }", stylesheet)

    def test_account_lifecycle_and_security_console_are_wired(self) -> None:
        html = (FRONTEND / "index.html").read_text(encoding="utf-8")
        javascript = (FRONTEND / "app.js").read_text(encoding="utf-8")
        api = (BACKEND / "main.py").read_text(encoding="utf-8")
        for identifier in ("auth-register", "auth-forgot", "auth-guest", "security-audit-list"):
            self.assertIn(f'id="{identifier}"', html)
        self.assertIn('"/api/auth/register"', javascript)
        self.assertIn('"/api/auth/guest"', javascript)
        self.assertIn('"/api/auth/password/forgot"', javascript)
        self.assertIn('@app.post("/api/auth/password/reset")', api)

    def test_frontend_backend_worker_and_deployment_are_separate(self) -> None:
        self.assertTrue((ROOT / "frontend" / "index.html").is_file())
        self.assertTrue((ROOT / "backend" / "main.py").is_file())
        self.assertTrue((ROOT / "backend" / "worker.py").is_file())
        compose = (ROOT / "compose.test.yaml").read_text(encoding="utf-8")
        self.assertIn("postgres:", compose)
        self.assertIn("worker:", compose)
        self.assertIn("frontend:", compose)

    def test_docker_deployment_is_cpu_portable_and_offline_ready(self) -> None:
        compose = (ROOT / "compose.test.yaml").read_text(encoding="utf-8")
        gpu = (ROOT / "compose.gpu.yaml").read_text(encoding="utf-8")
        generator = (ROOT / "src" / "generator.py").read_text(encoding="utf-8")
        admin_service = (BACKEND / "admin_service.py").read_text(encoding="utf-8")
        server_guide = (ROOT / "docs" / "deploiement_test.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("ollama:", compose)
        self.assertIn('RAG_FORCE_CPU: ${RAG_FORCE_CPU:-1}', compose)
        self.assertIn('RAG_OLLAMA_NUM_GPU: ${RAG_OLLAMA_NUM_GPU:-0}', compose)
        self.assertIn('HF_HUB_OFFLINE: "1"', compose)
        self.assertIn('TRANSFORMERS_OFFLINE: "1"', compose)
        self.assertIn('OLLAMA_NO_CLOUD: "1"', compose)
        self.assertIn("gpus: all", gpu)
        self.assertIn('options["num_gpu"]', generator)
        self.assertIn("settings.offline_mode and kind in NETWORK_TASKS", admin_service)
        self.assertNotIn("```powershell", server_guide.casefold())
        self.assertTrue((ROOT / "docs" / "deploiement_local.md").is_file())
        self.assertTrue(
            (ROOT / "deployment" / "linux" / "prepare_offline_bundle.sh").is_file()
        )
        self.assertFalse((ROOT / "compose.ollama.yaml").exists())


if __name__ == "__main__":
    unittest.main()
