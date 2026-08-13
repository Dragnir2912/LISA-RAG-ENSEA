from __future__ import annotations

import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from backend.database import AppDatabase
from backend.configuration import ApplicationSettings
from backend.rag_service import RAGService
from backend.security import AuthService
from src.config import load_config
from src.conversation import contextualize_question, update_memory_summary


class ApplicationFoundationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = AppDatabase(Path(self.temp_dir.name) / "app.sqlite3")
        self.auth = AuthService(self.database)
        self.admin = self.auth.setup_admin(
            email="admin@ensea.test",
            display_name="Administration",
            password="Password1234",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_conversations_are_isolated_by_user(self) -> None:
        second = self.auth.create_user(
            actor_user_id=self.admin["id"],
            email="user@ensea.test",
            display_name="Utilisateur",
            password="Password1234",
            role="user",
        )
        conversation = self.database.create_conversation(self.admin["id"], "Privée")
        self.assertIsNotNone(
            self.database.get_conversation(conversation["id"], self.admin["id"])
        )
        self.assertIsNone(
            self.database.get_conversation(conversation["id"], second["id"])
        )
        self.assertEqual(self.database.list_conversations(second["id"]), [])

    def test_login_session_is_server_side_and_revocable(self) -> None:
        token, session = self.auth.login(
            email="admin@ensea.test", password="Password1234"
        )
        authenticated = self.auth.authenticate(token)
        self.assertIsNotNone(authenticated)
        self.assertEqual(authenticated.user["role"], "admin")
        self.assertTrue(
            self.database.revoke_session(session.session_id, user_id=self.admin["id"])
        )
        self.assertIsNone(self.auth.authenticate(token))

    def test_registration_guest_and_password_recovery_lifecycle(self) -> None:
        user = self.auth.register(
            email="new@ensea.test",
            display_name="Nouveau compte",
            password="Password1234",
            ip_address="127.0.0.1",
        )
        self.assertEqual(user["account_type"], "registered")
        guest_token, guest = self.auth.create_guest_session(ip_address="127.0.0.1")
        self.assertEqual(guest.user["account_type"], "guest")
        self.assertIsNotNone(self.auth.authenticate(guest_token))
        reset_token = self.auth.request_password_reset(
            email="new@ensea.test", ip_address="127.0.0.1"
        )
        self.assertTrue(reset_token)
        self.auth.reset_password(raw_token=reset_token, password="NewPassword1234")
        token, _ = self.auth.login(email="new@ensea.test", password="NewPassword1234")
        self.assertIsNotNone(self.auth.authenticate(token))

    def test_schedule_and_settings_are_persistent(self) -> None:
        schedule = self.database.create_schedule(
            name="Collecte hebdomadaire",
            task_kind="collect_institutional",
            interval_minutes=10_080,
            next_run_at="2099-01-01T00:00:00+00:00",
            created_by=self.admin["id"],
        )
        self.assertEqual(self.database.get_schedule(schedule["id"])["interval_minutes"], 10_080)
        settings = self.database.set_settings(
            {"generation_model": "qwen3.5:9b"}, updated_by=self.admin["id"]
        )
        self.assertEqual(settings["generation_model"], "qwen3.5:9b")

    def test_qwen35_is_the_configured_default_and_evaluation_judge(self) -> None:
        config = load_config()
        self.assertEqual(config.generation.model_name, "qwen3.5:9b")
        self.assertEqual(config.evaluation.ragas_judge_model, "qwen3.5:9b")
        self.assertEqual(
            config.generation.alternatives,
            ["llama3.2:3b", "gemma3:12b", "mistral:7b"],
        )
        self.assertEqual(config.generation.min_num_ctx, 6144)
        self.assertEqual(config.generation.num_ctx, 6144)
        self.assertEqual(config.generation.reserved_output_tokens, 1536)
        self.assertEqual(config.generation.thinking_output_tokens, 1536)

    def test_cpu_and_offline_deployment_are_explicit(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "RAG_FORCE_CPU": "1",
                "RAG_OLLAMA_NUM_GPU": "0",
                "RAG_OFFLINE_MODE": "1",
            },
            clear=False,
        ):
            app_settings = ApplicationSettings.from_env()
            effective = RAGService()._load_effective_config()
        self.assertTrue(app_settings.offline_mode)
        self.assertEqual(effective.embeddings.device, "cpu")
        self.assertEqual(effective.reranker.device, "cpu")
        self.assertEqual(effective.embeddings.precision, "fp32")
        self.assertEqual(effective.generation.num_gpu, 0)

    def test_document_and_generic_job_keep_admin_ownership(self) -> None:
        document = self.database.create_document(
            filename="rapport.pdf",
            stored_name="abc123.pdf",
            title="Rapport",
            author=None,
            document_date=None,
            category="rapport",
            sha256="0" * 64,
            size_bytes=123,
            created_by=self.admin["id"],
        )
        job = self.database.create_job(
            document["id"], requested_by=self.admin["id"], payload={"mode": "test"}
        )
        self.assertEqual(document["created_by"], self.admin["id"])
        self.assertEqual(job["requested_by"], self.admin["id"])
        self.assertEqual(job["payload"], {"mode": "test"})

    def test_latency_observability_separates_queue_and_first_token(self) -> None:
        conversation = self.database.create_conversation(self.admin["id"], "Mesures")
        message = self.database.add_message(
            conversation["id"],
            "assistant",
            "Réponse mesurée",
            metrics={
                "queue_seconds": 0.25,
                "time_to_first_token_seconds": 1.5,
                "retrieval_seconds": 0.4,
                "generation_seconds": 2.0,
                "total_seconds": 2.4,
                "tokens_per_second": 30.0,
                "thinking_enabled": True,
                "model_name": "qwen3.5:9b",
            },
            reasoning={
                "summary_key": "sources",
                "content": "Je compare les passages disponibles avant de répondre.",
                "seconds": 1.1,
                "truncated": True,
            },
        )
        self.assertEqual(message["metrics"]["queue_seconds"], 0.25)
        self.assertEqual(message["metrics"]["time_to_first_token_seconds"], 1.5)
        self.assertTrue(message["metrics"]["thinking_enabled"])
        self.assertEqual(message["metrics"]["model_name"], "qwen3.5:9b")
        self.assertEqual(message["reasoning"]["summary_key"], "sources")
        self.assertIn("compare les passages", message["reasoning"]["content"])
        self.assertEqual(message["reasoning"]["seconds"], 1.1)
        self.assertTrue(message["reasoning"]["truncated"])
        metrics = self.database.metrics(self.admin["id"])
        self.assertEqual(metrics["avg_queue_seconds"], 0.25)
        self.assertEqual(metrics["avg_first_token_seconds"], 1.5)

    def test_conversation_memory_supports_pronoun_followup(self) -> None:
        history = [
            {"role": "user", "content": "Qui dirige actuellement l'ENSEA ?"},
            {"role": "assistant", "content": "Le directeur est indiqué dans la source [1]."},
        ]
        query, contextualized = contextualize_question("Depuis quand est-il en poste ?", history)
        self.assertTrue(contextualized)
        self.assertIn("Qui dirige actuellement", query)
        summary = update_memory_summary("", history[0]["content"], history[1]["content"])
        self.assertIn("Qui dirige", summary)
        self.assertNotIn("[1]", summary)


class LegacyMigrationTests(unittest.TestCase):
    def test_legacy_conversation_table_is_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.sqlite3"
            connection = sqlite3.connect(path)
            connection.execute(
                """
                CREATE TABLE conversations (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                )
                """
            )
            connection.commit()
            connection.close()
            database = AppDatabase(path)
            with database._connect() as migrated:
                columns = {
                    row["name"]
                    for row in migrated.execute("PRAGMA table_info(conversations)")
                }
            self.assertTrue({"user_id", "summary", "archived_at"} <= columns)


if __name__ == "__main__":
    unittest.main()
