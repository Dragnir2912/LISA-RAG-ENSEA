"""File de travaux et planificateur persistant de la console administrateur."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from backend.database import AppDatabase, utc_now
from backend.configuration import settings
from backend.rag_service import PROJECT_ROOT, RAGService
from backend.secret_store import secret_store


ALLOWED_TASKS = {
    "collect_institutional": "Collecte du site institutionnel",
    "collect_facebook": "Collecte des publications Facebook",
    "import_local_documents": "Import de la bibliothèque PDF locale",
    "clean": "Nettoyage des corpus",
    "chunk": "Reconstruction des chunks",
    "embed": "Recalcul des embeddings",
    "index": "Reconstruction LanceDB et BM25",
    "rebuild": "Reconstruction RAG sans nouvelle collecte",
    "full_refresh": "Collecte et reconstruction complètes",
}
NETWORK_TASKS = {"collect_institutional", "collect_facebook", "full_refresh"}
GPU_EXCLUSIVE_TASKS = {"embed", "index", "rebuild", "full_refresh"}
# Ces tâches déchargent le pipeline pour libérer la VRAM ou remplacent les
# index actifs. Elles doivent donc remettre un pipeline interrogeable en place
# avant d'être signalées comme terminées à l'interface.
PIPELINE_RESTORE_TASKS = GPU_EXCLUSIVE_TASKS
PROGRESS_PATTERN = re.compile(r"^PROGRESS\s+(\d+)\s+(.*)$")


class AdminTaskService:
    """Un worker protège les fichiers d'index et l'unique GPU de la machine."""

    def __init__(self, database: AppDatabase, rag_service: RAGService) -> None:
        self.database = database
        self.rag_service = rag_service
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-admin")
        self._stop = threading.Event()
        self._scheduler_thread: threading.Thread | None = None
        self._secret_lock = threading.Lock()
        self._job_apify_tokens: dict[str, str] = {}
        # Une saisie depuis l'interface peut être conservée pour la durée de vie
        # du serveur. Le secret n'est jamais écrit dans SQLite ni dans les logs.
        self._runtime_apify_token = (os.getenv("APIFY_TOKEN") or "").strip()

    @property
    def apify_token_configured(self) -> bool:
        with self._secret_lock:
            return bool(self._runtime_apify_token)

    def start(self) -> None:
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            return
        for job in self.database.list_jobs(limit=100):
            if job["status"] == "running":
                self.database.update_job(
                    job["id"], status="error", progress=100,
                    message="Interrompue par un redémarrage du serveur",
                    error="server_restarted",
                )
        self._stop.clear()
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop, name="rag-scheduler", daemon=True
        )
        self._scheduler_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.executor.shutdown(wait=False, cancel_futures=False)

    def submit(
        self,
        kind: str,
        *,
        requested_by: str | None,
        schedule_id: str | None = None,
        payload: dict[str, Any] | None = None,
        apify_token: str | None = None,
        remember_apify_token: bool = False,
    ) -> dict[str, Any]:
        if kind not in ALLOWED_TASKS:
            raise ValueError("Type de tâche non autorisé.")
        if settings.offline_mode and kind in NETWORK_TASKS:
            raise ValueError(
                "Mode hors ligne actif : les collectes réseau sont désactivées."
            )
        token = (apify_token or "").strip()
        if kind in {"collect_facebook", "full_refresh"}:
            with self._secret_lock:
                if token and remember_apify_token:
                    self._runtime_apify_token = token
                token = token or self._runtime_apify_token
            if not token:
                raise ValueError(
                    "Jeton Apify requis. Ouvrez la configuration Facebook, "
                    "saisissez le jeton puis relancez la collecte."
                )
        job_payload = dict(payload or {})
        if token and not settings.jobs_inline:
            job_payload["_secret_apify_token"] = secret_store.seal(token)
        job = self.database.create_job(
            kind=kind,
            requested_by=requested_by,
            schedule_id=schedule_id,
            payload=job_payload,
        )
        if token:
            with self._secret_lock:
                self._job_apify_tokens[job["id"]] = token
        if settings.jobs_inline:
            self.executor.submit(self._run, job["id"], kind)
        return job

    def process_existing_job(self, job: dict[str, Any]) -> None:
        kind = str(job.get("kind") or "")
        if kind not in ALLOWED_TASKS:
            raise ValueError(f"Type de tâche inconnu : {kind}")
        if settings.offline_mode and kind in NETWORK_TASKS:
            raise ValueError(
                "Mode hors ligne actif : cette ancienne tâche réseau ne sera pas exécutée."
            )
        self._run(job["id"], kind)

    def _run(self, job_id: str, kind: str) -> None:
        job = self.database.get_job(job_id, include_secrets=True) or {}
        self.database.update_job(
            job_id, status="running", progress=1, message=ALLOWED_TASKS[kind]
        )
        command = [sys.executable, str(PROJECT_ROOT / "scripts" / "admin_pipeline.py"), kind]
        payload = job.get("payload") or {}
        if kind in {"collect_facebook", "full_refresh"}:
            for page_key in payload.get("page_keys", []):
                command.extend(["--facebook-page", str(page_key)])
            if payload.get("max_posts_per_page") is not None:
                command.extend(["--max-posts", str(payload["max_posts_per_page"])])
        if kind == "collect_institutional":
            command.extend(["--max-pages", str(payload.get("max_pages", 0))])
            command.extend([
                "--max-concurrency", str(payload.get("max_concurrency", 4))
            ])
            if not payload.get("download_pdfs", True):
                command.append("--skip-institutional-pdfs")
        if kind == "import_local_documents" and payload.get(
            "continue_on_pdf_errors", True
        ):
            command.append("--continue-on-pdf-errors")
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        with self._secret_lock:
            apify_token = self._job_apify_tokens.pop(job_id, "")
        sealed_token = str(payload.get("_secret_apify_token") or "")
        if sealed_token:
            apify_token = secret_store.open(sealed_token)
        if apify_token:
            environment["APIFY_TOKEN"] = apify_token
        output_tail: list[str] = []
        explicit_error = ""
        try:
            context = (
                self.rag_service.maintenance()
                if kind in GPU_EXCLUSIVE_TASKS
                else _null_context()
            )
            with context:
                process = subprocess.Popen(
                    command,
                    cwd=str(PROJECT_ROOT),
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=(
                        subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                    ),
                )
                last_message = ALLOWED_TASKS[kind]
                assert process.stdout is not None
                for raw_line in process.stdout:
                    line = raw_line.strip()
                    if not line:
                        continue
                    output_tail.append(line)
                    output_tail = output_tail[-12:]
                    progress_match = PROGRESS_PATTERN.match(line)
                    if progress_match:
                        progress = max(1, min(int(progress_match.group(1)), 99))
                        last_message = progress_match.group(2)[:500]
                        self.database.update_job(
                            job_id, status="running", progress=progress,
                            message=last_message,
                        )
                    elif line.startswith("RESULT "):
                        last_message = "Traitement terminé"
                    elif line.startswith("ERROR "):
                        explicit_error = line.removeprefix("ERROR ").strip()
                exit_code = process.wait()
                if exit_code != 0:
                    detail = " | ".join(output_tail[-4:])
                    raise RuntimeError(
                        explicit_error
                        or f"Le traitement s'est arrêté avec le code {exit_code}. {detail}"
                    )
            if kind in PIPELINE_RESTORE_TASKS:
                self.database.update_job(
                    job_id,
                    status="running",
                    progress=99,
                    message="Rechargement du pipeline et vérification des index",
                )
                self.rag_service.reload()
                if settings.jobs_inline:
                    self.rag_service.get_pipeline()
            self.database.update_job(
                job_id, status="completed", progress=100,
                message=(
                    "Traitement terminé · nouveaux index actifs"
                    if kind in PIPELINE_RESTORE_TASKS
                    else "Traitement terminé · index actif conservé"
                ),
            )
            if kind in PIPELINE_RESTORE_TASKS:
                self.database.set_settings(
                    {"pipeline_revision": utc_now()},
                    updated_by=job.get("requested_by"),
                )
            self.database.add_audit_event(
                "maintenance.completed",
                actor_user_id=job.get("requested_by"),
                target_type="job",
                target_id=job_id,
                details={"kind": kind, "schedule_id": job.get("schedule_id")},
            )
        except Exception as error:
            current_job = self.database.get_job(job_id) or {}
            self.database.update_job(
                job_id, status="error", progress=int(current_job.get("progress") or 0),
                message="Échec du traitement",
                error=f"{type(error).__name__}: {error}",
            )
            self.database.add_audit_event(
                "maintenance.failed",
                actor_user_id=job.get("requested_by"),
                target_type="job",
                target_id=job_id,
                details={
                    "kind": kind,
                    "schedule_id": job.get("schedule_id"),
                    "error": f"{type(error).__name__}: {error}",
                },
            )

    def _scheduler_loop(self) -> None:
        while not self._stop.wait(30):
            now = datetime.now(timezone.utc)
            for schedule in self.database.list_due_schedules(now.isoformat()):
                try:
                    if (
                        settings.offline_mode
                        and schedule["task_kind"] in NETWORK_TASKS
                    ):
                        next_run = now + timedelta(
                            minutes=int(schedule["interval_minutes"])
                        )
                        self.database.update_schedule(
                            schedule["id"], next_run_at=next_run.isoformat()
                        )
                        self.database.add_audit_event(
                            "schedule.skipped_offline",
                            actor_user_id=schedule.get("created_by"),
                            target_type="schedule",
                            target_id=schedule["id"],
                            details={"task_kind": schedule["task_kind"]},
                        )
                        continue
                    self.submit(
                        schedule["task_kind"],
                        requested_by=schedule.get("created_by"),
                        schedule_id=schedule["id"],
                        payload=schedule.get("payload"),
                    )
                    next_run = now + timedelta(minutes=int(schedule["interval_minutes"]))
                    self.database.update_schedule(
                        schedule["id"],
                        last_run_at=utc_now(),
                        next_run_at=next_run.isoformat(),
                    )
                except Exception as error:
                    self.database.add_audit_event(
                        "schedule.failed",
                        actor_user_id=schedule.get("created_by"),
                        target_type="schedule",
                        target_id=schedule["id"],
                        details={"error": f"{type(error).__name__}: {error}"},
                    )


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False
