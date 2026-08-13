"""Worker unique des collectes, imports, embeddings et planifications.

Il tourne dans un processus/conteneur distinct de l'API. Le serveur web reste
donc réactif même lorsqu'un PDF ou un corpus complet est reconstruit.
"""

from __future__ import annotations

import logging
import signal
import threading
import time

from backend.admin_service import AdminTaskService
from backend.configuration import settings
from backend.database import AppDatabase, utc_now
from backend.document_service import DocumentIngestionService
from backend.rag_service import rag_service


logger = logging.getLogger(__name__)


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    database = AppDatabase(database_url=settings.database_url)
    rag_service.configure(database.get_settings)
    documents = DocumentIngestionService(database, rag_service)
    tasks = AdminTaskService(database, rag_service)
    stop = threading.Event()

    def request_stop(*_args) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    tasks.start()
    last_identity_cleanup = 0.0
    logger.info("Worker LISA prêt (base=%s).", "PostgreSQL" if database.is_postgres else "SQLite")
    try:
        while not stop.wait(1.0):
            database.set_settings({"worker_heartbeat": utc_now()}, updated_by=None)
            if time.monotonic() - last_identity_cleanup >= 3600:
                cleanup = database.purge_expired_identity_data(settings.guest_data_days)
                if any(cleanup.values()):
                    logger.info("Nettoyage des identités temporaires : %s", cleanup)
                last_identity_cleanup = time.monotonic()
            job = database.claim_next_job()
            if not job:
                continue
            try:
                if job["kind"] == "pdf_ingestion":
                    documents.process_existing_job(job)
                else:
                    tasks.process_existing_job(job)
            except Exception as error:
                logger.exception("Échec du travail %s", job["id"])
                database.update_job(
                    job["id"],
                    status="error",
                    progress=int(job.get("progress") or 0),
                    message="Échec du traitement",
                    error=f"{type(error).__name__}: {error}",
                )
    finally:
        tasks.stop()


if __name__ == "__main__":
    run()
