"""Import de PDF et synchronisation incrémentale avec l'index institutionnel."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pypdf import PdfReader

from backend.database import AppDatabase
from backend.configuration import settings
from backend.rag_service import PROJECT_ROOT, RAGService

if TYPE_CHECKING:
    from chunker import Chunk, HierarchicalChunker


logger = logging.getLogger(__name__)
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads" / "pdfs"
UPLOAD_CHUNKS_DIR = PROJECT_ROOT / "data" / "uploads" / "chunks"
MAX_PDF_BYTES = 50 * 1024 * 1024


def _safe_filename(filename: str) -> str:
    name = Path(filename or "document.pdf").name
    name = re.sub(r"[^\w.() -]+", "_", name, flags=re.UNICODE).strip(" .")
    return (name or "document.pdf")[:240]


class DocumentIngestionService:
    """File d'attente à un worker pour protéger l'unique GPU de la machine."""

    def __init__(self, database: AppDatabase, rag_service: RAGService) -> None:
        self.database = database
        self.rag_service = rag_service
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdf-index")
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        UPLOAD_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    def submit(
        self,
        *,
        content: bytes,
        filename: str,
        title: str | None,
        author: str | None,
        document_date: str | None,
        category: str,
        created_by: str | None = None,
    ) -> dict[str, Any]:
        if not content.startswith(b"%PDF-"):
            raise ValueError("Le fichier transmis n'est pas un PDF valide.")
        if len(content) > MAX_PDF_BYTES:
            raise ValueError("Le PDF dépasse la limite de 50 Mo.")
        if len(content) < 100:
            raise ValueError("Le fichier PDF est vide ou incomplet.")

        digest = hashlib.sha256(content).hexdigest()
        duplicate = self.database.find_document_by_hash(digest)
        if duplicate:
            raise FileExistsError(duplicate["id"])

        document_id = uuid.uuid4().hex
        stored_name = f"{document_id}.pdf"
        path = UPLOAD_DIR / stored_name
        path.write_bytes(content)
        clean_filename = _safe_filename(filename)
        clean_title = (title or Path(clean_filename).stem).strip()[:240]
        document = self.database.create_document(
            filename=clean_filename,
            stored_name=stored_name,
            title=clean_title or "Document ENSEA",
            author=author,
            document_date=document_date,
            category=category or "document",
            sha256=digest,
            size_bytes=len(content),
            created_by=created_by,
        )
        job = self.database.create_job(
            document["id"], requested_by=created_by
        )
        if settings.jobs_inline:
            self.executor.submit(self._process, document["id"], job["id"])
        return {"document": document, "job": job}

    def process_existing_job(self, job: dict[str, Any]) -> None:
        if not job.get("document_id"):
            raise ValueError("Travail PDF sans document associé.")
        self._process(job["document_id"], job["id"])

    def _extract_chunks(
        self, path: Path, document: dict[str, Any], chunker: HierarchicalChunker
    ) -> tuple[list[Chunk], int]:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as error:
                raise ValueError("Le PDF est protégé par un mot de passe.") from error

        chunks: list[Chunk] = []
        extracted_characters = 0
        page_count = len(reader.pages)
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            extracted_characters += len(text)
            if len(text) < 40:
                continue
            page_url = f"/api/documents/{document['id']}/file#page={page_number}"
            markdown = f"# {document['title']}\n\n## Page {page_number}\n\n{text}"
            metadata = {
                "url": page_url,
                "title": document["title"],
                "doc_type": "uploaded_pdf",
                "depth": 0,
                "date": document.get("document_date"),
                "category": f"uploads/{document['category']}",
            }
            chunks.extend(
                chunker.chunk_document(
                    f"upload_{document['id']}_page_{page_number}",
                    markdown,
                    metadata,
                )
            )

        if extracted_characters < 80 or not chunks:
            raise ValueError(
                "Aucun texte exploitable n'a été extrait. Le PDF est probablement "
                "scanné et nécessite un traitement OCR."
            )
        return chunks, page_count

    def _save_chunks(self, document_id: str, chunks: list[Chunk]) -> None:
        output_path = UPLOAD_CHUNKS_DIR / f"{document_id}.jsonl"
        with open(output_path, "w", encoding="utf-8") as file:
            for chunk in chunks:
                file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    def _load_saved_chunks(self) -> list[Chunk]:
        from chunker import Chunk

        chunks: list[Chunk] = []
        for document in self.database.list_documents():
            if document["status"] != "ready":
                continue
            path = UPLOAD_CHUNKS_DIR / f"{document['id']}.jsonl"
            if not path.exists():
                logger.warning("Chunks sauvegardés absents pour le PDF %s.", document["id"])
                continue
            with open(path, encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        chunks.append(Chunk(**json.loads(line)))
        return chunks

    def reconcile_saved_documents(self) -> None:
        """Réinjecte les PDF après une reconstruction du corpus institutionnel."""
        saved_chunks = self._load_saved_chunks()
        with self.rag_service.exclusive_pipeline() as pipeline:
            current_chunks = list(pipeline.store.sparse.chunk_lookup.values())
            indexed_uploads = [
                chunk for chunk in current_chunks
                if chunk.url.startswith("/api/documents/")
            ]
            if not saved_chunks and not indexed_uploads:
                return
            institutional_chunks = [
                chunk for chunk in current_chunks
                if not chunk.url.startswith("/api/documents/")
            ]
            desired_chunks = institutional_chunks + saved_chunks
            report = pipeline.store.sync_incremental(
                desired_chunks,
                pipeline.embedding_model,
            )
            logger.info("Réconciliation des PDF ajoutés : %s", report)

    def _process(self, document_id: str, job_id: str) -> None:
        from chunker import HierarchicalChunker

        document = self.database.get_document(document_id)
        if not document:
            return
        path = UPLOAD_DIR / document["stored_name"]
        try:
            self.database.update_document_status(document_id, "processing")
            self.database.update_job(
                job_id,
                status="running",
                progress=10,
                message="Extraction du texte du PDF",
            )
            with self.rag_service.exclusive_pipeline() as pipeline:
                chunker = HierarchicalChunker(pipeline.config)
                new_chunks, page_count = self._extract_chunks(path, document, chunker)
                self.database.update_job(
                    job_id,
                    status="running",
                    progress=45,
                    message=f"{page_count} page(s) analysée(s), création des embeddings",
                )

                existing_chunks = list(pipeline.store.sparse.chunk_lookup.values())
                document_prefix = f"/api/documents/{document_id}/file"
                retained_chunks = [
                    chunk for chunk in existing_chunks
                    if not chunk.url.startswith(document_prefix)
                ]
                all_chunks = retained_chunks + new_chunks
                report = pipeline.store.sync_incremental(
                    all_chunks,
                    pipeline.embedding_model,
                )
                self._save_chunks(document_id, new_chunks)

            self.database.update_document_status(
                document_id,
                "ready",
                page_count=page_count,
                chunk_count=len(new_chunks),
            )
            self.database.update_job(
                job_id,
                status="completed",
                progress=100,
                message=(
                    f"Document indexé : {len(new_chunks)} passage(s), "
                    f"{report.embedded_chunks} embedding(s) calculé(s)"
                ),
            )
            self.database.set_settings(
                {"pipeline_revision": self.database.get_job(job_id)["finished_at"] or "pdf"},
                updated_by=document.get("created_by"),
            )
            if not settings.jobs_inline:
                self.rag_service.reload()
            logger.info("PDF %s indexé avec %d chunks.", document_id, len(new_chunks))
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            logger.exception("Échec d'indexation du PDF %s", document_id)
            self.database.update_document_status(document_id, "error", error=message)
            self.database.update_job(
                job_id,
                status="error",
                progress=100,
                message="Échec du traitement",
                error=message,
            )
