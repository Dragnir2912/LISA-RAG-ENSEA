"""API FastAPI de la plateforme conversationnelle multi-utilisateur RAG ENSEA."""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Iterator

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.admin_service import ALLOWED_TASKS, AdminTaskService
from backend.configuration import settings
from backend.database import AppDatabase
from backend.document_service import MAX_PDF_BYTES, UPLOAD_DIR, DocumentIngestionService
from backend.mailer import send_password_reset
from backend.middleware import install_security_middleware
from backend.rag_service import PROJECT_ROOT, rag_service
from backend.security import (
    SESSION_COOKIE,
    AuthService,
    AuthenticatedSession,
    hash_password,
)
from config import load_config
from conversation import update_memory_summary


logger = logging.getLogger(__name__)
STATIC_DIR = settings.frontend_dir
database = AppDatabase(database_url=settings.database_url)
auth_service = AuthService(database)
rag_service.configure(database.get_settings)
document_service = DocumentIngestionService(database, rag_service)
admin_tasks = AdminTaskService(database, rag_service)
SOURCE_DOCUMENTS_DIR = load_config(
    PROJECT_ROOT / "config.yaml"
).local_documents.source_full_path.resolve()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.scheduler_enabled:
        admin_tasks.start()
    yield
    if settings.scheduler_enabled:
        admin_tasks.stop()


app = FastAPI(
    title="RAG ENSEA",
    description="Plateforme conversationnelle documentaire avec citations et administration.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url=None if settings.production else "/api/docs",
    redoc_url=None,
    openapi_url=None if settings.production else "/api/openapi.json",
)
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")
install_security_middleware(app, settings)


class AuthCredentials(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)


class InitialSetup(AuthCredentials):
    display_name: str = Field(min_length=1, max_length=120)


class Registration(AuthCredentials):
    display_name: str = Field(min_length=1, max_length=120)


class PasswordResetRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)


class PasswordResetComplete(BaseModel):
    token: str = Field(min_length=20, max_length=256)
    password: str = Field(min_length=10, max_length=256)


class ConversationCreate(BaseModel):
    title: str = Field(default="Nouvelle conversation", max_length=120)


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
    thinking: bool = False


class FeedbackCreate(BaseModel):
    value: str
    comment: str | None = Field(default=None, max_length=1000)


class UserCreate(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    display_name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=10, max_length=256)
    role: str = Field(default="user", pattern="^(admin|user)$")


class UserUpdate(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=120)
    role: str | None = Field(default=None, pattern="^(admin|user)$")
    is_active: bool | None = None
    password: str | None = Field(default=None, min_length=10, max_length=256)


class JobCreate(BaseModel):
    kind: str
    payload: dict = Field(default_factory=dict)
    apify_token: str | None = Field(default=None, max_length=512)
    remember_apify_token: bool = False


class ScheduleCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    task_kind: str
    interval_minutes: int = Field(ge=5, le=525_600)
    enabled: bool = True
    payload: dict = Field(default_factory=dict)


class ScheduleUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=160)
    interval_minutes: int | None = Field(default=None, ge=5, le=525_600)
    enabled: bool | None = None


class SettingsUpdate(BaseModel):
    generation_model: str | None = None
    performance_profile: str | None = Field(
        default=None, pattern="^(cpu_only|generation_priority_8gb|balanced_gpu)$"
    )


def _validated_facebook_payload(raw_payload: dict) -> dict:
    """Valide une sélection Facebook sans jamais manipuler le jeton secret."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    available_pages = config.facebook_scraping.pages
    requested_pages = raw_payload.get("page_keys") or list(available_pages)
    if not isinstance(requested_pages, list) or not requested_pages:
        raise ValueError("Sélectionnez au moins une page Facebook.")
    invalid_pages = [key for key in requested_pages if key not in available_pages]
    if invalid_pages:
        raise ValueError("Page Facebook inconnue : " + ", ".join(map(str, invalid_pages)))
    try:
        max_posts = int(
            raw_payload.get(
                "max_posts_per_page", config.facebook_scraping.max_posts_per_page
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Le nombre de publications doit être un entier.") from error
    if not 1 <= max_posts <= 100:
        raise ValueError(
            "Le nombre de publications doit être compris entre 1 et 100 par page."
        )
    return {
        "page_keys": list(dict.fromkeys(requested_pages)),
        "max_posts_per_page": max_posts,
    }


def _validated_job_payload(kind: str, raw_payload: dict) -> dict:
    """Ne conserve que les paramètres réellement compris par chaque étape."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    if kind == "collect_institutional":
        try:
            max_pages = int(raw_payload.get("max_pages", config.scraping.max_pages))
            concurrency = int(
                raw_payload.get("max_concurrency", config.scraping.max_concurrency)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Les limites de collecte doivent être des entiers.") from error
        if not 0 <= max_pages <= 20_000:
            raise ValueError("La limite de pages doit être comprise entre 0 et 20 000.")
        if not 1 <= concurrency <= 12:
            raise ValueError("Le nombre de téléchargements simultanés doit être compris entre 1 et 12.")
        return {
            "max_pages": max_pages,
            "max_concurrency": concurrency,
            "download_pdfs": bool(
                raw_payload.get("download_pdfs", config.scraping.download_pdfs)
            ),
        }
    if kind == "import_local_documents":
        return {
            "continue_on_pdf_errors": bool(
                raw_payload.get("continue_on_pdf_errors", True)
            )
        }
    return {}


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else ""


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=settings.session_days * 24 * 60 * 60,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="strict",
        path="/",
        domain=settings.cookie_domain,
    )


def _current_session(request: Request) -> AuthenticatedSession:
    session = auth_service.authenticate(request.cookies.get(SESSION_COOKIE))
    if not session:
        raise HTTPException(status_code=401, detail="Authentification requise.")
    return session


CurrentSession = Annotated[AuthenticatedSession, Depends(_current_session)]


def _csrf_session(request: Request, session: CurrentSession) -> AuthenticatedSession:
    supplied = request.headers.get("X-CSRF-Token", "")
    if not supplied or supplied != session.csrf_token:
        raise HTTPException(status_code=403, detail="Jeton de sécurité invalide.")
    return session


CsrfSession = Annotated[AuthenticatedSession, Depends(_csrf_session)]


def _admin_session(session: CurrentSession) -> AuthenticatedSession:
    if session.user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Accès administrateur requis.")
    return session


AdminSession = Annotated[AuthenticatedSession, Depends(_admin_session)]


def _admin_csrf_session(
    request: Request, session: AdminSession
) -> AuthenticatedSession:
    supplied = request.headers.get("X-CSRF-Token", "")
    if not supplied or supplied != session.csrf_token:
        raise HTTPException(status_code=403, detail="Jeton de sécurité invalide.")
    return session


AdminCsrfSession = Annotated[AuthenticatedSession, Depends(_admin_csrf_session)]


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/auth/status")
def auth_status() -> dict:
    return {
        "setup_required": auth_service.setup_required,
        "registration_enabled": settings.self_registration_enabled,
        "guest_enabled": settings.guest_access_enabled,
        "password_reset_enabled": settings.password_reset_enabled,
    }


@app.post("/api/auth/setup", status_code=201)
def setup(payload: InitialSetup, request: Request) -> JSONResponse:
    try:
        auth_service.setup_admin(
            email=payload.email,
            display_name=payload.display_name,
            password=payload.password,
        )
        token, session = auth_service.login(
            email=payload.email,
            password=payload.password,
            user_agent=request.headers.get("user-agent", ""),
            ip_address=_client_ip(request),
        )
    except (ValueError, PermissionError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    response = JSONResponse(
        {"user": session.user, "csrf_token": session.csrf_token}, status_code=201
    )
    _set_session_cookie(response, token)
    return response


@app.post("/api/auth/login")
def login(payload: AuthCredentials, request: Request) -> JSONResponse:
    try:
        token, session = auth_service.login(
            email=payload.email,
            password=payload.password,
            user_agent=request.headers.get("user-agent", ""),
            ip_address=_client_ip(request),
        )
    except (ValueError, PermissionError) as error:
        raise HTTPException(status_code=401, detail=str(error)) from error
    response = JSONResponse({"user": session.user, "csrf_token": session.csrf_token})
    _set_session_cookie(response, token)
    return response


@app.post("/api/auth/register", status_code=201)
def register(payload: Registration, request: Request) -> JSONResponse:
    if not settings.self_registration_enabled or auth_service.setup_required:
        raise HTTPException(status_code=403, detail="La création de compte est désactivée.")
    try:
        auth_service.register(
            email=payload.email,
            display_name=payload.display_name,
            password=payload.password,
            ip_address=_client_ip(request),
        )
        token, session = auth_service.login(
            email=payload.email,
            password=payload.password,
            user_agent=request.headers.get("user-agent", ""),
            ip_address=_client_ip(request),
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        if "UNIQUE" in str(error).upper() or "DUPLICATE" in str(error).upper():
            raise HTTPException(
                status_code=409, detail="Un compte utilise déjà cette adresse."
            ) from error
        raise
    response = JSONResponse(
        {"user": session.user, "csrf_token": session.csrf_token}, status_code=201
    )
    _set_session_cookie(response, token)
    return response


@app.post("/api/auth/guest", status_code=201)
def guest_login(request: Request) -> JSONResponse:
    if not settings.guest_access_enabled or auth_service.setup_required:
        raise HTTPException(status_code=403, detail="L'accès invité est désactivé.")
    token, session = auth_service.create_guest_session(
        user_agent=request.headers.get("user-agent", ""),
        ip_address=_client_ip(request),
    )
    response = JSONResponse(
        {"user": session.user, "csrf_token": session.csrf_token}, status_code=201
    )
    _set_session_cookie(response, token)
    return response


@app.post("/api/auth/password/forgot", status_code=202)
def forgot_password(payload: PasswordResetRequest, request: Request) -> dict:
    if not settings.password_reset_enabled:
        raise HTTPException(status_code=403, detail="La récupération est désactivée.")
    try:
        raw_token = auth_service.request_password_reset(
            email=payload.email, ip_address=_client_ip(request)
        )
    except ValueError:
        raw_token = None
    preview_url = None
    if raw_token:
        user = database.get_user_by_email(payload.email)
        reset_url = f"{settings.public_url}/?reset_token={raw_token}"
        try:
            delivered = bool(user) and send_password_reset(
                recipient=user["email"],
                display_name=user["display_name"],
                reset_url=reset_url,
            )
        except Exception:
            logger.exception("Échec d'envoi du message de récupération")
            delivered = False
        if not delivered and settings.password_reset_preview:
            preview_url = reset_url
    result = {
        "message": (
            "Si cette adresse correspond à un compte actif, un lien de récupération "
            "vient d'être envoyé."
        )
    }
    if preview_url:
        result["preview_url"] = preview_url
    return result


@app.post("/api/auth/password/reset")
def reset_password(payload: PasswordResetComplete) -> dict:
    if not settings.password_reset_enabled:
        raise HTTPException(status_code=403, detail="La récupération est désactivée.")
    try:
        auth_service.reset_password(raw_token=payload.token, password=payload.password)
    except (ValueError, PermissionError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"message": "Mot de passe mis à jour. Vous pouvez vous connecter."}


@app.get("/api/auth/me")
def me(session: CurrentSession) -> dict:
    return {"user": session.user, "csrf_token": session.csrf_token}


@app.post("/api/auth/logout", status_code=204, response_class=Response)
def logout(session: CsrfSession) -> Response:
    auth_service.logout(session)
    response = Response(status_code=204)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@app.get("/api/auth/sessions")
def list_my_sessions(session: CurrentSession) -> list[dict]:
    rows = database.list_sessions(session.user["id"])
    for row in rows:
        row["current"] = row["id"] == session.session_id
    return rows


@app.delete("/api/auth/sessions/{session_id}", status_code=204, response_class=Response)
def revoke_my_session(session_id: str, session: CsrfSession) -> Response:
    if not database.revoke_session(session_id, user_id=session.user["id"]):
        raise HTTPException(status_code=404, detail="Session introuvable.")
    return Response(status_code=204)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "lisa-api"}


@app.get("/api/system/status")
def system_status(_: CurrentSession) -> dict:
    status = rag_service.status()
    return {
        "status": "error" if status["error"] else "ok",
        "service": status,
    }


@app.post("/api/pipeline/load")
def load_pipeline(_: CurrentSession) -> dict:
    if rag_service.status()["maintenance"]:
        raise HTTPException(status_code=503, detail="Maintenance en cours.")
    try:
        rag_service.get_pipeline()
        document_service.reconcile_saved_documents()
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return rag_service.status()


@app.get("/api/route")
def preview_route(
    _: CurrentSession,
    question: str = Query(min_length=1, max_length=4000),
) -> dict:
    return rag_service.preview_route(question)


@app.get("/api/conversations")
def list_conversations(
    session: CurrentSession, search: str = Query(default="", max_length=120)
) -> list[dict]:
    return database.list_conversations(session.user["id"], search)


@app.post("/api/conversations", status_code=201)
def create_conversation(payload: ConversationCreate, session: CsrfSession) -> dict:
    return database.create_conversation(session.user["id"], payload.title)


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str, session: CurrentSession) -> dict:
    conversation = database.get_conversation(conversation_id, session.user["id"])
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation introuvable.")
    return {
        **conversation,
        "messages": database.list_messages(conversation_id, session.user["id"]),
    }


@app.patch("/api/conversations/{conversation_id}")
def update_conversation(
    conversation_id: str, payload: ConversationUpdate, session: CsrfSession
) -> dict:
    try:
        conversation = database.rename_conversation(
            conversation_id, session.user["id"], payload.title
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation introuvable.")
    return conversation


@app.delete(
    "/api/conversations/{conversation_id}", status_code=204, response_class=Response
)
def delete_conversation(conversation_id: str, session: CsrfSession) -> Response:
    if not database.delete_conversation(conversation_id, session.user["id"]):
        raise HTTPException(status_code=404, detail="Conversation introuvable.")
    return Response(status_code=204)


def _automatic_title(question: str) -> str:
    compact = " ".join(question.split())
    return compact if len(compact) <= 64 else compact[:61].rstrip() + "…"


@app.post("/api/conversations/{conversation_id}/messages/stream")
def stream_message(
    conversation_id: str, payload: MessageCreate, session: CsrfSession
) -> StreamingResponse:
    if (
        rag_service.status()["maintenance"]
        and not rag_service.is_direct_conversation_turn(payload.content)
    ):
        raise HTTPException(
            status_code=503,
            detail="Une mise à jour documentaire est en cours. Réessayez à sa fin.",
        )
    conversation = database.get_conversation(conversation_id, session.user["id"])
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation introuvable.")

    question = payload.content.strip()
    direct_conversation_turn = rag_service.is_direct_conversation_turn(question)
    previous_messages = database.list_messages(conversation_id, session.user["id"])
    recent_history = previous_messages[-10:]
    if conversation.get("summary") and len(previous_messages) > 10:
        recent_history = [
            {
                "role": "assistant",
                "content": "Mémoire résumée de la conversation : " + conversation["summary"],
            },
            *recent_history,
        ]
    database.add_message(conversation_id, "user", question)
    if (
        conversation["title"] == "Nouvelle conversation"
        and not direct_conversation_turn
    ):
        database.rename_conversation(
            conversation_id, session.user["id"], _automatic_title(question)
        )

    def events() -> Iterator[str]:
        try:
            for event in rag_service.stream(
                question,
                conversation_history=recent_history,
                thinking=payload.thinking,
            ):
                if event["type"] == "done":
                    saved_message = database.add_message(
                        conversation_id,
                        "assistant",
                        event["answer"],
                        route=event["route"],
                        route_reason=event["route_reason"],
                        metrics=event["metrics"],
                        citations=event["citations"],
                        reasoning=event.get("reasoning"),
                    )
                    if event["route"] != "conversation":
                        memory = update_memory_summary(
                            conversation.get("summary", ""), question, event["answer"]
                        )
                        database.update_conversation_summary(
                            conversation_id, session.user["id"], memory
                        )
                    event = {**event, "message": saved_message}
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as error:
            logger.exception("Échec de génération pour la conversation %s", conversation_id)
            yield json.dumps(
                {
                    "type": "error",
                    "message": (
                        "La réponse n'a pas pu être générée. Vérifiez Ollama et "
                        "l'état de la maintenance dans l'espace administrateur."
                    ),
                    "detail": f"{type(error).__name__}: {error}",
                },
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@app.post("/api/messages/{message_id}/feedback")
def save_feedback(
    message_id: str, payload: FeedbackCreate, session: CsrfSession
) -> dict:
    try:
        return database.set_feedback(
            message_id, payload.value, payload.comment, user_id=session.user["id"]
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Message introuvable.") from error


@app.get("/api/metrics")
def metrics(session: CurrentSession) -> dict:
    return {
        "application": database.metrics(
            None if session.user["role"] == "admin" else session.user["id"]
        ),
        "pipeline": rag_service.status(),
        "deployment": {"offline_mode": settings.offline_mode},
    }


@app.get("/api/documents")
def list_documents(_: CurrentSession) -> dict:
    return {
        "documents": database.list_documents(),
        # La bibliothèque expose uniquement l'état des imports PDF. Les travaux
        # de collecte et d'indexation restent réservés à l'administration.
        "jobs": [
            job for job in database.list_jobs(limit=100)
            if job["kind"] == "pdf_ingestion"
        ][:20],
        "max_pdf_bytes": MAX_PDF_BYTES,
    }


@app.get("/api/source-files/{source_path:path}", response_class=FileResponse)
def open_source_document(source_path: str, _: CurrentSession) -> FileResponse:
    path = (SOURCE_DOCUMENTS_DIR / source_path).resolve()
    allowed_media_types = {
        ".pdf": "application/pdf",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12",
    }
    if (
        not path.is_relative_to(SOURCE_DOCUMENTS_DIR)
        or path.suffix.casefold() not in allowed_media_types
        or not path.is_file()
    ):
        raise HTTPException(status_code=404, detail="Document source introuvable.")
    return FileResponse(
        path,
        media_type=allowed_media_types[path.suffix.casefold()],
        filename=path.name,
    )


@app.post("/api/documents", status_code=202)
async def upload_document(
    session: AdminCsrfSession,
    file: UploadFile = File(...),
    title: str | None = Form(default=None, max_length=240),
    author: str | None = Form(default=None, max_length=160),
    document_date: str | None = Form(default=None, max_length=32),
    category: str = Form(default="rapport", max_length=80),
) -> dict:
    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Seuls les fichiers PDF sont acceptés.")
    content = await file.read(MAX_PDF_BYTES + 1)
    try:
        return document_service.submit(
            content=content,
            filename=filename,
            title=title,
            author=author,
            document_date=document_date,
            category=category,
            created_by=session.user["id"],
        )
    except FileExistsError as error:
        raise HTTPException(
            status_code=409,
            detail=f"Ce PDF est déjà enregistré (document {error.args[0]}).",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        await file.close()


@app.get("/api/documents/{document_id}")
def get_document(document_id: str, _: CurrentSession) -> dict:
    document = database.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document introuvable.")
    jobs = [job for job in database.list_jobs(limit=100) if job["document_id"] == document_id]
    return {"document": document, "jobs": jobs}


@app.get("/api/documents/{document_id}/file", response_class=FileResponse)
def open_document(document_id: str, _: CurrentSession) -> FileResponse:
    document = database.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document introuvable.")
    path = UPLOAD_DIR / document["stored_name"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Fichier PDF introuvable.")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=document["filename"],
        content_disposition_type="inline",
    )


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, _: AdminSession) -> dict:
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Tâche introuvable.")
    return job


# ---------------------------------------------------------------------------
# Console d'administration
# ---------------------------------------------------------------------------


@app.get("/api/admin/overview")
def admin_overview(_: AdminSession) -> dict:
    config = load_config(PROJECT_ROOT / "config.yaml")
    saved_settings = database.get_settings()
    return {
        "users": database.list_users(),
        "identity": database.identity_overview(),
        "security_sessions": database.list_all_sessions(limit=80),
        "audit_events": database.list_audit_events(limit=80),
        "jobs": database.list_jobs(limit=50),
        "schedules": database.list_schedules(),
        "settings": saved_settings,
        "pipeline": rag_service.status(),
        "deployment": {"offline_mode": settings.offline_mode},
        "metrics": database.metrics(),
        "tasks": ALLOWED_TASKS,
        "models": {
            "generation": [config.generation.model_name, *config.generation.alternatives],
            "embedding": config.embeddings.model_name,
            "embedding_alternatives": config.embeddings.alternatives,
            "reranker": config.reranker.model_name,
        },
        "facebook": {
            "pages": config.facebook_scraping.pages,
            "default_max_posts_per_page": config.facebook_scraping.max_posts_per_page,
            "actor_id": config.facebook_scraping.actor_id,
            "token_configured": admin_tasks.apify_token_configured,
        },
        "operations": {
            "institutional": {
                "base_url": config.project.base_url,
                "max_pages": config.scraping.max_pages,
                "max_concurrency": config.scraping.max_concurrency,
                "download_pdfs": config.scraping.download_pdfs,
            },
            "local_documents": {
                "source_dir": config.local_documents.source_dir,
                "pdf_count": sum(
                    1
                    for path in config.local_documents.source_full_path.rglob("*")
                    if path.is_file() and path.suffix.casefold() == ".pdf"
                ) if config.local_documents.source_full_path.exists() else 0,
            },
            "cleaning": {
                "min_content_length": config.cleaning.min_content_length,
                "corpora": ["institutionnel", "économique", "Facebook"],
            },
            "chunking": {
                "child_tokens": config.chunking.child_chunk_size_tokens,
                "parent_tokens": config.chunking.parent_chunk_size_tokens,
                "overlap_percent": round(config.chunking.chunk_overlap_ratio * 100),
            },
            "embeddings": {
                "model": config.embeddings.model_name,
                "device": config.embeddings.device,
                "precision": config.embeddings.precision,
                "batch_size": config.embeddings.batch_size,
            },
            "indexing": {
                "backend": config.vectorstore.backend,
                "dense": "LanceDB",
                "lexical": "BM25",
                "corpora": ["institutionnel", "économique", "Facebook"],
            },
        },
    }


@app.post("/api/admin/users", status_code=201)
def admin_create_user(payload: UserCreate, session: AdminCsrfSession) -> dict:
    try:
        return auth_service.create_user(
            actor_user_id=session.user["id"],
            email=payload.email,
            display_name=payload.display_name,
            password=payload.password,
            role=payload.role,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        if "UNIQUE" in str(error).upper():
            raise HTTPException(status_code=409, detail="Cette adresse existe déjà.") from error
        raise


@app.patch("/api/admin/users/{user_id}")
def admin_update_user(
    user_id: str, payload: UserUpdate, session: AdminCsrfSession
) -> dict:
    if user_id == session.user["id"] and (
        payload.is_active is False or payload.role == "user"
    ):
        raise HTTPException(
            status_code=422,
            detail="Vous ne pouvez pas retirer vos propres droits administrateur.",
        )
    target = database.get_user(user_id)
    if (
        target and target["role"] == "admin" and target["is_active"]
        and (payload.is_active is False or payload.role == "user")
        and database.active_admin_count() <= 1
    ):
        raise HTTPException(
            status_code=422, detail="Au moins un administrateur actif est requis."
        )
    password_hash = hash_password(payload.password) if payload.password else None
    user = database.update_user(
        user_id,
        display_name=payload.display_name,
        role=payload.role,
        is_active=payload.is_active,
        password_hash=password_hash,
    )
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable.")
    database.add_audit_event(
        "user.updated",
        actor_user_id=session.user["id"],
        target_type="user",
        target_id=user_id,
        details=payload.model_dump(exclude_none=True, exclude={"password"}),
    )
    return user


@app.delete(
    "/api/admin/sessions/{session_id}", status_code=204, response_class=Response
)
def admin_revoke_session(session_id: str, session: AdminCsrfSession) -> Response:
    if session_id == session.session_id:
        raise HTTPException(
            status_code=422,
            detail="Utilisez la déconnexion pour fermer votre session courante.",
        )
    if not database.revoke_session(session_id):
        raise HTTPException(status_code=404, detail="Session active introuvable.")
    database.add_audit_event(
        "auth.session_revoked_by_admin",
        actor_user_id=session.user["id"],
        target_type="session",
        target_id=session_id,
    )
    return Response(status_code=204)


@app.post("/api/admin/jobs", status_code=202)
def admin_create_job(payload: JobCreate, session: AdminCsrfSession) -> dict:
    if settings.offline_mode and payload.kind in {
        "collect_institutional", "collect_facebook", "full_refresh",
    }:
        raise HTTPException(
            status_code=409,
            detail=(
                "Le serveur fonctionne en mode hors ligne. Les collectes Web et "
                "Facebook nécessitent temporairement une connexion ; les imports "
                "locaux, chunks, embeddings et index restent disponibles."
            ),
        )
    try:
        job_payload = dict(payload.payload)
        if payload.kind in {"collect_facebook", "full_refresh"}:
            job_payload = _validated_facebook_payload(job_payload)
        else:
            job_payload = _validated_job_payload(payload.kind, job_payload)
        job = admin_tasks.submit(
            payload.kind,
            requested_by=session.user["id"],
            payload=job_payload,
            apify_token=payload.apify_token,
            remember_apify_token=payload.remember_apify_token,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    database.add_audit_event(
        "maintenance.started",
        actor_user_id=session.user["id"],
        target_type="job",
        target_id=job["id"],
        details={"kind": payload.kind},
    )
    return job


@app.post("/api/admin/schedules", status_code=201)
def admin_create_schedule(payload: ScheduleCreate, session: AdminCsrfSession) -> dict:
    if settings.offline_mode and payload.task_kind in {
        "collect_institutional", "collect_facebook", "full_refresh",
    }:
        raise HTTPException(
            status_code=409,
            detail="Les collectes réseau ne peuvent pas être planifiées en mode hors ligne.",
        )
    if payload.task_kind not in ALLOWED_TASKS:
        raise HTTPException(status_code=422, detail="Type de tâche non autorisé.")
    schedule_payload = dict(payload.payload)
    if payload.task_kind in {"collect_facebook", "full_refresh"}:
        if not admin_tasks.apify_token_configured:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Configurez d'abord le jeton Apify sur le serveur. Pour une "
                    "planification durable, définissez APIFY_TOKEN avant son démarrage."
                ),
            )
        try:
            schedule_payload = _validated_facebook_payload(schedule_payload)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    next_run = datetime.now(timezone.utc) + timedelta(minutes=payload.interval_minutes)
    schedule = database.create_schedule(
        name=payload.name,
        task_kind=payload.task_kind,
        interval_minutes=payload.interval_minutes,
        enabled=payload.enabled,
        payload=schedule_payload,
        next_run_at=next_run.isoformat(),
        created_by=session.user["id"],
    )
    database.add_audit_event(
        "schedule.created",
        actor_user_id=session.user["id"],
        target_type="schedule",
        target_id=schedule["id"],
    )
    return schedule


@app.patch("/api/admin/schedules/{schedule_id}")
def admin_update_schedule(
    schedule_id: str, payload: ScheduleUpdate, session: AdminCsrfSession
) -> dict:
    next_run = None
    if payload.interval_minutes is not None:
        next_run = (
            datetime.now(timezone.utc) + timedelta(minutes=payload.interval_minutes)
        ).isoformat()
    schedule = database.update_schedule(
        schedule_id,
        name=payload.name,
        interval_minutes=payload.interval_minutes,
        enabled=payload.enabled,
        next_run_at=next_run,
    )
    if not schedule:
        raise HTTPException(status_code=404, detail="Planification introuvable.")
    database.add_audit_event(
        "schedule.updated",
        actor_user_id=session.user["id"],
        target_type="schedule",
        target_id=schedule_id,
    )
    return schedule


@app.delete("/api/admin/schedules/{schedule_id}", status_code=204, response_class=Response)
def admin_delete_schedule(schedule_id: str, session: AdminCsrfSession) -> Response:
    if not database.delete_schedule(schedule_id):
        raise HTTPException(status_code=404, detail="Planification introuvable.")
    database.add_audit_event(
        "schedule.deleted",
        actor_user_id=session.user["id"],
        target_type="schedule",
        target_id=schedule_id,
    )
    return Response(status_code=204)


@app.patch("/api/admin/settings")
def admin_update_settings(payload: SettingsUpdate, session: AdminCsrfSession) -> dict:
    config = load_config(PROJECT_ROOT / "config.yaml")
    values = payload.model_dump(exclude_none=True)
    model = values.get("generation_model")
    allowed_models = {config.generation.model_name, *config.generation.alternatives}
    if model and model not in allowed_models:
        raise HTTPException(status_code=422, detail="Modèle de génération non autorisé.")
    settings = database.set_settings(values, updated_by=session.user["id"])
    rag_service.reload()
    database.add_audit_event(
        "settings.updated",
        actor_user_id=session.user["id"],
        target_type="system_settings",
        details=values,
    )
    return {"settings": settings, "pipeline": rag_service.status()}


@app.get("/api/admin/audit")
def admin_audit(
    _: AdminSession, limit: int = Query(default=100, ge=1, le=500)
) -> list[dict]:
    return database.list_audit_events(limit)
