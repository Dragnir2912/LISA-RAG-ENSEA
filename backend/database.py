"""Persistance transactionnelle de l'application conversationnelle RAG ENSEA.

SQLite/WAL est le moteur local par défaut : il est fiable, sans service annexe et
adapté au serveur mono-machine qui pilote l'unique GPU. Toute l'identité et tous
les contrôles d'accès restent néanmoins appliqués dans ce dépôt afin que la
couche API puisse être migrée vers PostgreSQL sans modifier le moteur RAG.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "app" / "rag_ensea.sqlite3"


class _HybridRow(dict):
    """Ligne compatible avec ``sqlite3.Row`` et les dictionnaires psycopg."""

    def __getitem__(self, key):
        if isinstance(key, int):
            return tuple(self.values())[key]
        return super().__getitem__(key)


class _CursorAdapter:
    def __init__(self, cursor) -> None:
        self._cursor = cursor

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    @staticmethod
    def _row(value):
        return _HybridRow(value) if isinstance(value, dict) else value

    def fetchone(self):
        return self._row(self._cursor.fetchone())

    def fetchall(self):
        return [self._row(row) for row in self._cursor.fetchall()]

    def __iter__(self):
        for row in self._cursor:
            yield self._row(row)


class _PostgresConnection:
    dialect = "postgresql"

    def __init__(self, raw) -> None:
        self.raw = raw

    @staticmethod
    def _sql(statement: str) -> str:
        return (
            statement.replace(" COLLATE NOCASE", "")
            .replace("ORDER BY created_at, rowid", "ORDER BY created_at, id")
            .replace("?", "%s")
        )

    def execute(self, statement: str, parameters=()) -> _CursorAdapter:
        from psycopg.rows import dict_row

        cursor = self.raw.cursor(row_factory=dict_row)
        cursor.execute(self._sql(statement), parameters)
        return _CursorAdapter(cursor)

    def executescript(self, script: str) -> None:
        for statement in script.split(";"):
            if statement.strip():
                self.execute(statement)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AppDatabase:
    """Dépôt partagé : SQLite en local, PostgreSQL sur les serveurs."""

    def __init__(
        self,
        path: Path = DEFAULT_DATABASE_PATH,
        *,
        database_url: str | None = None,
    ) -> None:
        self.database_url = (database_url or os.getenv("RAG_DATABASE_URL") or "").strip()
        self.is_postgres = self.database_url.startswith(("postgresql://", "postgres://"))
        if self.database_url.startswith("sqlite:///"):
            self.path = Path(self.database_url.removeprefix("sqlite:///"))
        else:
            self.path = Path(path)
        if not self.is_postgres:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        if self.is_postgres:
            try:
                import psycopg
            except ImportError as error:
                raise RuntimeError(
                    "Le paquet psycopg[binary] est requis pour PostgreSQL."
                ) from error
            raw = psycopg.connect(self.database_url)
            connection = _PostgresConnection(raw)
            try:
                yield connection
                raw.commit()
            except Exception:
                raw.rollback()
                raise
            finally:
                raw.close()
            return
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connect() as connection:
            if not self.is_postgres:
                connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('admin', 'user')),
                    is_active INTEGER NOT NULL DEFAULT 1,
                    failed_login_count INTEGER NOT NULL DEFAULT 0,
                    locked_until TEXT,
                    last_login_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    account_type TEXT NOT NULL DEFAULT 'registered',
                    email_verified INTEGER NOT NULL DEFAULT 0,
                    registration_ip TEXT
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    csrf_token TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    user_agent TEXT,
                    ip_address TEXT,
                    revoked_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    archived_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    route TEXT,
                    route_reason TEXT,
                    queue_seconds REAL,
                    time_to_first_token_seconds REAL,
                    retrieval_seconds REAL,
                    generation_seconds REAL,
                    total_seconds REAL,
                    prompt_tokens INTEGER,
                    generated_tokens INTEGER,
                    tokens_per_second REAL,
                    thinking_enabled INTEGER NOT NULL DEFAULT 0,
                    model_name TEXT,
                    reasoning_summary_key TEXT,
                    reasoning_content TEXT,
                    reasoning_seconds REAL,
                    reasoning_truncated INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS citations (
                    id TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL,
                    citation_index INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    section TEXT,
                    url TEXT NOT NULL,
                    passage TEXT NOT NULL,
                    rerank_score REAL,
                    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    message_id TEXT PRIMARY KEY,
                    value TEXT NOT NULL CHECK (value IN ('up', 'down')),
                    comment TEXT,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    created_by TEXT,
                    filename TEXT NOT NULL,
                    stored_name TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    author TEXT,
                    document_date TEXT,
                    category TEXT NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    size_bytes INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('queued', 'processing', 'ready', 'error')
                    ),
                    page_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS maintenance_jobs (
                    id TEXT PRIMARY KEY,
                    document_id TEXT,
                    requested_by TEXT,
                    schedule_id TEXT,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL CHECK (
                        status IN ('queued', 'running', 'completed', 'error')
                    ),
                    progress INTEGER NOT NULL DEFAULT 0,
                    message TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                    FOREIGN KEY (requested_by) REFERENCES users(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS schedules (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    task_kind TEXT NOT NULL,
                    interval_minutes INTEGER NOT NULL CHECK (interval_minutes >= 5),
                    enabled INTEGER NOT NULL DEFAULT 1,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_by TEXT,
                    last_run_at TEXT,
                    next_run_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS system_settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_by TEXT,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (updated_by) REFERENCES users(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    actor_user_id TEXT,
                    action TEXT NOT NULL,
                    target_type TEXT,
                    target_id TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (actor_user_id) REFERENCES users(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS password_reset_tokens (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    used_at TEXT,
                    requested_ip TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_conversations_updated_at
                    ON conversations(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
                    ON messages(conversation_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_citations_message
                    ON citations(message_id, citation_index);
                CREATE INDEX IF NOT EXISTS idx_documents_status_updated
                    ON documents(status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_maintenance_jobs_created
                    ON maintenance_jobs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_sessions_token_active
                    ON sessions(token_hash, revoked_at, expires_at);
                CREATE INDEX IF NOT EXISTS idx_sessions_user_active
                    ON sessions(user_id, revoked_at, expires_at);
                CREATE INDEX IF NOT EXISTS idx_schedules_due
                    ON schedules(enabled, next_run_at);
                CREATE INDEX IF NOT EXISTS idx_audit_created
                    ON audit_events(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_password_reset_active
                    ON password_reset_tokens(token_hash, used_at, expires_at);
                """
            )
            # Migration non destructive des bases créées par les premières
            # versions locales de l'application.
            self._ensure_column(connection, "conversations", "user_id", "TEXT")
            self._ensure_column(
                connection, "conversations", "summary", "TEXT NOT NULL DEFAULT ''"
            )
            self._ensure_column(connection, "conversations", "archived_at", "TEXT")
            self._ensure_column(connection, "documents", "created_by", "TEXT")
            self._ensure_column(
                connection, "users", "account_type", "TEXT NOT NULL DEFAULT 'registered'"
            )
            self._ensure_column(
                connection, "users", "email_verified", "INTEGER NOT NULL DEFAULT 0"
            )
            self._ensure_column(connection, "users", "registration_ip", "TEXT")
            self._ensure_column(connection, "messages", "queue_seconds", "REAL")
            self._ensure_column(
                connection, "messages", "time_to_first_token_seconds", "REAL"
            )
            self._ensure_column(
                connection,
                "messages",
                "thinking_enabled",
                "INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(connection, "messages", "model_name", "TEXT")
            self._ensure_column(
                connection, "messages", "reasoning_summary_key", "TEXT"
            )
            self._ensure_column(connection, "messages", "reasoning_content", "TEXT")
            self._ensure_column(connection, "messages", "reasoning_seconds", "REAL")
            self._ensure_column(
                connection,
                "messages",
                "reasoning_truncated",
                "INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(connection, "maintenance_jobs", "requested_by", "TEXT")
            self._ensure_column(connection, "maintenance_jobs", "schedule_id", "TEXT")
            self._ensure_column(
                connection,
                "maintenance_jobs",
                "payload_json",
                "TEXT NOT NULL DEFAULT '{}'",
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_user_updated "
                "ON conversations(user_id, archived_at, updated_at DESC)"
            )
            if not self.is_postgres:
                connection.execute("PRAGMA optimize")

    def _ensure_column(
        self, connection: Any, table: str, column: str, definition: str
    ) -> None:
        if self.is_postgres:
            columns = {
                row["column_name"]
                for row in connection.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = ?",
                    (table,),
                )
            }
        else:
            columns = {
                row["name"] for row in connection.execute(f"PRAGMA table_info({table})")
            }
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    # ------------------------------------------------------------------
    # Identité, sessions et audit
    # ------------------------------------------------------------------

    @staticmethod
    def _user_from_row(row: sqlite3.Row, *, include_secret: bool = False) -> dict[str, Any]:
        user = {
            "id": row["id"],
            "email": row["email"],
            "display_name": row["display_name"],
            "role": row["role"],
            "is_active": bool(row["is_active"]),
            "failed_login_count": int(row["failed_login_count"]),
            "locked_until": row["locked_until"],
            "last_login_at": row["last_login_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "account_type": row["account_type"],
            "email_verified": bool(row["email_verified"]),
            "registration_ip": row["registration_ip"],
        }
        if include_secret:
            user["password_hash"] = row["password_hash"]
        return user

    def user_count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def active_admin_count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute(
                "SELECT COUNT(*) FROM users WHERE role = 'admin' AND is_active = 1"
            ).fetchone()[0])

    def create_user(
        self,
        *,
        email: str,
        display_name: str,
        password_hash: str,
        role: str = "user",
        adopt_legacy_records: bool = False,
        account_type: str = "registered",
        email_verified: bool = False,
        registration_ip: str = "",
    ) -> dict[str, Any]:
        if role not in {"admin", "user"}:
            raise ValueError("Rôle utilisateur invalide.")
        if account_type not in {"registered", "guest"}:
            raise ValueError("Type de compte invalide.")
        user_id = uuid.uuid4().hex
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO users(
                    id, email, display_name, password_hash, role,
                    is_active, created_at, updated_at, account_type,
                    email_verified, registration_ip
                ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    email.strip().casefold(),
                    display_name.strip()[:120],
                    password_hash,
                    role,
                    timestamp,
                    timestamp,
                    account_type,
                    int(email_verified),
                    registration_ip[:80] or None,
                ),
            )
            if adopt_legacy_records:
                connection.execute(
                    "UPDATE conversations SET user_id = ? WHERE user_id IS NULL",
                    (user_id,),
                )
                connection.execute(
                    "UPDATE documents SET created_by = ? WHERE created_by IS NULL",
                    (user_id,),
                )
        return self.get_user(user_id) or {}

    def get_user(self, user_id: str, *, include_secret: bool = False) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return self._user_from_row(row, include_secret=include_secret) if row else None

    def get_user_by_email(
        self, email: str, *, include_secret: bool = False
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE email = ? COLLATE NOCASE",
                (email.strip().casefold(),),
            ).fetchone()
        return self._user_from_row(row, include_secret=include_secret) if row else None

    def list_users(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM users ORDER BY display_name COLLATE NOCASE, email"
            ).fetchall()
        return [self._user_from_row(row) for row in rows]

    def create_password_reset_token(
        self,
        *,
        user_id: str,
        token_hash: str,
        expires_at: str,
        requested_ip: str = "",
    ) -> dict[str, Any]:
        token_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                "UPDATE password_reset_tokens SET used_at = ? "
                "WHERE user_id = ? AND used_at IS NULL",
                (utc_now(), user_id),
            )
            connection.execute(
                """
                INSERT INTO password_reset_tokens(
                    id, user_id, token_hash, expires_at, created_at, requested_ip
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (token_id, user_id, token_hash, expires_at, utc_now(), requested_ip[:80]),
            )
        return {"id": token_id, "user_id": user_id, "expires_at": expires_at}

    def consume_password_reset_token(self, token_hash: str) -> dict[str, Any] | None:
        now = utc_now()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM password_reset_tokens
                WHERE token_hash = ? AND used_at IS NULL AND expires_at > ?
                """,
                (token_hash, now),
            ).fetchone()
            if not row:
                return None
            connection.execute(
                "UPDATE password_reset_tokens SET used_at = ? WHERE id = ?",
                (now, row["id"]),
            )
        return dict(row)

    def identity_overview(self) -> dict[str, int]:
        now = utc_now()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM users WHERE account_type = 'registered') AS registered,
                    (SELECT COUNT(*) FROM users WHERE account_type = 'guest') AS guests,
                    (SELECT COUNT(*) FROM users WHERE is_active = 1) AS active_users,
                    (SELECT COUNT(*) FROM sessions
                     WHERE revoked_at IS NULL AND expires_at > ?) AS active_sessions,
                    (SELECT COUNT(*) FROM password_reset_tokens) AS recovery_requests,
                    (SELECT COUNT(*) FROM users
                     WHERE created_at >= ?) AS registrations_24h
                """,
                (now, (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()),
            ).fetchone()
        return {key: int(value or 0) for key, value in dict(row).items()}

    def purge_expired_identity_data(self, guest_retention_days: int = 7) -> dict[str, int]:
        """Supprime les comptes invités expirés et les jetons devenus inutiles."""
        guest_cutoff = (
            datetime.now(timezone.utc) - timedelta(days=max(1, guest_retention_days))
        ).isoformat()
        now = utc_now()
        with self._connect() as connection:
            guest_cursor = connection.execute(
                "DELETE FROM users WHERE account_type = 'guest' AND created_at < ?",
                (guest_cutoff,),
            )
            reset_cursor = connection.execute(
                "DELETE FROM password_reset_tokens "
                "WHERE expires_at < ? OR used_at IS NOT NULL",
                (now,),
            )
            session_cursor = connection.execute(
                "DELETE FROM sessions WHERE expires_at < ? OR revoked_at IS NOT NULL",
                (now,),
            )
        return {
            "guests": max(0, guest_cursor.rowcount),
            "reset_tokens": max(0, reset_cursor.rowcount),
            "sessions": max(0, session_cursor.rowcount),
        }

    def update_user(
        self,
        user_id: str,
        *,
        display_name: str | None = None,
        role: str | None = None,
        is_active: bool | None = None,
        password_hash: str | None = None,
    ) -> dict[str, Any] | None:
        user = self.get_user(user_id)
        if not user:
            return None
        if role is not None and role not in {"admin", "user"}:
            raise ValueError("Rôle utilisateur invalide.")
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE users SET display_name = ?, role = ?, is_active = ?,
                    password_hash = COALESCE(?, password_hash), updated_at = ?
                WHERE id = ?
                """,
                (
                    (display_name or user["display_name"]).strip()[:120],
                    role or user["role"],
                    int(user["is_active"] if is_active is None else is_active),
                    password_hash,
                    utc_now(),
                    user_id,
                ),
            )
            if is_active is False or password_hash is not None:
                connection.execute(
                    "UPDATE sessions SET revoked_at = ? "
                    "WHERE user_id = ? AND revoked_at IS NULL",
                    (utc_now(), user_id),
                )
        return self.get_user(user_id)

    def record_login_failure(self, user_id: str, locked_until: str | None) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE users SET failed_login_count = failed_login_count + 1,
                    locked_until = ?, updated_at = ? WHERE id = ?
                """,
                (locked_until, utc_now(), user_id),
            )

    def record_login_success(self, user_id: str) -> None:
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE users SET failed_login_count = 0, locked_until = NULL,
                    last_login_at = ?, updated_at = ? WHERE id = ?
                """,
                (timestamp, timestamp, user_id),
            )

    def create_session(
        self,
        *,
        user_id: str,
        token_hash: str,
        csrf_token: str,
        expires_at: str,
        user_agent: str = "",
        ip_address: str = "",
    ) -> dict[str, Any]:
        session_id = uuid.uuid4().hex
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions(
                    id, user_id, token_hash, csrf_token, expires_at,
                    created_at, last_seen_at, user_agent, ip_address
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id, user_id, token_hash, csrf_token, expires_at,
                    timestamp, timestamp, user_agent[:300], ip_address[:80],
                ),
            )
        return self.get_session_by_hash(token_hash) or {}

    def get_session_by_hash(self, token_hash: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT s.*, u.email, u.display_name, u.role, u.is_active,
                       u.account_type
                FROM sessions s JOIN users u ON u.id = s.user_id
                WHERE s.token_hash = ?
                """,
                (token_hash,),
            ).fetchone()
        return dict(row) if row else None

    def touch_session(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET last_seen_at = ? WHERE id = ?",
                (utc_now(), session_id),
            )

    def revoke_session(self, session_id: str, *, user_id: str | None = None) -> bool:
        with self._connect() as connection:
            if user_id:
                cursor = connection.execute(
                    "UPDATE sessions SET revoked_at = ? "
                    "WHERE id = ? AND user_id = ? AND revoked_at IS NULL",
                    (utc_now(), session_id, user_id),
                )
            else:
                cursor = connection.execute(
                    "UPDATE sessions SET revoked_at = ? "
                    "WHERE id = ? AND revoked_at IS NULL",
                    (utc_now(), session_id),
                )
        return cursor.rowcount > 0

    def list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, user_id, expires_at, created_at, last_seen_at,
                       user_agent, ip_address, revoked_at
                FROM sessions WHERE user_id = ? ORDER BY last_seen_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_all_sessions(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT s.id, s.user_id, s.expires_at, s.created_at, s.last_seen_at,
                       s.user_agent, s.ip_address, s.revoked_at,
                       u.email, u.display_name, u.account_type
                FROM sessions s JOIN users u ON u.id = s.user_id
                ORDER BY s.last_seen_at DESC LIMIT ?
                """,
                (max(1, min(int(limit), 500)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_audit_event(
        self,
        action: str,
        *,
        actor_user_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO audit_events(
                    id, actor_user_id, action, target_type, target_id,
                    details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex, actor_user_id, action, target_type,
                    target_id, json.dumps(details or {}, ensure_ascii=False), utc_now(),
                ),
            )

    def list_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT a.*, u.email AS actor_email
                FROM audit_events a LEFT JOIN users u ON u.id = a.actor_user_id
                ORDER BY a.created_at DESC LIMIT ?
                """,
                (max(1, min(int(limit), 500)),),
            ).fetchall()
        events = []
        for row in rows:
            event = dict(row)
            event["details"] = json.loads(event.pop("details_json") or "{}")
            events.append(event)
        return events

    @staticmethod
    def _conversation_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "title": row["title"],
            "summary": row["summary"],
            "archived_at": row["archived_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def create_conversation(
        self, user_id: str, title: str = "Nouvelle conversation"
    ) -> dict[str, Any]:
        conversation_id = uuid.uuid4().hex
        timestamp = utc_now()
        normalized_title = title.strip()[:120] or "Nouvelle conversation"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations(
                    id, user_id, title, summary, created_at, updated_at
                ) VALUES (?, ?, ?, '', ?, ?)
                """,
                (conversation_id, user_id, normalized_title, timestamp, timestamp),
            )
        return self.get_conversation(conversation_id, user_id) or {}

    def get_conversation(
        self, conversation_id: str, user_id: str | None = None
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            if user_id:
                row = connection.execute(
                    "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
                    (conversation_id, user_id),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
                ).fetchone()
        return self._conversation_from_row(row) if row else None

    def list_conversations(self, user_id: str, search: str = "") -> list[dict[str, Any]]:
        with self._connect() as connection:
            if search.strip():
                rows = connection.execute(
                    """
                    SELECT *
                    FROM conversations
                    WHERE user_id = ? AND archived_at IS NULL
                      AND title LIKE ? ESCAPE '\\'
                    ORDER BY updated_at DESC
                    """,
                    (user_id, f"%{search.strip()}%"),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM conversations
                    WHERE user_id = ? AND archived_at IS NULL
                    ORDER BY updated_at DESC
                    """,
                    (user_id,),
                ).fetchall()
        return [self._conversation_from_row(row) for row in rows]

    def rename_conversation(
        self, conversation_id: str, user_id: str, title: str
    ) -> dict[str, Any] | None:
        normalized_title = title.strip()[:120]
        if not normalized_title:
            raise ValueError("Le titre ne peut pas être vide.")
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE conversations SET title = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (normalized_title, utc_now(), conversation_id, user_id),
            )
        return self.get_conversation(conversation_id, user_id)

    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            )
        return cursor.rowcount > 0

    def update_conversation_summary(
        self, conversation_id: str, user_id: str, summary: str
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE conversations SET summary = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (summary.strip()[:4000], utc_now(), conversation_id, user_id),
            )

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        route: str | None = None,
        route_reason: str | None = None,
        metrics: dict[str, Any] | None = None,
        citations: list[dict[str, Any]] | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if role not in {"user", "assistant"}:
            raise ValueError("Rôle de message invalide.")
        message_id = uuid.uuid4().hex
        timestamp = utc_now()
        metrics = metrics or {}
        reasoning = reasoning or {}
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if not exists:
                raise KeyError(conversation_id)
            connection.execute(
                """
                INSERT INTO messages(
                    id, conversation_id, role, content, route, route_reason,
                    queue_seconds, time_to_first_token_seconds,
                    retrieval_seconds, generation_seconds, total_seconds,
                    prompt_tokens, generated_tokens, tokens_per_second,
                    thinking_enabled, model_name, reasoning_summary_key,
                    reasoning_content, reasoning_seconds, reasoning_truncated,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    role,
                    content,
                    route,
                    route_reason,
                    metrics.get("queue_seconds"),
                    metrics.get("time_to_first_token_seconds"),
                    metrics.get("retrieval_seconds"),
                    metrics.get("generation_seconds"),
                    metrics.get("total_seconds"),
                    metrics.get("prompt_tokens"),
                    metrics.get("generated_tokens"),
                    metrics.get("tokens_per_second"),
                    int(bool(metrics.get("thinking_enabled", False))),
                    metrics.get("model_name"),
                    reasoning.get("summary_key"),
                    reasoning.get("content"),
                    reasoning.get("seconds"),
                    int(bool(reasoning.get("truncated", False))),
                    timestamp,
                ),
            )
            for citation in citations or []:
                connection.execute(
                    """
                    INSERT INTO citations(
                        id, message_id, citation_index, title, section,
                        url, passage, rerank_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        message_id,
                        int(citation.get("index", 0)),
                        str(citation.get("title") or "Source"),
                        str(citation.get("section") or ""),
                        str(citation.get("url") or ""),
                        str(citation.get("passage") or ""),
                        citation.get("rerank_score"),
                    ),
                )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (timestamp, conversation_id),
            )
        return self.get_message(message_id)

    def _citations_for_message(
        self, connection: sqlite3.Connection, message_id: str
    ) -> list[dict[str, Any]]:
        rows = connection.execute(
            """
            SELECT citation_index, title, section, url, passage, rerank_score
            FROM citations WHERE message_id = ? ORDER BY citation_index
            """,
            (message_id,),
        ).fetchall()
        return [
            {
                "index": row["citation_index"],
                "title": row["title"],
                "section": row["section"],
                "url": row["url"],
                "passage": row["passage"],
                "rerank_score": row["rerank_score"],
            }
            for row in rows
        ]

    def _message_from_row(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> dict[str, Any]:
        feedback_row = connection.execute(
            "SELECT value, comment, updated_at FROM feedback WHERE message_id = ?",
            (row["id"],),
        ).fetchone()
        feedback_value = dict(feedback_row) if feedback_row else None
        return {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "role": row["role"],
            "content": row["content"],
            "route": row["route"],
            "route_reason": row["route_reason"],
            "reasoning": (
                {
                    "summary_key": row["reasoning_summary_key"] or "general",
                    "content": row["reasoning_content"] or "",
                    "seconds": row["reasoning_seconds"] or 0.0,
                    "truncated": bool(row["reasoning_truncated"]),
                }
                if row["reasoning_content"]
                else None
            ),
            "metrics": {
                "queue_seconds": row["queue_seconds"],
                "time_to_first_token_seconds": row["time_to_first_token_seconds"],
                "retrieval_seconds": row["retrieval_seconds"],
                "generation_seconds": row["generation_seconds"],
                "total_seconds": row["total_seconds"],
                "prompt_tokens": row["prompt_tokens"],
                "generated_tokens": row["generated_tokens"],
                "tokens_per_second": row["tokens_per_second"],
                "thinking_enabled": bool(row["thinking_enabled"]),
                "model_name": row["model_name"],
            },
            "citations": self._citations_for_message(connection, row["id"]),
            "feedback": feedback_value,
            "created_at": row["created_at"],
        }

    def get_message(
        self, message_id: str, user_id: str | None = None
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            if user_id:
                row = connection.execute(
                    """
                    SELECT m.* FROM messages m
                    JOIN conversations c ON c.id = m.conversation_id
                    WHERE m.id = ? AND c.user_id = ?
                    """,
                    (message_id, user_id),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM messages WHERE id = ?", (message_id,)
                ).fetchone()
            return self._message_from_row(connection, row) if row else None

    def list_messages(
        self,
        conversation_id: str,
        user_id: str | None = None,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if user_id and not connection.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone():
                return []
            limit_clause = " LIMIT ?" if limit else ""
            parameters: tuple[Any, ...] = (conversation_id,)
            if limit:
                parameters += (max(1, min(int(limit), 500)),)
            rows = connection.execute(
                f"""
                SELECT * FROM messages
                WHERE conversation_id = ? ORDER BY created_at, rowid
                {limit_clause}
                """,
                parameters,
            ).fetchall()
            return [self._message_from_row(connection, row) for row in rows]

    def set_feedback(
        self,
        message_id: str,
        value: str,
        comment: str | None = None,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        if value not in {"up", "down"}:
            raise ValueError("Le retour doit être 'up' ou 'down'.")
        timestamp = utc_now()
        with self._connect() as connection:
            if user_id:
                exists = connection.execute(
                    """
                    SELECT 1 FROM messages m
                    JOIN conversations c ON c.id = m.conversation_id
                    WHERE m.id = ? AND m.role = 'assistant' AND c.user_id = ?
                    """,
                    (message_id, user_id),
                ).fetchone()
            else:
                exists = connection.execute(
                    "SELECT 1 FROM messages WHERE id = ? AND role = 'assistant'",
                    (message_id,),
                ).fetchone()
            if not exists:
                raise KeyError(message_id)
            connection.execute(
                """
                INSERT INTO feedback(message_id, value, comment, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    value = excluded.value,
                    comment = excluded.comment,
                    updated_at = excluded.updated_at
                """,
                (message_id, value, (comment or "").strip()[:1000] or None, timestamp),
            )
        return self.get_message(message_id, user_id) or {}

    def metrics(self, user_id: str | None = None) -> dict[str, Any]:
        with self._connect() as connection:
            if user_id:
                counts = connection.execute(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM conversations WHERE user_id = ?) AS conversations,
                        (SELECT COUNT(*) FROM messages m JOIN conversations c
                         ON c.id = m.conversation_id
                         WHERE m.role = 'user' AND c.user_id = ?) AS questions,
                        (SELECT COUNT(*) FROM feedback f JOIN messages m ON m.id = f.message_id
                         JOIN conversations c ON c.id = m.conversation_id
                         WHERE f.value = 'up' AND c.user_id = ?) AS positive_feedback,
                        (SELECT COUNT(*) FROM feedback f JOIN messages m ON m.id = f.message_id
                         JOIN conversations c ON c.id = m.conversation_id
                         WHERE f.value = 'down' AND c.user_id = ?) AS negative_feedback,
                        (SELECT COUNT(*) FROM documents WHERE status = 'ready') AS indexed_documents,
                        (SELECT COALESCE(SUM(chunk_count), 0) FROM documents
                         WHERE status = 'ready') AS uploaded_chunks,
                        (SELECT AVG(m.total_seconds) FROM messages m JOIN conversations c
                         ON c.id = m.conversation_id
                         WHERE m.role = 'assistant' AND c.user_id = ?) AS avg_seconds,
                        (SELECT AVG(m.queue_seconds) FROM messages m JOIN conversations c
                         ON c.id = m.conversation_id
                         WHERE m.role = 'assistant' AND c.user_id = ?) AS avg_queue_seconds,
                        (SELECT AVG(m.time_to_first_token_seconds) FROM messages m
                         JOIN conversations c ON c.id = m.conversation_id
                         WHERE m.role = 'assistant' AND c.user_id = ?) AS avg_first_token_seconds,
                        (SELECT AVG(m.tokens_per_second) FROM messages m JOIN conversations c
                         ON c.id = m.conversation_id
                         WHERE m.role = 'assistant' AND c.user_id = ?) AS avg_tps
                    """,
                    (
                        user_id, user_id, user_id, user_id,
                        user_id, user_id, user_id, user_id,
                    ),
                ).fetchone()
                routes = connection.execute(
                    """
                    SELECT COALESCE(m.route, 'unknown') AS route, COUNT(*) AS count
                    FROM messages m JOIN conversations c ON c.id = m.conversation_id
                    WHERE m.role = 'assistant' AND c.user_id = ?
                    GROUP BY COALESCE(m.route, 'unknown')
                    """,
                    (user_id,),
                ).fetchall()
            else:
                counts = connection.execute(
                    """
                SELECT
                    (SELECT COUNT(*) FROM conversations) AS conversations,
                    (SELECT COUNT(*) FROM messages WHERE role = 'user') AS questions,
                    (SELECT COUNT(*) FROM feedback WHERE value = 'up') AS positive_feedback,
                    (SELECT COUNT(*) FROM feedback WHERE value = 'down') AS negative_feedback,
                    (SELECT COUNT(*) FROM documents WHERE status = 'ready') AS indexed_documents,
                    (SELECT COALESCE(SUM(chunk_count), 0) FROM documents WHERE status = 'ready') AS uploaded_chunks,
                    (SELECT AVG(total_seconds) FROM messages WHERE role = 'assistant') AS avg_seconds,
                    (SELECT AVG(queue_seconds) FROM messages WHERE role = 'assistant') AS avg_queue_seconds,
                    (SELECT AVG(time_to_first_token_seconds) FROM messages
                     WHERE role = 'assistant') AS avg_first_token_seconds,
                    (SELECT AVG(tokens_per_second) FROM messages WHERE role = 'assistant') AS avg_tps
                    """
                ).fetchone()
                routes = connection.execute(
                    """
                    SELECT COALESCE(route, 'unknown') AS route, COUNT(*) AS count
                    FROM messages WHERE role = 'assistant'
                    GROUP BY COALESCE(route, 'unknown')
                    """
                ).fetchall()
        result = dict(counts)
        result["avg_seconds"] = float(result["avg_seconds"] or 0.0)
        result["avg_queue_seconds"] = float(result["avg_queue_seconds"] or 0.0)
        result["avg_first_token_seconds"] = float(
            result["avg_first_token_seconds"] or 0.0
        )
        result["avg_tps"] = float(result["avg_tps"] or 0.0)
        result["routes"] = {row["route"]: row["count"] for row in routes}
        return result

    @staticmethod
    def _document_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "created_by": row["created_by"],
            "filename": row["filename"],
            "stored_name": row["stored_name"],
            "title": row["title"],
            "author": row["author"],
            "document_date": row["document_date"],
            "category": row["category"],
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
            "status": row["status"],
            "page_count": row["page_count"],
            "chunk_count": row["chunk_count"],
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def create_document(
        self,
        *,
        filename: str,
        stored_name: str,
        title: str,
        author: str | None,
        document_date: str | None,
        category: str,
        sha256: str,
        size_bytes: int,
        created_by: str | None = None,
    ) -> dict[str, Any]:
        document_id = Path(stored_name).stem
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents(
                    id, created_by, filename, stored_name, title, author, document_date,
                    category, sha256, size_bytes, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    document_id,
                    created_by,
                    filename[:255],
                    stored_name,
                    title.strip()[:240],
                    (author or "").strip()[:160] or None,
                    (document_date or "").strip()[:32] or None,
                    category.strip()[:80] or "document",
                    sha256,
                    int(size_bytes),
                    timestamp,
                    timestamp,
                ),
            )
        return self.get_document(document_id) or {}

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return self._document_from_row(row) if row else None

    def find_document_by_hash(self, sha256: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE sha256 = ?", (sha256,)
            ).fetchone()
        return self._document_from_row(row) if row else None

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY updated_at DESC"
            ).fetchall()
        return [self._document_from_row(row) for row in rows]

    def update_document_status(
        self,
        document_id: str,
        status: str,
        *,
        page_count: int | None = None,
        chunk_count: int | None = None,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        if status not in {"queued", "processing", "ready", "error"}:
            raise ValueError("Statut de document invalide.")
        document = self.get_document(document_id)
        if not document:
            return None
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents SET
                    status = ?, page_count = ?, chunk_count = ?, error = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    document["page_count"] if page_count is None else int(page_count),
                    document["chunk_count"] if chunk_count is None else int(chunk_count),
                    error,
                    utc_now(),
                    document_id,
                ),
            )
        return self.get_document(document_id)

    @staticmethod
    def _job_from_row(
        row: sqlite3.Row, *, include_secrets: bool = False
    ) -> dict[str, Any]:
        job = {
            "id": row["id"],
            "document_id": row["document_id"],
            "requested_by": row["requested_by"],
            "schedule_id": row["schedule_id"],
            "kind": row["kind"],
            "status": row["status"],
            "progress": row["progress"],
            "message": row["message"],
            "error": row["error"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
        }
        payload = json.loads(row["payload_json"] or "{}")
        if not include_secrets:
            payload = {
                key: value for key, value in payload.items()
                if not str(key).startswith("_secret_")
            }
        job["payload"] = payload
        return job

    def create_job(
        self,
        document_id: str | None = None,
        kind: str = "pdf_ingestion",
        *,
        requested_by: str | None = None,
        schedule_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO maintenance_jobs(
                    id, document_id, requested_by, schedule_id, kind, payload_json,
                    status, progress, message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'queued', 0, ?, ?)
                """,
                (
                    job_id, document_id, requested_by, schedule_id, kind,
                    json.dumps(payload or {}, ensure_ascii=False), "En attente", timestamp,
                ),
            )
        return self.get_job(job_id) or {}

    def get_job(
        self, job_id: str, *, include_secrets: bool = False
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM maintenance_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._job_from_row(row, include_secrets=include_secrets) if row else None

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM maintenance_jobs
                ORDER BY created_at DESC LIMIT ?
                """,
                (max(1, min(int(limit), 100)),),
            ).fetchall()
        return [self._job_from_row(row) for row in rows]

    def claim_next_job(self) -> dict[str, Any] | None:
        """Réserve atomiquement le prochain travail pour un worker."""
        with self._connect() as connection:
            suffix = " FOR UPDATE SKIP LOCKED" if self.is_postgres else ""
            row = connection.execute(
                "SELECT * FROM maintenance_jobs WHERE status = 'queued' "
                f"ORDER BY created_at LIMIT 1{suffix}"
            ).fetchone()
            if not row:
                return None
            connection.execute(
                "UPDATE maintenance_jobs SET status = 'running', started_at = ?, "
                "progress = 1, message = ? WHERE id = ? AND status = 'queued'",
                (utc_now(), "Pris en charge par le worker", row["id"]),
            )
        return self.get_job(row["id"], include_secrets=True)

    def update_job(
        self,
        job_id: str,
        *,
        status: str,
        progress: int,
        message: str,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        if status not in {"queued", "running", "completed", "error"}:
            raise ValueError("Statut de tâche invalide.")
        existing = self.get_job(job_id)
        if not existing:
            return None
        timestamp = utc_now()
        started_at = existing["started_at"] or (timestamp if status == "running" else None)
        finished_at = timestamp if status in {"completed", "error"} else None
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE maintenance_jobs SET
                    status = ?, progress = ?, message = ?, error = ?,
                    started_at = ?, finished_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    max(0, min(int(progress), 100)),
                    message[:500],
                    error,
                    started_at,
                    finished_at,
                    job_id,
                ),
            )
        return self.get_job(job_id)

    # ------------------------------------------------------------------
    # Planification et paramètres administratifs
    # ------------------------------------------------------------------

    @staticmethod
    def _schedule_from_row(row: sqlite3.Row) -> dict[str, Any]:
        schedule = dict(row)
        schedule["enabled"] = bool(schedule["enabled"])
        schedule["payload"] = json.loads(schedule.pop("payload_json") or "{}")
        return schedule

    def create_schedule(
        self,
        *,
        name: str,
        task_kind: str,
        interval_minutes: int,
        next_run_at: str,
        created_by: str,
        enabled: bool = True,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if int(interval_minutes) < 5:
            raise ValueError("La fréquence minimale est de cinq minutes.")
        schedule_id = uuid.uuid4().hex
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO schedules(
                    id, name, task_kind, interval_minutes, enabled, payload_json,
                    created_by, next_run_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    schedule_id, name.strip()[:160], task_kind,
                    int(interval_minutes), int(enabled),
                    json.dumps(payload or {}, ensure_ascii=False), created_by,
                    next_run_at, timestamp, timestamp,
                ),
            )
        return self.get_schedule(schedule_id) or {}

    def get_schedule(self, schedule_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM schedules WHERE id = ?", (schedule_id,)
            ).fetchone()
        return self._schedule_from_row(row) if row else None

    def list_schedules(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM schedules ORDER BY name COLLATE NOCASE"
            ).fetchall()
        return [self._schedule_from_row(row) for row in rows]

    def list_due_schedules(self, now: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM schedules
                WHERE enabled = 1 AND next_run_at <= ? ORDER BY next_run_at
                """,
                (now,),
            ).fetchall()
        return [self._schedule_from_row(row) for row in rows]

    def update_schedule(
        self,
        schedule_id: str,
        *,
        name: str | None = None,
        interval_minutes: int | None = None,
        enabled: bool | None = None,
        next_run_at: str | None = None,
        last_run_at: str | None = None,
    ) -> dict[str, Any] | None:
        schedule = self.get_schedule(schedule_id)
        if not schedule:
            return None
        interval = schedule["interval_minutes"] if interval_minutes is None else int(interval_minutes)
        if interval < 5:
            raise ValueError("La fréquence minimale est de cinq minutes.")
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE schedules SET name = ?, interval_minutes = ?, enabled = ?,
                    next_run_at = ?, last_run_at = ?, updated_at = ? WHERE id = ?
                """,
                (
                    (name or schedule["name"]).strip()[:160], interval,
                    int(schedule["enabled"] if enabled is None else enabled),
                    next_run_at or schedule["next_run_at"],
                    last_run_at if last_run_at is not None else schedule["last_run_at"],
                    utc_now(), schedule_id,
                ),
            )
        return self.get_schedule(schedule_id)

    def delete_schedule(self, schedule_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM schedules WHERE id = ?", (schedule_id,)
            )
        return cursor.rowcount > 0

    def get_settings(self) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT key, value_json FROM system_settings"
            ).fetchall()
        return {row["key"]: json.loads(row["value_json"]) for row in rows}

    def set_settings(
        self, values: dict[str, Any], *, updated_by: str | None
    ) -> dict[str, Any]:
        timestamp = utc_now()
        with self._connect() as connection:
            for key, value in values.items():
                connection.execute(
                    """
                    INSERT INTO system_settings(key, value_json, updated_by, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json,
                        updated_by = excluded.updated_by, updated_at = excluded.updated_at
                    """,
                    (key, json.dumps(value, ensure_ascii=False), updated_by, timestamp),
                )
        return self.get_settings()
