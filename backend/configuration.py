"""Configuration de déploiement, chargée exclusivement depuis l'environnement.

Les valeurs sensibles ne sont jamais stockées dans ``config.yaml`` : ce dernier
reste réservé au pipeline RAG. Cette séparation évite qu'un secret de serveur
soit exporté avec un notebook ou une sauvegarde documentaire.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


def _items(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(item.strip() for item in os.getenv(name, default).split(",") if item.strip())


@dataclass(frozen=True)
class ApplicationSettings:
    environment: str
    database_url: str | None
    frontend_dir: Path
    public_url: str
    trusted_hosts: tuple[str, ...]
    allowed_origins: tuple[str, ...]
    cookie_secure: bool
    cookie_domain: str | None
    session_days: int
    guest_session_hours: int
    guest_data_days: int
    self_registration_enabled: bool
    guest_access_enabled: bool
    password_reset_enabled: bool
    password_reset_preview: bool
    scheduler_enabled: bool
    jobs_inline: bool
    max_request_bytes: int
    offline_mode: bool

    @property
    def production(self) -> bool:
        return self.environment in {"staging", "production"}

    @classmethod
    def from_env(cls) -> "ApplicationSettings":
        environment = os.getenv("RAG_ENV", "development").strip().casefold()
        public_url = os.getenv("RAG_PUBLIC_URL", "http://127.0.0.1:8000").rstrip("/")
        return cls(
            environment=environment,
            database_url=(os.getenv("RAG_DATABASE_URL") or "").strip() or None,
            frontend_dir=Path(
                os.getenv("RAG_FRONTEND_DIR", str(PROJECT_ROOT / "frontend"))
            ).resolve(),
            public_url=public_url,
            trusted_hosts=_items("RAG_TRUSTED_HOSTS", "127.0.0.1,localhost,testserver"),
            allowed_origins=_items("RAG_ALLOWED_ORIGINS", public_url),
            cookie_secure=_flag("RAG_COOKIE_SECURE", environment != "development"),
            cookie_domain=(os.getenv("RAG_COOKIE_DOMAIN") or "").strip() or None,
            session_days=max(1, min(int(os.getenv("RAG_SESSION_DAYS", "7")), 30)),
            guest_session_hours=max(
                1, min(int(os.getenv("RAG_GUEST_SESSION_HOURS", "8")), 72)
            ),
            guest_data_days=max(1, min(int(os.getenv("RAG_GUEST_DATA_DAYS", "7")), 30)),
            self_registration_enabled=_flag("RAG_SELF_REGISTRATION", True),
            guest_access_enabled=_flag("RAG_GUEST_ACCESS", True),
            password_reset_enabled=_flag("RAG_PASSWORD_RESET", True),
            password_reset_preview=_flag(
                "RAG_PASSWORD_RESET_PREVIEW", environment == "development"
            ),
            scheduler_enabled=_flag("RAG_RUN_SCHEDULER", environment == "development"),
            jobs_inline=_flag("RAG_JOBS_INLINE", environment == "development"),
            max_request_bytes=max(
                1_048_576,
                min(int(os.getenv("RAG_MAX_REQUEST_BYTES", str(55 * 1024 * 1024))), 100 * 1024 * 1024),
            ),
            offline_mode=_flag("RAG_OFFLINE_MODE", False),
        )


settings = ApplicationSettings.from_env()
