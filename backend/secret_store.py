"""Transport chiffré des secrets éphémères entre l'API et le worker."""

from __future__ import annotations

import os


class SecretStore:
    """Scelle un secret avant son passage temporaire par PostgreSQL."""

    def __init__(self, key: str | None = None) -> None:
        self.key = (key or os.getenv("RAG_SECRET_KEY") or "").strip()

    @property
    def available(self) -> bool:
        return bool(self.key)

    def seal(self, value: str) -> str:
        if not self.key:
            raise RuntimeError(
                "RAG_SECRET_KEY est requis pour transmettre un secret au worker."
            )
        try:
            from cryptography.fernet import Fernet

            return Fernet(self.key.encode("ascii")).encrypt(
                value.encode("utf-8")
            ).decode("ascii")
        except ImportError as exc:
            raise RuntimeError("Le paquet cryptography est requis sur le serveur.") from exc
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError("RAG_SECRET_KEY n'est pas une clé Fernet valide.") from exc

    def open(self, value: str) -> str:
        if not value:
            return ""
        if not self.key:
            raise RuntimeError("RAG_SECRET_KEY est absent du worker.")
        try:
            from cryptography.fernet import Fernet, InvalidToken

            return Fernet(self.key.encode("ascii")).decrypt(
                value.encode("ascii")
            ).decode("utf-8")
        except ImportError as exc:
            raise RuntimeError("Le paquet cryptography est requis sur le worker.") from exc
        except (InvalidToken, ValueError, UnicodeError) as exc:
            raise RuntimeError("Le secret de la tâche ne peut pas être déchiffré.") from exc


secret_store = SecretStore()
