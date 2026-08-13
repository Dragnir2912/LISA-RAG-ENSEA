"""Authentification locale robuste par sessions serveur et contrôle de rôles."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from backend.configuration import settings
from backend.database import AppDatabase


SESSION_COOKIE = "rag_ensea_session"
PASSWORD_ITERATIONS = 600_000
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def _utc_datetime() -> datetime:
    return datetime.now(timezone.utc)


def _decode_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def validate_email(email: str) -> str:
    normalized = email.strip().casefold()
    if len(normalized) > 254 or not EMAIL_PATTERN.match(normalized):
        raise ValueError("Adresse e-mail invalide.")
    return normalized


def validate_password(password: str) -> None:
    if len(password) < 10:
        raise ValueError("Le mot de passe doit contenir au moins 10 caractères.")
    if len(password) > 256:
        raise ValueError("Le mot de passe est trop long.")
    if not re.search(r"[A-Za-zÀ-ÿ]", password) or not re.search(r"\d", password):
        raise ValueError("Le mot de passe doit contenir au moins une lettre et un chiffre.")


def hash_password(password: str) -> str:
    validate_password(password)
    try:
        from argon2 import PasswordHasher

        return PasswordHasher(
            time_cost=2,
            memory_cost=19_456,
            parallelism=1,
            hash_len=32,
            salt_len=16,
        ).hash(password)
    except ImportError:
        # Compatibilité du poste local tant que les dépendances serveur ne sont
        # pas installées. Les conteneurs utilisent toujours Argon2id.
        pass
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )
    return "pbkdf2_sha256${}${}${}".format(
        PASSWORD_ITERATIONS,
        base64.urlsafe_b64encode(salt).decode("ascii").rstrip("="),
        base64.urlsafe_b64encode(digest).decode("ascii").rstrip("="),
    )


def verify_password(password: str, encoded: str) -> bool:
    if encoded.startswith("$argon2"):
        try:
            from argon2 import PasswordHasher
            from argon2.exceptions import InvalidHashError, VerifyMismatchError

            return PasswordHasher().verify(encoded, password)
        except (ImportError, InvalidHashError, VerifyMismatchError):
            return False
    try:
        algorithm, iterations_text, salt_text, expected_text = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.urlsafe_b64decode(salt_text + "=" * (-len(salt_text) % 4))
        expected = base64.urlsafe_b64decode(
            expected_text + "=" * (-len(expected_text) % 4)
        )
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, int(iterations_text)
        )
        return hmac.compare_digest(actual, expected)
    except (TypeError, ValueError):
        return False


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AuthenticatedSession:
    user: dict[str, Any]
    session_id: str
    csrf_token: str


class AuthService:
    """Gère le bootstrap, les connexions et les sessions révocables."""

    def __init__(self, database: AppDatabase) -> None:
        self.database = database

    @property
    def setup_required(self) -> bool:
        return self.database.active_admin_count() == 0

    def setup_admin(
        self, *, email: str, display_name: str, password: str
    ) -> dict[str, Any]:
        if not self.setup_required:
            raise PermissionError("L'administrateur initial existe déjà.")
        user = self.database.create_user(
            email=validate_email(email),
            display_name=display_name.strip() or "Administrateur",
            password_hash=hash_password(password),
            role="admin",
            adopt_legacy_records=True,
            email_verified=True,
        )
        self.database.add_audit_event(
            "auth.initial_admin_created",
            actor_user_id=user["id"],
            target_type="user",
            target_id=user["id"],
        )
        return user

    def create_user(
        self,
        *,
        actor_user_id: str,
        email: str,
        display_name: str,
        password: str,
        role: str,
    ) -> dict[str, Any]:
        user = self.database.create_user(
            email=validate_email(email),
            display_name=display_name.strip(),
            password_hash=hash_password(password),
            role=role,
            email_verified=True,
        )
        self.database.add_audit_event(
            "user.created",
            actor_user_id=actor_user_id,
            target_type="user",
            target_id=user["id"],
            details={"email": user["email"], "role": user["role"]},
        )
        return user

    def register(
        self,
        *,
        email: str,
        display_name: str,
        password: str,
        ip_address: str = "",
    ) -> dict[str, Any]:
        user = self.database.create_user(
            email=validate_email(email),
            display_name=display_name.strip() or "Utilisateur",
            password_hash=hash_password(password),
            role="user",
            account_type="registered",
            registration_ip=ip_address,
        )
        self.database.add_audit_event(
            "auth.registration",
            actor_user_id=user["id"],
            target_type="user",
            target_id=user["id"],
            details={"email": user["email"]},
        )
        return user

    def _issue_session(
        self,
        user: dict[str, Any],
        *,
        user_agent: str,
        ip_address: str,
        lifetime: timedelta,
    ) -> tuple[str, AuthenticatedSession]:
        raw_token = secrets.token_urlsafe(48)
        csrf_token = secrets.token_urlsafe(32)
        session = self.database.create_session(
            user_id=user["id"],
            token_hash=token_hash(raw_token),
            csrf_token=csrf_token,
            expires_at=(_utc_datetime() + lifetime).isoformat(),
            user_agent=user_agent,
            ip_address=ip_address,
        )
        return raw_token, AuthenticatedSession(
            user=self.database.get_user(user["id"]) or user,
            session_id=session["id"],
            csrf_token=csrf_token,
        )

    def create_guest_session(
        self, *, user_agent: str = "", ip_address: str = ""
    ) -> tuple[str, AuthenticatedSession]:
        guest_id = secrets.token_hex(8)
        user = self.database.create_user(
            email=f"guest-{guest_id}@guest.local",
            display_name="Invité",
            password_hash=hash_password(secrets.token_urlsafe(32) + "9a"),
            role="user",
            account_type="guest",
            registration_ip=ip_address,
        )
        token, session = self._issue_session(
            user,
            user_agent=user_agent,
            ip_address=ip_address,
            lifetime=timedelta(hours=settings.guest_session_hours),
        )
        self.database.add_audit_event(
            "auth.guest_started",
            actor_user_id=user["id"],
            target_type="session",
            target_id=session.session_id,
        )
        return token, session

    def request_password_reset(
        self, *, email: str, ip_address: str = ""
    ) -> str | None:
        normalized = validate_email(email)
        user = self.database.get_user_by_email(normalized)
        if not user or user.get("account_type") == "guest" or not user["is_active"]:
            self.database.add_audit_event(
                "auth.password_reset_requested",
                details={"account_found": False},
            )
            return None
        raw_token = secrets.token_urlsafe(48)
        self.database.create_password_reset_token(
            user_id=user["id"],
            token_hash=token_hash(raw_token),
            expires_at=(_utc_datetime() + timedelta(minutes=30)).isoformat(),
            requested_ip=ip_address,
        )
        self.database.add_audit_event(
            "auth.password_reset_requested",
            actor_user_id=user["id"],
            target_type="user",
            target_id=user["id"],
            details={"account_found": True},
        )
        return raw_token

    def reset_password(self, *, raw_token: str, password: str) -> dict[str, Any]:
        token = self.database.consume_password_reset_token(token_hash(raw_token))
        if not token:
            raise PermissionError("Ce lien de récupération est invalide ou expiré.")
        user = self.database.update_user(
            token["user_id"], password_hash=hash_password(password)
        )
        if not user:
            raise PermissionError("Compte introuvable.")
        self.database.add_audit_event(
            "auth.password_reset_completed",
            actor_user_id=user["id"],
            target_type="user",
            target_id=user["id"],
        )
        return user

    def login(
        self,
        *,
        email: str,
        password: str,
        user_agent: str = "",
        ip_address: str = "",
    ) -> tuple[str, AuthenticatedSession]:
        user = self.database.get_user_by_email(validate_email(email), include_secret=True)
        if not user:
            # Calcul volontaire pour rendre le temps comparable à celui d'un compte réel.
            hashlib.pbkdf2_hmac("sha256", password.encode(), b"unknown-user-salt", 100_000)
            raise PermissionError("Identifiants incorrects.")
        locked_until = _decode_timestamp(user.get("locked_until"))
        if locked_until and locked_until > _utc_datetime():
            raise PermissionError("Compte temporairement verrouillé. Réessayez plus tard.")
        if not user["is_active"]:
            raise PermissionError("Ce compte est désactivé.")
        if user.get("account_type") == "guest":
            raise PermissionError("Identifiants incorrects.")
        encoded_password = user.pop("password_hash")
        if not verify_password(password, encoded_password):
            failures = int(user.get("failed_login_count", 0)) + 1
            lock = (
                (_utc_datetime() + timedelta(minutes=15)).isoformat()
                if failures >= 5 else None
            )
            self.database.record_login_failure(user["id"], lock)
            raise PermissionError("Identifiants incorrects.")

        self.database.record_login_success(user["id"])
        if encoded_password.startswith("pbkdf2_"):
            self.database.update_user(user["id"], password_hash=hash_password(password))
        raw_token, authenticated = self._issue_session(
            user,
            user_agent=user_agent,
            ip_address=ip_address,
            lifetime=timedelta(days=settings.session_days),
        )
        self.database.add_audit_event(
            "auth.login", actor_user_id=user["id"], target_type="session",
            target_id=authenticated.session_id,
        )
        return raw_token, authenticated

    def authenticate(self, raw_token: str | None) -> AuthenticatedSession | None:
        if not raw_token:
            return None
        session = self.database.get_session_by_hash(token_hash(raw_token))
        if not session or session["revoked_at"] or not bool(session["is_active"]):
            return None
        expires_at = _decode_timestamp(session["expires_at"])
        if not expires_at or expires_at <= _utc_datetime():
            self.database.revoke_session(session["id"])
            return None
        last_seen = _decode_timestamp(session.get("last_seen_at"))
        if not last_seen or (_utc_datetime() - last_seen).total_seconds() >= 60:
            self.database.touch_session(session["id"])
        user = {
            "id": session["user_id"],
            "email": session["email"],
            "display_name": session["display_name"],
            "role": session["role"],
            "is_active": bool(session["is_active"]),
            "account_type": session.get("account_type", "registered"),
        }
        return AuthenticatedSession(
            user=user, session_id=session["id"], csrf_token=session["csrf_token"]
        )

    def logout(self, session: AuthenticatedSession) -> None:
        self.database.revoke_session(session.session_id, user_id=session.user["id"])
        self.database.add_audit_event(
            "auth.logout", actor_user_id=session.user["id"],
            target_type="session", target_id=session.session_id,
        )
