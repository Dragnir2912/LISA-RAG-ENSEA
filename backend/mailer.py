"""Envoi des messages transactionnels sans dépendre d'un fournisseur précis."""

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage


def send_password_reset(*, recipient: str, display_name: str, reset_url: str) -> bool:
    host = (os.getenv("RAG_SMTP_HOST") or "").strip()
    if not host:
        return False
    message = EmailMessage()
    message["Subject"] = "Réinitialisation de votre mot de passe LISA"
    message["From"] = os.getenv("RAG_SMTP_FROM", "LISA <noreply@ensea.local>")
    message["To"] = recipient
    message.set_content(
        f"Bonjour {display_name},\n\n"
        "Une réinitialisation de mot de passe a été demandée pour votre compte LISA.\n"
        f"Utilisez ce lien pendant les 30 prochaines minutes :\n{reset_url}\n\n"
        "Si vous n'êtes pas à l'origine de cette demande, ignorez ce message."
    )
    port = int(os.getenv("RAG_SMTP_PORT", "587"))
    username = os.getenv("RAG_SMTP_USERNAME", "")
    password = os.getenv("RAG_SMTP_PASSWORD", "")
    with smtplib.SMTP(host, port, timeout=15) as client:
        client.starttls()
        if username:
            client.login(username, password)
        client.send_message(message)
    return True
