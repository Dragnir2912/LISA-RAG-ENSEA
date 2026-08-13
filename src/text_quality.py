"""Filtres légers de qualité textuelle partagés par le nettoyage et le chunking."""

from __future__ import annotations

import re


# Fragments caractéristiques des textes de démonstration laissés par les thèmes
# WordPress. On exige des expressions spécifiques afin de ne pas filtrer du latin
# légitime ou des contenus courts simplement inhabituels.
_PLACEHOLDER_PATTERNS = (
    re.compile(r"\blorem ipsum\b", re.IGNORECASE),
    re.compile(r"\bconsectetur adipiscing elit\b", re.IGNORECASE),
    re.compile(r"\bnullam faucibus\b", re.IGNORECASE),
    re.compile(r"\bsuspendisse sapien mauris\b", re.IGNORECASE),
    re.compile(r"\bphasellus imperdiet\b", re.IGNORECASE),
)

_WORD_RE = re.compile(r"[\wÀ-ÿ]+", re.UNICODE)


def is_placeholder_line(line: str) -> bool:
    """Indique si une ligne appartient clairement à un gabarit factice."""
    return any(pattern.search(line) for pattern in _PLACEHOLDER_PATTERNS)


def strip_placeholder_lines(text: str) -> str:
    """Retire les lignes factices tout en conservant le contenu éditorial voisin."""
    return "\n".join(
        line for line in text.splitlines() if not is_placeholder_line(line)
    ).strip()


def has_substantive_body(text: str) -> bool:
    """Rejette un fragment composé uniquement de titres Markdown.

    Une ligne de corps très courte reste acceptée : un nom, une date ou une
    adresse peuvent constituer une excellente preuve dans un système RAG.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or re.match(r"^#{1,6}(?:\s+|$)", stripped):
            continue
        if _WORD_RE.search(stripped):
            return True
    return False
