"""Lance l'application RAG ENSEA sur la machine locale."""

from __future__ import annotations

import logging
import os

import uvicorn


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    uvicorn.run(
        "backend.main:app",
        host=os.getenv("RAG_HOST", "127.0.0.1"),
        port=int(os.getenv("RAG_PORT", "8000")),
        reload=False,
        workers=1,  # un seul processus possède la file GPU et le planificateur
        log_level="info",
    )
