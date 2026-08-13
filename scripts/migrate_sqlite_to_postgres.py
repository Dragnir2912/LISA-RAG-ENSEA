"""Migre les données durables de l'application locale vers PostgreSQL.

L'URL PostgreSQL est lue dans RAG_DATABASE_URL afin de ne jamais placer le mot
de passe dans l'historique de commande. Arrêter API et worker pendant l'opération.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import AppDatabase


DURABLE_TABLES = (
    "users", "conversations", "messages", "citations", "feedback",
    "documents", "schedules", "system_settings", "audit_events",
)


def migrate(sqlite_path: Path, database_url: str) -> dict[str, int]:
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"Base SQLite introuvable : {sqlite_path}")
    if not database_url.startswith(("postgresql://", "postgres://")):
        raise ValueError("RAG_DATABASE_URL doit désigner PostgreSQL.")

    AppDatabase(database_url=database_url)
    import psycopg

    report: dict[str, int] = {}
    source = sqlite3.connect(sqlite_path)
    source.row_factory = sqlite3.Row
    try:
        with psycopg.connect(database_url) as target:
            with target.cursor() as cursor:
                for table in DURABLE_TABLES:
                    exists = source.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (table,),
                    ).fetchone()
                    if not exists:
                        report[table] = 0
                        continue
                    columns = [
                        row["name"]
                        for row in source.execute(f'PRAGMA table_info("{table}")')
                    ]
                    rows = source.execute(f'SELECT * FROM "{table}"').fetchall()
                    if not rows:
                        report[table] = 0
                        continue
                    quoted = ", ".join(f'"{column}"' for column in columns)
                    placeholders = ", ".join(["%s"] * len(columns))
                    statement = (
                        f'INSERT INTO "{table}" ({quoted}) VALUES ({placeholders}) '
                        "ON CONFLICT DO NOTHING"
                    )
                    for row in rows:
                        cursor.execute(statement, tuple(row[column] for column in columns))
                    report[table] = len(rows)
            target.commit()
    finally:
        source.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sqlite", type=Path,
        default=PROJECT_ROOT / "data" / "app" / "rag_ensea.sqlite3",
    )
    args = parser.parse_args()
    database_url = (os.getenv("RAG_DATABASE_URL") or "").strip()
    if not database_url:
        raise SystemExit("Définissez RAG_DATABASE_URL avant la migration.")
    for table, count in migrate(args.sqlite.resolve(), database_url).items():
        print(f"{table}: {count} ligne(s) examinée(s)")


if __name__ == "__main__":
    main()
