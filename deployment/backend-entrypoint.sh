#!/bin/sh
set -eu

if [ -n "${RAG_DATABASE_PASSWORD_FILE:-}" ] && [ -f "$RAG_DATABASE_PASSWORD_FILE" ]; then
  database_password="$(cat "$RAG_DATABASE_PASSWORD_FILE")"
  export RAG_DATABASE_URL="postgresql://${POSTGRES_USER}:${database_password}@${POSTGRES_HOST:-postgres}:5432/${POSTGRES_DB}"
fi

if [ -n "${RAG_SECRET_KEY_FILE:-}" ] && [ -f "$RAG_SECRET_KEY_FILE" ]; then
  export RAG_SECRET_KEY="$(cat "$RAG_SECRET_KEY_FILE")"
fi

exec "$@"
