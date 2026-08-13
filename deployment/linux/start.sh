#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROFILE=${1:-cpu}
TLS_MODE=${2:-offline}
ENV_FILE=${LISA_ENV_FILE:-.env.test}

case "$PROFILE" in
  cpu) EXTRA_PROFILE=() ;;
  gpu) EXTRA_PROFILE=(-f compose.gpu.yaml) ;;
  *) echo 'Profil invalide. Utilisez cpu ou gpu.' >&2; exit 2 ;;
esac
case "$TLS_MODE" in
  offline) TLS_FILE=compose.tls.offline.yaml ;;
  public) TLS_FILE=compose.tls.yaml ;;
  none) TLS_FILE= ;;
  *) echo 'Mode TLS invalide. Utilisez offline, public ou none.' >&2; exit 2 ;;
esac

cd "$ROOT"
test -f "$ENV_FILE" || { echo "Fichier absent : $ENV_FILE" >&2; exit 1; }
test -s secrets/postgres_password.txt || { echo 'Secret PostgreSQL absent.' >&2; exit 1; }
test -s secrets/application_key.txt || { echo 'Clé applicative absente.' >&2; exit 1; }
test -d models/huggingface/hub || { echo 'Cache Hugging Face absent.' >&2; exit 1; }
test -d models/ollama || { echo 'Cache Ollama absent.' >&2; exit 1; }
test -d data/embeddings || { echo 'Index institutionnel absent.' >&2; exit 1; }

COMPOSE=(docker compose --env-file "$ENV_FILE" -f compose.test.yaml)
COMPOSE+=("${EXTRA_PROFILE[@]}")
if [ -n "$TLS_FILE" ]; then COMPOSE+=(-f "$TLS_FILE"); fi

"${COMPOSE[@]}" config >/dev/null
"${COMPOSE[@]}" up -d --no-build
docker compose --env-file "$ENV_FILE" -f compose.test.yaml ps
