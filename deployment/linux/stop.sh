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
COMPOSE=(docker compose --env-file "$ENV_FILE" -f compose.test.yaml)
COMPOSE+=("${EXTRA_PROFILE[@]}")
if [ -n "$TLS_FILE" ]; then COMPOSE+=(-f "$TLS_FILE"); fi
"${COMPOSE[@]}" down
echo 'LISA est arrêtée. Les volumes, données et modèles sont conservés.'
