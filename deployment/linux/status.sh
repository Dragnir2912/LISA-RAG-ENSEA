#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ENV_FILE=${LISA_ENV_FILE:-.env.test}
MODE=${1:-once}
case "$MODE" in
  once) FOLLOW=() ;;
  --follow) FOLLOW=(-f) ;;
  *) echo 'Option invalide. Utilisez --follow pour suivre les journaux.' >&2; exit 2 ;;
esac
cd "$ROOT"

docker compose --env-file "$ENV_FILE" -f compose.test.yaml ps
docker compose --env-file "$ENV_FILE" -f compose.test.yaml \
  logs --tail=100 "${FOLLOW[@]}" api worker ollama postgres
