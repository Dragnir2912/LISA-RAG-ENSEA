#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ENV_FILE=${1:-.env.test}
LISA_IMAGE_TAG=${LISA_IMAGE_TAG:-test}
OLLAMA_MODEL=${OLLAMA_MODEL:-qwen3.5:9b}

cd "$ROOT"
test -f "$ENV_FILE"
test -s secrets/postgres_password.txt
test -s secrets/application_key.txt
test -d data/embeddings
test -d models/huggingface/hub
test -d models/ollama

docker image inspect "lisa-api:$LISA_IMAGE_TAG" >/dev/null
docker image inspect "lisa-frontend:$LISA_IMAGE_TAG" >/dev/null

docker run --rm \
  -e HF_HOME=/srv/lisa/models/huggingface \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v "$ROOT/models/huggingface:/srv/lisa/models/huggingface:ro" \
  "lisa-api:$LISA_IMAGE_TAG" \
  python /srv/lisa/deployment/prefetch_models.py --check-only

docker compose --env-file "$ENV_FILE" -f compose.test.yaml config >/dev/null
docker compose --env-file "$ENV_FILE" -f compose.test.yaml up -d ollama
docker compose --env-file "$ENV_FILE" -f compose.test.yaml exec -T ollama \
  ollama show "$OLLAMA_MODEL" >/dev/null
docker compose --env-file "$ENV_FILE" -f compose.test.yaml stop ollama

echo 'Validation hors ligne réussie : images, modèles et index sont présents.'
