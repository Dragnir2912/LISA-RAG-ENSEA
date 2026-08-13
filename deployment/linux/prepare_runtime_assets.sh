#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ENV_FILE=${1:-.env.test}
LISA_IMAGE_TAG=${LISA_IMAGE_TAG:-test}
OLLAMA_MODEL=${OLLAMA_MODEL:-qwen3.5:9b}
POSTGRES_IMAGE=${POSTGRES_IMAGE:-postgres:17-alpine}
OLLAMA_IMAGE=${OLLAMA_IMAGE:-ollama/ollama:latest}
CADDY_IMAGE=${CADDY_IMAGE:-caddy:2.10-alpine}

cd "$ROOT"
test -f "$ENV_FILE" || { echo "Fichier absent : $ENV_FILE" >&2; exit 1; }
mkdir -p models/huggingface models/ollama

echo 'Construction des images LISA...'
docker build -f Dockerfile.backend -t "lisa-api:$LISA_IMAGE_TAG" .
docker build -f Dockerfile.frontend -t "lisa-frontend:$LISA_IMAGE_TAG" .
docker pull "$POSTGRES_IMAGE"
docker pull "$OLLAMA_IMAGE"
docker pull "$CADDY_IMAGE"

echo 'Préparation des modèles BGE-M3 et BGE Reranker...'
docker run --rm \
  -e HF_HOME=/srv/lisa/models/huggingface \
  -e HF_HUB_OFFLINE=0 \
  -e TRANSFORMERS_OFFLINE=0 \
  -v "$ROOT/models/huggingface:/srv/lisa/models/huggingface" \
  "lisa-api:$LISA_IMAGE_TAG" \
  python /srv/lisa/deployment/prefetch_models.py

echo "Préparation du modèle Ollama $OLLAMA_MODEL..."
PREPARER_CONTAINER="lisa-ollama-model-preparer-$$"
cleanup() {
  docker rm -f "$PREPARER_CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT
docker run -d --name "$PREPARER_CONTAINER" \
  -e OLLAMA_NO_CLOUD=1 \
  -v "$ROOT/models/ollama:/root/.ollama" \
  "$OLLAMA_IMAGE" >/dev/null
for _ in $(seq 1 30); do
  docker exec "$PREPARER_CONTAINER" ollama list >/dev/null 2>&1 && break
  sleep 2
done
docker exec "$PREPARER_CONTAINER" ollama pull "$OLLAMA_MODEL"
docker exec "$PREPARER_CONTAINER" ollama show "$OLLAMA_MODEL" >/dev/null
cleanup
trap - EXIT

echo 'Images et modèles nécessaires à l’exploitation hors ligne sont prêts.'
