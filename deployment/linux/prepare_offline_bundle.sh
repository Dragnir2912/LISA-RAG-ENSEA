#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ENV_FILE=${1:-.env.test}
OUTPUT=${2:-dist/lisa-offline}
LISA_IMAGE_TAG=${LISA_IMAGE_TAG:-test}
OLLAMA_MODEL=${OLLAMA_MODEL:-qwen3.5:9b}
POSTGRES_IMAGE=${POSTGRES_IMAGE:-postgres:17-alpine}
OLLAMA_IMAGE=${OLLAMA_IMAGE:-ollama/ollama:latest}
CADDY_IMAGE=${CADDY_IMAGE:-caddy:2.10-alpine}

cd "$ROOT"
test -f "$ENV_FILE" || { echo "Fichier absent : $ENV_FILE" >&2; exit 1; }
test -d data/embeddings || { echo 'Embeddings absents dans data/embeddings' >&2; exit 1; }
mkdir -p models/huggingface models/ollama "$OUTPUT"

echo '1/4 Préparation connectée des images et modèles...'
LISA_IMAGE_TAG="$LISA_IMAGE_TAG" OLLAMA_MODEL="$OLLAMA_MODEL" \
  POSTGRES_IMAGE="$POSTGRES_IMAGE" OLLAMA_IMAGE="$OLLAMA_IMAGE" \
  CADDY_IMAGE="$CADDY_IMAGE" \
  deployment/linux/prepare_runtime_assets.sh "$ENV_FILE"

echo '2/4 Export des images Docker...'
docker save \
  -o "$OUTPUT/docker-images.tar" \
  "lisa-api:$LISA_IMAGE_TAG" \
  "lisa-frontend:$LISA_IMAGE_TAG" \
  "$POSTGRES_IMAGE" "$OLLAMA_IMAGE" "$CADDY_IMAGE"

echo '3/4 Archivage de l’application, des index et des modèles...'
tar \
  --exclude='./.venv' --exclude='./notebooks' --exclude='./reports' \
  --exclude='./dist' --exclude='./data' --exclude='./models' \
  --exclude='./secrets' --exclude='./.env.test' --exclude='./.env.local' \
  -czf "$OUTPUT/application.tar.gz" .
tar -czf "$OUTPUT/data.tar.gz" data
tar -czf "$OUTPUT/models.tar.gz" models

echo '4/4 Empreintes de contrôle...'
(
  cd "$OUTPUT"
  sha256sum docker-images.tar application.tar.gz data.tar.gz models.tar.gz > SHA256SUMS
)

echo "Paquet hors ligne prêt : $OUTPUT"
echo 'Copiez tout ce dossier sur le serveur Linux déconnecté.'
