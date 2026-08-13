#!/usr/bin/env bash
set -euo pipefail

failures=0

check_command() {
  if command -v "$1" >/dev/null 2>&1; then
    printf 'OK   %-18s %s\n' "$1" "$(command -v "$1")"
  else
    printf 'NON  %-18s commande absente\n' "$1"
    failures=$((failures + 1))
  fi
}

printf '%s\n' 'Vérification des prérequis LISA'
check_command docker
check_command openssl
check_command tar
check_command sha256sum
check_command curl

if command -v docker >/dev/null 2>&1; then
  docker version >/dev/null 2>&1 || {
    printf '%s\n' 'NON  le démon Docker ne répond pas pour cet utilisateur'
    failures=$((failures + 1))
  }
  docker compose version >/dev/null 2>&1 || {
    printf '%s\n' 'NON  plugin Docker Compose absent'
    failures=$((failures + 1))
  }
fi

available_kb=$(df -Pk . | awk 'NR==2 {print $4}')
if [ "${available_kb:-0}" -lt 104857600 ]; then
  printf '%s\n' 'ATTENTION moins de 100 Gio libres sur le volume courant'
else
  printf 'OK   espace disque       %s Gio libres\n' "$((available_kb / 1024 / 1024))"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  printf '%s\n' 'OK   GPU NVIDIA détecté (facultatif)'
else
  printf '%s\n' 'INFO aucun GPU NVIDIA : le profil CPU sera utilisé'
fi

if [ "$failures" -ne 0 ]; then
  printf '\n%d prérequis obligatoire(s) manquant(s).\n' "$failures"
  exit 1
fi

printf '%s\n' 'Tous les prérequis obligatoires sont présents.'
