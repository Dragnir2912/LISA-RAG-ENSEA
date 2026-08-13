"""Télécharge ou valide les modèles Hugging Face requis par LISA.

Ce script est exécuté pendant la préparation connectée du paquet hors ligne.
En exploitation, ``HF_HUB_OFFLINE=1`` interdit ensuite tout accès réseau.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Valide le cache local sans aucun téléchargement.",
    )
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "config.yaml")
    model_ids = (config.embeddings.model_name, config.reranker.model_name)
    cache_root = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)

    for model_id in model_ids:
        path = snapshot_download(
            repo_id=model_id,
            local_files_only=args.check_only,
        )
        print(f"OK {model_id} -> {path}", flush=True)

    print(f"Cache Hugging Face prêt : {cache_root}", flush=True)


if __name__ == "__main__":
    main()
