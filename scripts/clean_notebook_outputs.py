"""Retire uniquement les sorties de progression très répétitives des notebooks.

Usage ponctuel avant une publication Git :
    python scripts/clean_notebook_outputs.py chemin/vers/notebook.ipynb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _text(output: dict) -> str:
    value = output.get("text", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _redact_local_paths(value: object, roots: tuple[str, ...]) -> object:
    if isinstance(value, str):
        for root in roots:
            value = value.replace(root, "<PROJECT_ROOT>")
        return value
    if isinstance(value, list):
        return [_redact_local_paths(item, roots) for item in value]
    if isinstance(value, dict):
        return {key: _redact_local_paths(item, roots) for key, item in value.items()}
    return value


def clean(path: Path) -> tuple[int, int]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    removed = 0
    for cell in notebook.get("cells", []):
        outputs = cell.get("outputs")
        if not isinstance(outputs, list):
            continue
        kept = []
        for output in outputs:
            plain = output.get("data", {}).get("text/plain", "")
            plain = "".join(plain) if isinstance(plain, list) else str(plain)
            text = _text(output)
            is_widget = (
                output.get("output_type") == "display_data"
                and "application/vnd.jupyter.widget-view+json"
                in output.get("data", {})
            )
            is_progress = (
                "pre tokenize:" in text
                or "Inference Embeddings:" in text
                or ("Fetching " in plain and "files:" in plain)
            )
            is_empty_stream = output.get("output_type") == "stream" and not text.strip()
            if is_widget or is_progress or is_empty_stream:
                removed += 1
            else:
                kept.append(output)
        cell["outputs"] = kept
    project_root = Path(__file__).resolve().parents[1]
    roots = (str(project_root), project_root.as_posix())
    notebook = _redact_local_paths(notebook, roots)
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return removed, sum(len(cell.get("outputs", [])) for cell in notebook["cells"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage : clean_notebook_outputs.py NOTEBOOK.ipynb")
    target = Path(sys.argv[1])
    removed_count, remaining_count = clean(target)
    print(
        f"{target}: {removed_count} sorties de progression retirées, "
        f"{remaining_count} sorties utiles conservées."
    )
