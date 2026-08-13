"""Recalcule les metriques d'un run Banque 100 sans rappeler le LLM."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation_metrics import (  # noqa: E402
    answerable_automated_pass,
    is_abstention,
    reference_fact_metrics,
    route_compatible,
)


def _latest_run() -> Path:
    runs = sorted(
        (ROOT / "data" / "evaluation_runs").glob("*_banque_100"),
        key=lambda path: path.stat().st_mtime,
    )
    if not runs:
        raise FileNotFoundError("Aucun run Banque 100 disponible.")
    return runs[-1]


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else 0.0


def rescore(run_dir: Path) -> dict:
    source = run_dir / "results_complete.jsonl"
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        answer = row.get("generated_answer", "")
        if row.get("should_answer"):
            metrics = reference_fact_metrics(row.get("reference_answer", ""), answer)
            row["reference_fact_coverage"] = metrics.score
            row["reference_numeric_recall"] = metrics.numeric_recall
            row["reference_unit_recall"] = metrics.unit_recall
            row["reference_lexical_recall"] = metrics.lexical_recall
            row["abstained"] = is_abstention(answer)
            row["automated_pass"] = answerable_automated_pass(
                row.get("reference_answer", ""),
                answer,
                citation_source_hit=bool(row.get("citation_source_hit")),
            )
        else:
            row["abstained"] = is_abstention(answer)
            row["automated_pass"] = row["abstained"]
        row["route_exact"] = row.get("route") == row.get("expected_route")
        row["route_compatible"] = route_compatible(
            row.get("route", ""), row.get("expected_route", ""),
        )
        row["route_correct"] = row["route_compatible"]

    successful = [row for row in rows if not row.get("error")]
    answerable = [row for row in successful if row.get("should_answer")]
    unanswerable = [row for row in successful if not row.get("should_answer")]
    summary = {
        "rescored_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_results": str(source),
        "questions_completed": len(successful),
        "automated_pass_rate": _mean(successful, "automated_pass"),
        "answerable_pass_rate": _mean(answerable, "automated_pass"),
        "abstention_accuracy": _mean(unanswerable, "abstained"),
        "reference_fact_coverage_mean": _mean(answerable, "reference_fact_coverage"),
        "route_compatibility_rate": _mean(successful, "route_compatible"),
        "route_exact_rate": _mean(successful, "route_exact"),
    }

    jsonl_path = run_dir / "results_rescored.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    csv_path = run_dir / "results_rescored.csv"
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })
    (run_dir / "summary_rescored.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", type=Path)
    args = parser.parse_args()
    run_dir = (args.run_dir or _latest_run()).resolve()
    print(json.dumps(rescore(run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
