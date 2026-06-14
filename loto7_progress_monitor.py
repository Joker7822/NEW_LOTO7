#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate LOTO7 evolution and ML outputs into one summary."""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path
from typing import Dict, List


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default


def main() -> int:
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "loto7_progress_summary.json"
    md_path = out / "loto7_progress_summary.md"

    history_files = sorted(glob.glob("outputs/evolution_history_*.csv")) + sorted(glob.glob("outputs/evolution_history.csv"))
    state_files = sorted(glob.glob("outputs/evolution_state_*.json"))
    best_model_files = sorted(glob.glob("loto7_best_model*.json"))
    ml_status_path = Path("outputs/ml_stack/ml_stack_status.json")
    ml_report_path = Path("outputs/ml_stack/ml_model_report.csv")

    total_rows = 0
    best_score = None
    best_row = None
    max_generation = -1
    rank_counts = {"rank_1等": 0, "rank_2等": 0, "rank_3等": 0, "rank_4等": 0, "rank_5等": 0, "rank_6等": 0, "rank_外れ": 0}

    for path in history_files:
        for row in read_csv_rows(path):
            total_rows += 1
            score = safe_float(row.get("score"))
            gen = safe_int(row.get("generation"), -1)
            max_generation = max(max_generation, gen)
            if best_score is None or score > best_score:
                best_score = score
                best_row = {**row, "source_file": path}
            for k in rank_counts:
                rank_counts[k] += safe_int(row.get(k), 0)

    states = []
    for path in state_files:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            data["source_file"] = path
            states.append(data)
        except Exception as exc:
            states.append({"source_file": path, "error": str(exc)})

    best_models = []
    for path in best_model_files:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            genome = data.get("genome", data)
            best_models.append({"source_file": path, "score": genome.get("score"), "id": genome.get("id"), "generation": genome.get("generation")})
        except Exception as exc:
            best_models.append({"source_file": path, "error": str(exc)})

    ml_status = None
    if ml_status_path.exists():
        try:
            ml_status = json.loads(ml_status_path.read_text(encoding="utf-8"))
        except Exception as exc:
            ml_status = {"error": str(exc)}

    ml_reports = read_csv_rows(str(ml_report_path))

    summary = {
        "history_files": history_files,
        "state_files": state_files,
        "best_model_files": best_model_files,
        "evaluated_genomes_rows": total_rows,
        "max_generation_seen": max_generation,
        "best_score": best_score,
        "best_row": best_row,
        "rank_counts": rank_counts,
        "states": states,
        "best_models": best_models,
        "ml_status": ml_status,
        "ml_reports": ml_reports,
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# LOTO7 Progress Summary",
        "",
        f"- evaluated_genomes_rows: {total_rows}",
        f"- max_generation_seen: {max_generation}",
        f"- best_score: {best_score}",
        f"- history_files: {len(history_files)}",
        f"- state_files: {len(state_files)}",
        f"- best_model_files: {len(best_model_files)}",
        "",
        "## Rank counts",
    ]
    for k, v in rank_counts.items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## Best row", "```json", json.dumps(best_row, ensure_ascii=False, indent=2), "```", "", "## ML reports"])
    for r in ml_reports:
        lines.append(f"- {r}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
