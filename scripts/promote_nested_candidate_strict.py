#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-closed nested candidate promotion with aggregate ROI rejection."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.strict_adoption_gates import nested_total_roi_gate, read_json, write_json  # noqa: E402


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def option_value(args: List[str], name: str) -> str:
    try:
        index = args.index(name)
        return args[index + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"required option missing: {name}") from exc


def write_rejection_report(path: str, payload: Dict[str, object]) -> None:
    gate = payload.get("strict_nested_total_roi_gate", {})
    lines = [
        "LOTO7 Strict Nested Candidate Promotion",
        "=======================================",
        "",
        f"created_at: {payload.get('created_at')}",
        "promoted: False",
        "",
        "[Aggregate Nested ROI Gate]",
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "Candidate promotion was completely rejected before production model replacement.",
        "Historical validation does not guarantee future lottery results.",
    ]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--min-nested-total-roi-percent", type=float, default=8.0)
    parser.add_argument("--min-nested-total-roi-delta-percent", type=float, default=0.0)
    strict, delegated = parser.parse_known_args(raw_args)

    baseline_model = option_value(delegated, "--baseline-model")
    candidate_model = option_value(delegated, "--candidate-model")
    best_model = option_value(delegated, "--best-model")
    nested_summary_path = option_value(delegated, "--nested-summary")
    decision_path = option_value(delegated, "--decision")
    report_path = option_value(delegated, "--report")

    for required_path in (baseline_model, candidate_model, nested_summary_path):
        if not Path(required_path).exists():
            raise SystemExit(f"required file missing: {required_path}")

    from scripts.robust_model_metrics import load_genome  # noqa: E402

    candidate_model_id = load_genome(candidate_model).id
    nested = read_json(nested_summary_path)
    gate = nested_total_roi_gate(
        nested,
        min_candidate_roi_percent=float(strict.min_nested_total_roi_percent),
        min_roi_delta_percent=float(strict.min_nested_total_roi_delta_percent),
        expected_model_id=candidate_model_id,
    )

    if not gate["passed"]:
        if Path(baseline_model).resolve() != Path(best_model).resolve():
            Path(best_model).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(baseline_model, best_model)
        payload: Dict[str, object] = {
            "created_at": now_iso(),
            "kind": "loto7_nested_candidate_promotion_strict",
            "baseline_model_path": baseline_model,
            "candidate_model_path": candidate_model,
            "candidate_model_id": candidate_model_id,
            "best_model_path": best_model,
            "nested_summary": nested,
            "strict_nested_total_roi_gate": gate,
            "decision": {
                "promoted": False,
                "reasons": [],
                "warnings": list(gate.get("failures", [])),
                "complete_rejection": True,
            },
        }
        write_json(decision_path, payload)
        write_rejection_report(report_path, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    import scripts.promote_nested_candidate as original  # noqa: E402

    previous_argv = sys.argv
    try:
        sys.argv = [str(Path(original.__file__).resolve()), *delegated]
        result = int(original.main())
    finally:
        sys.argv = previous_argv

    decision_file = Path(decision_path)
    if decision_file.exists() and decision_file.stat().st_size > 0:
        payload = read_json(decision_path)
        payload["strict_nested_total_roi_gate"] = gate
        decision = payload.get("decision")
        if isinstance(decision, dict):
            decision["complete_rejection"] = False
        write_json(decision_path, payload)
    with Path(report_path).open("a", encoding="utf-8") as stream:
        stream.write("\n[Strict Aggregate Nested ROI Gate]\n")
        stream.write(json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
