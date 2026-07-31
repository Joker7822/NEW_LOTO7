#!/usr/bin/env python3
"""Promote a Generation 5 candidate only after every hardened gate passes."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

from scripts.prepromotion_hardening import run_prepromotion_hardening


def read_json(path: str) -> dict[str, object]:
    file = Path(path)
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def write_json(path: str, payload: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", default="outputs/generation5/generation5_candidate_model.json"
    )
    parser.add_argument("--baseline", default="loto7_best_model.json")
    parser.add_argument("--summary", default="outputs/generation5/generation5_summary.json")
    parser.add_argument(
        "--null-summary", default="outputs/generation5/null_strategy_league_summary.json"
    )
    parser.add_argument(
        "--financial-null-summary", default="outputs/generation5/financial_null_summary.json"
    )
    parser.add_argument("--seed-bank", default="outputs/generation5/null_seed_bank.json")
    parser.add_argument("--ablation", default="outputs/generation5/prediction_ablation.json")
    parser.add_argument("--decision", default="outputs/generation5/promotion_decision.json")
    parser.add_argument("--report", default="outputs/generation5/promotion_report.txt")
    parser.add_argument("--run-status", default="outputs/generation5/run_status.json")
    parser.add_argument("--max-pbo", type=float, default=0.40)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    run_prepromotion_hardening(
        candidate=args.candidate,
        baseline=args.baseline,
        summary=args.summary,
        null_summary=args.null_summary,
        financial_null_summary=args.financial_null_summary,
        seed_bank=args.seed_bank,
        ablation=args.ablation,
    )

    candidate = read_json(args.candidate)
    baseline = read_json(args.baseline)
    summary = read_json(args.summary)
    null_summary = read_json(args.null_summary)
    financial_null = read_json(args.financial_null_summary)
    adoption = mapping(summary.get("adoption"))
    null_decision = mapping(null_summary.get("decision"))
    candidate_metrics = mapping(mapping(summary.get("candidate")).get("full_metrics"))
    financial_decision = mapping(financial_null.get("decision"))
    candidate_genome = mapping(candidate.get("genome"))
    baseline_genome = mapping(baseline.get("genome", baseline))
    candidate_id = str(candidate_genome.get("id", ""))
    baseline_id = str(baseline_genome.get("id", ""))
    candidate_sha = sha256(args.candidate)
    baseline_sha = sha256(args.baseline)
    pbo = float(financial_null.get("pbo", 1.0) or 1.0)

    checks = {
        "generation5_hardened_gate": bool(adoption.get("passed")),
        "hit_first_null_gate": bool(null_decision.get("passed")),
        "financial_pbo_gate": pbo <= args.max_pbo,
        "financial_diagnostic_available": bool(financial_decision),
        "metric_schema_v2": str(candidate_metrics.get("metric_schema_version", ""))
        == "loto7-metrics-2026.07.31-v2",
        "ablation_available": Path(args.ablation).exists()
        and Path(args.ablation).stat().st_size > 0,
        "different_genome_id": bool(candidate_id and candidate_id != baseline_id),
        "different_model_sha256": candidate_sha != baseline_sha,
        "candidate_objective_version_present": bool(candidate.get("objective_version")),
    }
    failures = [name for name, passed in checks.items() if not passed]
    promoted = not failures
    applied = bool(promoted and args.apply)
    if applied:
        shutil.copyfile(args.candidate, args.baseline)

    payload: dict[str, object] = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "kind": "loto7_generation5_hardened_promotion_decision",
        "promoted": promoted,
        "apply_requested": bool(args.apply),
        "applied": applied,
        "candidate": args.candidate,
        "baseline": args.baseline,
        "candidate_genome_id": candidate_id,
        "baseline_genome_id": baseline_id,
        "candidate_sha256": candidate_sha,
        "baseline_sha256_before": baseline_sha,
        "checks": checks,
        "failures": failures,
        "internal_gate": adoption,
        "hit_first_null_gate": null_decision,
        "financial_null_gate": {
            "pbo": pbo,
            "max_pbo": args.max_pbo,
            "decision": financial_decision,
        },
    }
    write_json(args.decision, payload)
    lines = [
        "LOTO7 Generation 5 Hardened Promotion Decision",
        "================================================",
        "",
        f"promoted: {promoted}",
        f"apply_requested: {args.apply}",
        f"applied: {applied}",
        f"candidate_genome_id: {candidate_id}",
        f"baseline_genome_id: {baseline_id}",
        f"financial_pbo: {pbo:.6f}",
        "failures:",
        *([f"- {item}" for item in failures] or ["- none"]),
    ]
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        args.run_status,
        {
            "kind": "loto7_generation5_run_status",
            "created_at": payload["created_at"],
            "status": "complete",
            "promoted": promoted,
            "applied": applied,
            "required_outputs": [
                args.summary,
                args.null_summary,
                args.financial_null_summary,
                args.ablation,
                args.decision,
            ],
        },
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
