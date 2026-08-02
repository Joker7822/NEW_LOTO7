#!/usr/bin/env python3
"""Run Generation 5 evolution, then choose its finalist with isolated Selection Null evidence.

The base four-island walk-forward evolution is allowed to finish first.  Only a small
walk-forward shortlist is exposed to the physically isolated Selection seed file.
The Final seed file is neither accepted as an argument nor opened by this module.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from loto7.evaluation.model_audit import year_of
from loto7.evaluation.null_permutation import adaptive_null_test
from loto7.evolution.generation5 import generation5_adoption_gate, select_pareto_records
from loto7_evolution_trainer import Genome, generate_tickets, genome_from_dict, load_draws
from merge_evolution_shards import select_target_indices
from scripts import generation5_evolver as base


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_selection_seeds(path: str | Path, *, expected_count: int = 150) -> tuple[list[int], dict[str, object]]:
    payload = read_json(path)
    phase = payload.get("phase")
    phases = payload.get("phases")
    if phase != "selection":
        raise ValueError(f"selection seed file has wrong phase: {phase!r}")
    if not isinstance(phases, Mapping) or set(phases.keys()) != {"selection"}:
        raise ValueError("selection seed file must physically contain only the selection phase")
    raw = phases.get("selection")
    if not isinstance(raw, list) or not raw:
        raise ValueError("selection seed phase is missing or empty")
    seeds = [int(value) for value in raw]
    if len(seeds) != expected_count:
        raise ValueError(f"selection seed count {len(seeds)} != expected {expected_count}")
    if len(seeds) != len(set(seeds)):
        raise ValueError("selection seed file contains duplicate seeds")
    return seeds, payload


def _public_record_with_genome(record: Mapping[str, object]) -> dict[str, object]:
    result = {key: value for key, value in record.items() if key != "_genome"}
    genome = record.get("_genome")
    if isinstance(genome, Genome):
        result["genome"] = asdict(genome)
    return result


def internal_walk_forward_key(record: Mapping[str, object]) -> tuple[float, ...]:
    walk = record.get("walk_forward")
    source = walk if isinstance(walk, Mapping) else {}
    return (
        float(source.get("generation5_score", 0.0) or 0.0),
        float(source.get("recent_weighted_objective", 0.0) or 0.0),
        float(source.get("latest_fold_objective", 0.0) or 0.0),
        float(source.get("fold_objective_min", 0.0) or 0.0),
        -float(source.get("fold_objective_stddev", 0.0) or 0.0),
        float(source.get("draw_main6_plus_count", 0) or 0),
        float(source.get("draw_main5_plus_count", 0) or 0),
        float(source.get("average_max_main_match", 0.0) or 0.0),
        -float(source.get("mean_ticket_pair_overlap", 7.0) or 7.0),
    )


def selection_candidate_key(evidence: Mapping[str, object]) -> tuple[float, ...]:
    decision = evidence.get("decision")
    decision_map = decision if isinstance(decision, Mapping) else {}
    internal = evidence.get("internal_key")
    internal_values = tuple(float(value) for value in internal) if isinstance(internal, list) else ()
    return (
        1.0 if bool(evidence.get("adoption_passed")) else 0.0,
        1.0 if bool(decision_map.get("passed")) else 0.0,
        -float(decision_map.get("wilson_ci_upper", 1.0) or 1.0),
        -float(decision_map.get("exceedance", 1.0) or 1.0),
        float(evidence.get("null_margin_vs_adjusted_p90", -1e9) or -1e9),
        *internal_values,
    )


def adoption_for_record(
    record: Mapping[str, object],
    baseline: Mapping[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    return generation5_adoption_gate(
        record,
        baseline,
        min_positive_folds=args.min_positive_folds,
        min_fold_objective_delta=args.min_fold_objective_delta,
        min_average_max_delta=args.min_average_max_delta,
        min_draw4_rate_delta_percent=args.min_draw4_rate_delta_percent,
        min_draw5_count_delta=args.min_draw5_count_delta,
        min_draw6_count_delta=args.min_draw6_count_delta,
        max_worst_fold_drop_percent=args.max_worst_fold_drop_percent,
        min_average_unique_numbers=args.min_average_unique_numbers,
        max_mean_overlap=args.max_mean_overlap,
        max_pair_overlap=args.max_pair_overlap,
        min_payout_roi_percent=args.min_payout_roi_percent,
        max_roi_drop_percent=args.max_roi_drop_percent,
        max_top1_payout_share=args.max_top1_payout_share,
    )


def evaluate_selection_null(
    record: Mapping[str, object],
    *,
    draws: Sequence[object],
    target_indices: Sequence[int],
    seeds: Sequence[int],
    purchase_count: int,
    search_width: int,
    max_exceedance: float,
    adoption: Mapping[str, object],
) -> dict[str, object]:
    raw_genome = record.get("genome")
    if not isinstance(raw_genome, Mapping):
        raise ValueError(f"candidate genome is missing: {record.get('genome_id')}")
    genome = genome_from_dict(dict(raw_genome))
    portfolios = [generate_tickets(draws[:index], genome, purchase_count) for index in target_indices]
    mains = [draws[index].main for index in target_indices]  # type: ignore[attr-defined]
    result = adaptive_null_test(
        portfolios=portfolios,
        mains=mains,
        seeds=seeds,
        checkpoints=[len(seeds)],
        search_width=search_width,
        max_exceedance=max_exceedance,
    )
    null_distribution = result.get("null_distribution")
    null_map = null_distribution if isinstance(null_distribution, Mapping) else {}
    observed = float(result.get("observed_score", 0.0) or 0.0)
    adjusted_p90 = float(null_map.get("adjusted_p90", 0.0) or 0.0)
    return {
        "genome_id": record.get("genome_id"),
        "island": record.get("island"),
        "generation": record.get("generation"),
        "adoption_passed": bool(adoption.get("passed")),
        "internal_key": list(internal_walk_forward_key(record)),
        "observed_score": observed,
        "null_adjusted_p90": adjusted_p90,
        "null_margin_vs_adjusted_p90": round(observed - adjusted_p90, 6),
        "decision": result.get("decision"),
        "adaptive_checkpoints": result.get("adaptive_checkpoints"),
        "null_distribution": result.get("null_distribution"),
        "search_width": search_width,
        "selection_seed_count": len(seeds),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(add_help=False)
    result.add_argument(
        "--selection-seed-bank",
        default="outputs/generation5/selection_null_seed_bank.json",
    )
    result.add_argument(
        "--selection-null-summary",
        default="outputs/generation5/selection_null_candidate_summary.json",
    )
    result.add_argument("--selection-null-shortlist", type=int, default=4)
    result.add_argument("--selection-null-start-year", type=int, default=2020)
    result.add_argument("--selection-null-search-width", type=int, default=6)
    result.add_argument("--selection-null-max-exceedance", type=float, default=0.10)
    result.add_argument("--selection-null-seed-count", type=int, default=150)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    selection_args, base_argv = parser().parse_known_args(list(argv) if argv is not None else None)
    base_args = base.parser().parse_args(base_argv)
    if selection_args.selection_null_shortlist <= 0:
        raise SystemExit("selection-null-shortlist must be positive")
    if selection_args.selection_null_search_width <= 0:
        raise SystemExit("selection-null-search-width must be positive")

    seeds, seed_payload = load_selection_seeds(
        selection_args.selection_seed_bank,
        expected_count=selection_args.selection_null_seed_count,
    )

    original_public_record = base.public_record
    base.public_record = _public_record_with_genome
    try:
        result = base.main(list(base_argv))
    finally:
        base.public_record = original_public_record
    if result != 0:
        return result

    summary = read_json(base_args.summary)
    baseline = summary.get("baseline")
    archive = summary.get("archive")
    if not isinstance(baseline, Mapping) or not isinstance(archive, Mapping):
        raise SystemExit("Generation 5 summary is missing baseline/archive records")

    records: list[Mapping[str, object]] = []
    for island_records in archive.values():
        if isinstance(island_records, list):
            records.extend(item for item in island_records if isinstance(item, Mapping))
    if not records:
        raise SystemExit("Generation 5 archive has no candidates for Selection Null screening")

    global_front = select_pareto_records(
        records,
        limit=max(1, len(records)),
        island="robust_diversity",
    )
    shortlist = sorted(global_front, key=internal_walk_forward_key, reverse=True)[
        : selection_args.selection_null_shortlist
    ]

    draws = load_draws(base_args.csv)
    target_indices = select_target_indices(
        draws,
        min_train_draws=base_args.min_train_draws,
        holdout_start_draw=base_args.holdout_start_draw,
        holdout_end_draw=None,
    )
    if base_args.max_targets > 0:
        target_indices = target_indices[-base_args.max_targets :]
    selection_indices = [
        index
        for index in target_indices
        if year_of(draws[index]) >= selection_args.selection_null_start_year
    ]
    if not selection_indices:
        raise SystemExit("Selection Null target period is empty")

    evidence_by_id: dict[str, dict[str, object]] = {}
    adoption_by_id: dict[str, dict[str, object]] = {}
    record_by_id: dict[str, Mapping[str, object]] = {}
    for record in shortlist:
        genome_id = str(record.get("genome_id", ""))
        if not genome_id:
            raise SystemExit("shortlisted Generation 5 candidate has no genome_id")
        adoption = adoption_for_record(record, baseline, base_args)
        evidence = evaluate_selection_null(
            record,
            draws=draws,
            target_indices=selection_indices,
            seeds=seeds,
            purchase_count=base_args.purchase_count,
            search_width=selection_args.selection_null_search_width,
            max_exceedance=selection_args.selection_null_max_exceedance,
            adoption=adoption,
        )
        evidence_by_id[genome_id] = evidence
        adoption_by_id[genome_id] = adoption
        record_by_id[genome_id] = record
        print("GEN5_SELECTION_NULL=" + json.dumps(evidence, ensure_ascii=False, sort_keys=True))

    selected_id = max(evidence_by_id, key=lambda genome_id: selection_candidate_key(evidence_by_id[genome_id]))
    selected_record = record_by_id[selected_id]
    selected_adoption = adoption_by_id[selected_id]
    selected_evidence = evidence_by_id[selected_id]
    raw_genome = selected_record.get("genome")
    if not isinstance(raw_genome, Mapping):
        raise SystemExit("selected Generation 5 genome payload is missing")

    previous_candidate = summary.get("candidate")
    summary["walk_forward_candidate"] = previous_candidate
    summary["candidate"] = dict(selected_record)
    summary["adoption"] = selected_adoption
    summary["selection_mode"] = "walk_forward_pareto_then_isolated_selection_null_150"
    summary["selection_null"] = {
        "selected_genome_id": selected_id,
        "selection_seed_bank": selection_args.selection_seed_bank,
        "selection_seed_bank_sha256": file_sha256(selection_args.selection_seed_bank),
        "selection_seed_count": len(seeds),
        "selection_phase": seed_payload.get("phase"),
        "physically_isolated_phase_file": True,
        "final_phase_accessed": False,
        "target_draws": len(selection_indices),
        "start_year": selection_args.selection_null_start_year,
        "search_width": selection_args.selection_null_search_width,
        "max_null_exceedance": selection_args.selection_null_max_exceedance,
        "ranking_order": [
            "internal_adoption_pass",
            "selection_null_pass",
            "lower_wilson_upper",
            "lower_null_exceedance",
            "higher_margin_vs_adjusted_p90",
            "walk_forward_quality",
        ],
        "shortlist": [evidence_by_id[str(item.get("genome_id"))] for item in shortlist],
    }
    write_json(base_args.summary, summary)

    candidate_payload: dict[str, object] = {
        "updated_at": now_iso(),
        "kind": "loto7_generation5_candidate_model",
        "source": "generation5_selection_evolver",
        "objective": summary.get("objective"),
        "objective_version": summary.get("objective_version"),
        "dataset_sha256": summary.get("dataset_sha256"),
        "baseline_model_sha256": summary.get("baseline_model_sha256"),
        "stable_seed": summary.get("stable_seed"),
        "selection_mode": summary.get("selection_mode"),
        "genome": dict(raw_genome),
        "selected_holdout": selected_record.get("full_metrics"),
        "walk_forward": selected_record.get("walk_forward"),
        "fold_metrics": selected_record.get("fold_metrics"),
        "adoption_gate": selected_adoption,
        "selection_null_evidence": selected_evidence,
        "notes": [
            "The four-island walk-forward evolution completes before Selection Null screening.",
            "Only the physically isolated selection-phase seed file is available to candidate selection.",
            "The fixed Final 150 seed file is reserved for post-selection promotion evidence and is not read here.",
            "Selection Null uses payout-independent hit-first permutation evidence.",
            "Historical evaluation does not guarantee future lottery winnings.",
        ],
    }
    write_json(base_args.candidate_model, candidate_payload)

    selection_summary: dict[str, object] = {
        "created_at": now_iso(),
        "kind": "loto7_generation5_selection_null_screen",
        "selected_genome_id": selected_id,
        "candidate_model": base_args.candidate_model,
        "selection_seed_bank": selection_args.selection_seed_bank,
        "selection_seed_bank_sha256": file_sha256(selection_args.selection_seed_bank),
        "selection_seed_count": len(seeds),
        "selection_phase": "selection",
        "physically_isolated_phase_file": True,
        "final_phase_accessed": False,
        "target_draws": len(selection_indices),
        "start_year": selection_args.selection_null_start_year,
        "search_width": selection_args.selection_null_search_width,
        "shortlist_size": len(shortlist),
        "shortlist": summary["selection_null"]["shortlist"],  # type: ignore[index]
    }
    write_json(selection_args.selection_null_summary, selection_summary)
    base.write_report(base_args.report, summary)
    print(
        json.dumps(
            {
                "selected_genome_id": selected_id,
                "selection_seed_count": len(seeds),
                "selection_null_passed": bool(
                    (selected_evidence.get("decision") or {}).get("passed")
                    if isinstance(selected_evidence.get("decision"), Mapping)
                    else False
                ),
                "final_phase_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
