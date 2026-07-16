#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the fourth-generation LOTO7 prediction portfolio.

Generation 4 combines the existing model candidate pools with:
- rolling conformal number-pool calibration,
- Bayesian live source weighting,
- change-point gating,
- DPP diversity,
- hypergraph pair/triple coverage,
- shadow champion/challenger outputs.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loto7_evolution_trainer import Draw, Genome, load_draws  # noqa: E402
from merge_evolution_shards import fmt_ticket, write_prediction, write_prediction_report  # noqa: E402
from scripts.build_dual_model_prediction import (  # noqa: E402
    Candidate,
    load_history_tickets,
    load_optional_model,
    load_required_model,
    model_candidate_pool,
    overlap,
    regime_candidate_pool,
    select_portfolio,
)
from scripts.generation4_core import (  # noqa: E402
    allocate_quotas,
    conformal_number_pool,
    detect_change_point,
    dpp_logdet,
    dynamic_source_weights,
    hypergraph_coverage_score,
    hypergraph_weights,
    strategy_posteriors,
)

Ticket = Tuple[int, ...]


@dataclass
class G4State:
    selected: Tuple[Candidate, ...]
    usage: Counter[int]
    score: float
    components: Dict[str, float]


def entropy_from_usage(usage: Counter[int]) -> float:
    total = sum(usage.values()) or 1
    return -sum((count / total) * math.log(count / total) for count in usage.values() if count > 0)


def adjusted_candidate(
    candidate: Candidate,
    *,
    source_weight: float,
    conformal_numbers: set[int],
    minimum_conformal_hits: int,
) -> Candidate:
    hits = len(set(candidate.ticket) & conformal_numbers)
    conformal_bonus = hits * 1.7
    if hits < minimum_conformal_hits:
        conformal_bonus -= (minimum_conformal_hits - hits) * 14.0
    source_bonus = math.log(max(1e-9, source_weight) + 1.0) * 18.0
    return Candidate(
        ticket=candidate.ticket,
        source=candidate.source,
        model_id=candidate.model_id,
        model_score=candidate.model_score,
        source_model=candidate.source_model,
        method=candidate.method,
        support=(candidate.support + f" / G4 source_weight={source_weight:.4f}" + f" / conformal_hits={hits}"),
        individual_score=candidate.individual_score + source_bonus + conformal_bonus,
        raw_rank=candidate.raw_rank,
        created_at=candidate.created_at,
    )


def portfolio_components(
    selected: Sequence[Candidate],
    *,
    graph_weights: Mapping[str, Mapping[Tuple[int, ...], float]],
    conformal_numbers: set[int],
    dpp_weight: float,
    hypergraph_weight: float,
    conformal_weight: float,
) -> Dict[str, float]:
    if not selected:
        return {"quality": 0.0, "dpp_logdet": 0.0, "hypergraph": 0.0, "conformal": 0.0,
                "entropy": 0.0, "overlap_penalty": 0.0, "total": 0.0}
    tickets = [candidate.ticket for candidate in selected]
    sources = [candidate.source for candidate in selected]
    qualities = [candidate.individual_score for candidate in selected]
    quality = sum(qualities) / 100.0
    dpp = dpp_logdet(tickets, sources, qualities)
    graph = hypergraph_coverage_score(tickets, graph_weights)
    conformal_hits = sum(len(set(ticket) & conformal_numbers) for ticket in tickets)
    usage: Counter[int] = Counter(number for ticket in tickets for number in ticket)
    entropy = entropy_from_usage(usage)
    overlaps = [overlap(left, right) for left, right in combinations(tickets, 2)]
    overlap_penalty = sum(value * value for value in overlaps) / 12.0
    total = (quality + dpp * dpp_weight + float(graph["total"]) * hypergraph_weight
             + conformal_hits * conformal_weight + entropy * 1.5 - overlap_penalty)
    return {
        "quality": quality,
        "dpp_logdet": dpp,
        "hypergraph": float(graph["total"]),
        "hypergraph_pair": float(graph["pair_score"]),
        "hypergraph_triple": float(graph["triple_score"]),
        "conformal": float(conformal_hits),
        "entropy": entropy,
        "overlap_penalty": overlap_penalty,
        "total": total,
    }


def source_sequence(quotas: Mapping[str, int], pools: Mapping[str, Sequence[Candidate]]) -> List[str]:
    active = [(source, count) for source, count in quotas.items() if count > 0]
    output: List[str] = []
    for source, count in sorted(active, key=lambda item: (len(pools.get(item[0], [])), item[0])):
        output.extend([source] * count)
    return output


def select_generation4_portfolio(
    pools: Mapping[str, Sequence[Candidate]],
    quotas: Mapping[str, int],
    *,
    graph_weights: Mapping[str, Mapping[Tuple[int, ...], float]],
    conformal_numbers: set[int],
    max_number_usage: int,
    max_pair_overlap: int,
    beam_width: int,
    candidates_per_step: int,
    dpp_weight: float,
    hypergraph_weight: float,
    conformal_weight: float,
) -> Tuple[List[Candidate], Dict[str, object]]:
    sequence = source_sequence(quotas, pools)
    if not sequence:
        raise SystemExit("generation4 quotas selected zero tickets")
    states = [G4State(tuple(), Counter(), 0.0, {})]
    for position, source in enumerate(sequence, start=1):
        candidates = list(pools.get(source, []))[:candidates_per_step]
        if not candidates:
            raise SystemExit(f"no generation4 candidates for source={source}")
        expanded: List[G4State] = []
        for state in states:
            existing_tickets = [candidate.ticket for candidate in state.selected]
            for candidate in candidates:
                if candidate.ticket in existing_tickets:
                    continue
                if any(overlap(candidate.ticket, ticket) > max_pair_overlap for ticket in existing_tickets):
                    continue
                usage = state.usage.copy()
                valid = True
                for number in candidate.ticket:
                    usage[number] += 1
                    if usage[number] > max_number_usage:
                        valid = False
                        break
                if not valid:
                    continue
                selected = state.selected + (candidate,)
                components = portfolio_components(
                    selected,
                    graph_weights=graph_weights,
                    conformal_numbers=conformal_numbers,
                    dpp_weight=dpp_weight,
                    hypergraph_weight=hypergraph_weight,
                    conformal_weight=conformal_weight,
                )
                expanded.append(G4State(selected, usage, components["total"], components))
        if not expanded:
            raise SystemExit(f"generation4 DPP search exhausted at position={position} source={source}")
        expanded.sort(key=lambda state: state.score, reverse=True)
        states = expanded[:max(1, beam_width)]
    best = max(states, key=lambda state: state.score)
    selected = sorted(best.selected, key=lambda candidate: candidate.individual_score, reverse=True)
    tickets = [candidate.ticket for candidate in selected]
    overlaps = [overlap(left, right) for left, right in combinations(tickets, 2)]
    summary: Dict[str, object] = {
        "objective_score": round(best.score, 6),
        "objective_components": {key: round(value, 6) for key, value in best.components.items()},
        "source_quotas": dict(quotas),
        "source_counts": dict(Counter(candidate.source for candidate in selected)),
        "candidate_pool_counts": {source: len(pool) for source, pool in pools.items()},
        "unique_number_count": len({number for ticket in tickets for number in ticket}),
        "max_number_usage": max(best.usage.values()) if best.usage else 0,
        "number_usage": {str(number): count for number, count in sorted(best.usage.items())},
        "max_pair_overlap": max(overlaps) if overlaps else 0,
        "average_pair_overlap": round(sum(overlaps) / len(overlaps), 3) if overlaps else 0.0,
        "post_selection_number_replacements": 0,
        "selector": "dpp_hypergraph_beam_search_v1",
    }
    return selected, summary


def constrained_source_portfolio(
    candidates: Sequence[Candidate], *, count: int, max_number_usage: int, max_pair_overlap: int
) -> List[Candidate]:
    selected: List[Candidate] = []
    usage: Counter[int] = Counter()
    for candidate in candidates:
        if any(overlap(candidate.ticket, previous.ticket) > max_pair_overlap for previous in selected):
            continue
        next_usage = usage.copy()
        for number in candidate.ticket:
            next_usage[number] += 1
        if any(value > max_number_usage for value in next_usage.values()):
            continue
        selected.append(candidate)
        usage = next_usage
        if len(selected) >= count:
            return selected
    for candidate in candidates:
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) >= count:
            break
    return selected[:count]


def random_control(draw_no: int, count: int) -> List[Ticket]:
    rng = random.Random(draw_no * 10007 + 4)
    selected: List[Ticket] = []
    usage: Counter[int] = Counter()
    attempts = 0
    while len(selected) < count and attempts < 10000:
        attempts += 1
        ticket = tuple(sorted(rng.sample(range(1, 38), 7)))
        if ticket in selected or any(overlap(ticket, previous) > 4 for previous in selected):
            continue
        next_usage = usage.copy()
        for number in ticket:
            next_usage[number] += 1
        if any(value > 4 for value in next_usage.values()):
            continue
        selected.append(ticket)
        usage = next_usage
    return selected


def serialize_tickets(tickets: Sequence[Ticket]) -> List[str]:
    return [fmt_ticket(ticket) for ticket in tickets]


def load_null_league(path: str) -> Dict[str, object]:
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size <= 0:
        return {"available": False, "passed": None}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"available": False, "passed": None}
    decision = payload.get("decision", {}) if isinstance(payload, dict) else {}
    return {
        "available": True,
        "passed": decision.get("passed") if isinstance(decision, dict) else None,
        "model_percentile": payload.get("model_percentile") if isinstance(payload, dict) else None,
        "pbo": payload.get("pbo") if isinstance(payload, dict) else None,
        "summary_path": path,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the complete LOTO7 generation 4 prediction.")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--full-model", default="loto7_best_model.json")
    parser.add_argument("--recent-model", default="outputs/recent_era/recent_era_best_model.json")
    parser.add_argument("--super-recent-model", default="outputs/super_recent/super_recent_best_model.json")
    parser.add_argument("--regime-strategy", default="outputs/role_ensemble/regime_strategy.json")
    parser.add_argument("--prediction-history", default="outputs/evolution_prediction_history.csv")
    parser.add_argument("--shadow-history", default="outputs/generation4/shadow_history.csv")
    parser.add_argument("--null-league-summary", default="outputs/generation4/null_strategy_league_summary.json")
    parser.add_argument("--prediction", default="outputs/evolution_best_prediction.csv")
    parser.add_argument("--prediction-report", default="outputs/holdout/latest_prediction_report.txt")
    parser.add_argument("--summary", default="outputs/generation4/latest_generation4_summary.json")
    parser.add_argument("--shadow-output", default="outputs/generation4/latest_shadow_predictions.json")
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--candidates-per-source", type=int, default=220)
    parser.add_argument("--candidates-per-step", type=int, default=80)
    parser.add_argument("--beam-width", type=int, default=220)
    parser.add_argument("--max-number-usage", type=int, default=4)
    parser.add_argument("--max-pair-overlap", type=int, default=4)
    parser.add_argument("--history-overlap-limit", type=int, default=6)
    parser.add_argument("--conformal-alpha", type=float, default=0.20)
    parser.add_argument("--conformal-calibration-draws", type=int, default=104)
    parser.add_argument("--minimum-conformal-hits", type=int, default=4)
    parser.add_argument("--dpp-weight", type=float, default=3.0)
    parser.add_argument("--hypergraph-weight", type=float, default=0.075)
    parser.add_argument("--conformal-weight", type=float, default=0.55)
    args = parser.parse_args(argv)

    draws = load_draws(args.csv)
    if not draws:
        raise SystemExit(f"no draws loaded: {args.csv}")
    latest: Draw = draws[-1]
    history = load_history_tickets(args.prediction_history)
    full_item = load_required_model(args.full_model)
    recent_item, recent_fallback = load_optional_model(args.recent_model, full_item)
    super_item, super_fallback = load_optional_model(args.super_recent_model, recent_item)
    full_genome: Genome = full_item["genome"]  # type: ignore[assignment]
    recent_genome: Genome = recent_item["genome"]  # type: ignore[assignment]
    super_genome: Genome = super_item["genome"]  # type: ignore[assignment]
    super_independent = bool(not super_fallback and super_genome.id != recent_genome.id)

    conformal = conformal_number_pool(draws, alpha=args.conformal_alpha,
                                      calibration_draws=args.conformal_calibration_draws)
    conformal_numbers = {int(number) for number in conformal.get("numbers", [])}
    change_point = detect_change_point(draws)
    posterior = strategy_posteriors(args.shadow_history, strategies=("full", "recent", "super", "regime"))
    weights = dynamic_source_weights(posterior.get("weights", {}), change_point,
                                     super_independent=super_independent)
    null_league = load_null_league(args.null_league_summary)
    if null_league.get("available") and null_league.get("passed") is False:
        weights["full"] *= 0.80
        total = sum(weights.values()) or 1.0
        weights = {source: value / total for source, value in weights.items()}
    quotas = allocate_quotas(weights, purchase_count=args.purchase_count,
                             super_independent=super_independent)

    original_pools: Dict[str, Sequence[Candidate]] = {
        "full": model_candidate_pool(
            genome=full_genome, source="full", source_model=str(full_item.get("path") or args.full_model),
            method="generation4_full", support="Full model candidate", draws=draws, history=history,
            candidate_count=args.candidates_per_source, history_overlap_limit=args.history_overlap_limit,
            diversity_weight=1.15, first_prize_weight=1.25, created_at=latest.date),
        "recent": model_candidate_pool(
            genome=recent_genome, source="recent", source_model=str(recent_item.get("path") or args.recent_model),
            method="generation4_recent", support="Recent model candidate" if not recent_fallback else "Recent fallback",
            draws=draws, history=history, candidate_count=args.candidates_per_source,
            history_overlap_limit=args.history_overlap_limit, diversity_weight=1.20,
            first_prize_weight=1.25, created_at=latest.date),
        "super": model_candidate_pool(
            genome=super_genome, source="super", source_model=str(super_item.get("path") or args.super_recent_model),
            method="generation4_super", support="Independent Super Recent candidate", draws=draws, history=history,
            candidate_count=args.candidates_per_source, history_overlap_limit=args.history_overlap_limit,
            diversity_weight=1.25, first_prize_weight=1.35, created_at=latest.date) if super_independent else [],
        "regime": regime_candidate_pool(
            genome=full_genome, source_model=str(full_item.get("path") or args.full_model), draws=draws,
            history=history, strategy_path=args.regime_strategy, candidate_count=args.candidates_per_source,
            history_overlap_limit=args.history_overlap_limit, diversity_weight=1.20,
            first_prize_weight=1.25, created_at=latest.date),
    }
    pools: Dict[str, Sequence[Candidate]] = {
        source: [adjusted_candidate(candidate, source_weight=weights.get(source, 0.0),
                                    conformal_numbers=conformal_numbers,
                                    minimum_conformal_hits=args.minimum_conformal_hits)
                 for candidate in candidates]
        for source, candidates in original_pools.items()
    }
    graph = hypergraph_weights(draws)
    selected, portfolio_summary = select_generation4_portfolio(
        pools, quotas, graph_weights=graph, conformal_numbers=conformal_numbers,
        max_number_usage=args.max_number_usage, max_pair_overlap=args.max_pair_overlap,
        beam_width=args.beam_width, candidates_per_step=args.candidates_per_step,
        dpp_weight=args.dpp_weight, hypergraph_weight=args.hypergraph_weight,
        conformal_weight=args.conformal_weight)

    rows: List[Dict[str, object]] = []
    for rank, candidate in enumerate(selected, start=1):
        rows.append({
            "confidence_rank": rank, "base_latest_draw_no": latest.draw_no,
            "base_latest_date": latest.date, "prediction_draw_no": latest.draw_no + 1,
            "combo_index": rank, "numbers": fmt_ticket(candidate.ticket),
            "model_id": candidate.model_id, "model_score": round(candidate.model_score, 6),
            "source_model": candidate.source_model, "prediction_method": candidate.method,
            "ensemble_score": round(max(0.50, 0.95 - (rank - 1) * 0.05), 2),
            "support_models": candidate.support + " / DPP+Hypergraph+Conformal / replacement=0",
            "created_at": candidate.created_at,
        })
    write_prediction(args.prediction, rows)
    write_prediction_report(
        args.prediction_report, rows, full_genome, str(full_item.get("path") or args.full_model),
        model_count=len({candidate.model_id for candidate in selected}), min_models=1,
        selection_reason="Generation 4: Bayesian DMA + Change-Point + Conformal + DPP + Hypergraph",
        prediction_mode="generation4_complete", role_strategy_path=args.regime_strategy)

    baseline_selected, baseline_summary = select_portfolio(
        original_pools, quotas, max_number_usage=args.max_number_usage,
        max_pair_overlap=args.max_pair_overlap, beam_width=min(args.beam_width, 220),
        candidates_per_step=min(args.candidates_per_step, 80))
    strategies: Dict[str, List[Ticket]] = {
        "generation4": [candidate.ticket for candidate in selected],
        "beam_baseline": [candidate.ticket for candidate in baseline_selected],
        "full": [c.ticket for c in constrained_source_portfolio(original_pools["full"], count=5,
                                                                  max_number_usage=4, max_pair_overlap=4)],
        "recent": [c.ticket for c in constrained_source_portfolio(original_pools["recent"], count=5,
                                                                    max_number_usage=4, max_pair_overlap=4)],
        "regime": [c.ticket for c in constrained_source_portfolio(original_pools["regime"], count=5,
                                                                    max_number_usage=4, max_pair_overlap=4)],
        "random_control": random_control(latest.draw_no + 1, args.purchase_count),
    }
    if super_independent:
        strategies["super"] = [c.ticket for c in constrained_source_portfolio(
            original_pools["super"], count=5, max_number_usage=4, max_pair_overlap=4)]
    shadow_payload = {
        "kind": "loto7_generation4_shadow_predictions",
        "base_latest_draw_no": latest.draw_no, "base_latest_date": latest.date,
        "prediction_draw_no": latest.draw_no + 1,
        "strategies": {name: serialize_tickets(tickets) for name, tickets in strategies.items()},
    }
    shadow_path = Path(args.shadow_output)
    shadow_path.parent.mkdir(parents=True, exist_ok=True)
    shadow_path.write_text(json.dumps(shadow_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")

    summary = {
        "kind": "loto7_generation4_complete", "base_latest_draw_no": latest.draw_no,
        "prediction_draw_no": latest.draw_no + 1, "super_recent_independent": super_independent,
        "bayesian_source_posterior": posterior,
        "dynamic_source_weights": {source: round(value, 9) for source, value in weights.items()},
        "change_point": change_point, "conformal_number_pool": conformal,
        "null_strategy_league": null_league, "source_quotas": quotas,
        "generation4_portfolio": portfolio_summary,
        "beam_baseline_portfolio": baseline_summary,
        "shadow_strategy_names": sorted(strategies),
        "notes": [
            "All conformal calibration targets use only earlier draws.",
            "Final tickets are original model candidates; no post-selection number replacement is allowed.",
            "Historical diagnostics do not guarantee future lottery winnings.",
        ],
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
