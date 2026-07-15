#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the final five-ticket LOTO7 portfolio in one optimization pass.

Full-period, Recent Era, Super Recent and regime-role models each contribute an
independent candidate pool. A beam search selects all five tickets together
under hard pair-overlap and global-number-usage constraints. Every final ticket
is an original model candidate; no number is replaced after selection.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loto7_evolution_trainer import Draw, Genome, generate_tickets, load_draws  # noqa: E402
from merge_evolution_shards import (  # noqa: E402
    ROLE_ORDER,
    clone_genome_for_role,
    fmt_ticket,
    load_model,
    load_role_strategy,
    role_ticket_score,
    ticket_key,
    write_prediction,
    write_prediction_report,
)

Ticket = Tuple[int, ...]
Pair = Tuple[int, int]
Triple = Tuple[int, int, int]


@dataclass(frozen=True)
class Candidate:
    ticket: Ticket
    source: str
    model_id: str
    model_score: float
    source_model: str
    method: str
    support: str
    individual_score: float
    raw_rank: int
    created_at: str


@dataclass
class BeamState:
    score: float
    selected: Tuple[Candidate, ...]
    usage: Counter[int]
    pair_coverage: FrozenSet[Pair]
    triple_coverage: FrozenSet[Triple]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_required_model(path: str) -> Dict[str, object]:
    item = load_model(Path(path))
    if item is None:
        raise SystemExit(f"cannot load model: {path}")
    return item


def load_optional_model(path: str, fallback: Dict[str, object]) -> Tuple[Dict[str, object], bool]:
    model_path = Path(path)
    if not model_path.exists() or model_path.stat().st_size <= 0:
        return fallback, True
    item = load_model(model_path)
    return (item, False) if item is not None else (fallback, True)


def parse_ticket(text: object) -> Optional[Ticket]:
    numbers = [int(value) for value in re.findall(r"\d+", str(text or ""))]
    if len(numbers) < 7:
        return None
    numbers = numbers[:7]
    if len(set(numbers)) != 7 or any(number < 1 or number > 37 for number in numbers):
        return None
    return ticket_key(numbers)


def load_history_tickets(path: str) -> List[Ticket]:
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size <= 0:
        return []
    output: List[Ticket] = []
    seen: Set[Ticket] = set()
    with file_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            for key, value in row.items():
                if not str(key or "").startswith("予測"):
                    continue
                ticket = parse_ticket(value)
                if ticket is None or ticket in seen:
                    continue
                seen.add(ticket)
                output.append(ticket)
    return output


def overlap(left: Ticket, right: Ticket) -> int:
    return len(set(left) & set(right))


def diversity_score(ticket: Ticket, latest: Draw) -> float:
    total = sum(ticket)
    odd = sum(1 for number in ticket if number % 2)
    span = ticket[-1] - ticket[0]
    latest_overlap = len(set(ticket) & set(latest.main))
    consecutive = sum(1 for left, right in zip(ticket, ticket[1:]) if right == left + 1)
    bands = {0 if n <= 9 else 1 if n <= 19 else 2 if n <= 29 else 3 for n in ticket}
    low = sum(1 for number in ticket if number <= 18)
    high = sum(1 for number in ticket if number >= 25)
    return (
        len(bands) * 1.5
        + (2.5 if 105 <= total <= 175 else -abs(total - 140) * 0.035)
        + (2.0 if 3 <= odd <= 4 else -1.2)
        + (1.2 if 24 <= span <= 36 else -0.8)
        + (1.0 if 2 <= low <= 5 else -0.8)
        + (1.0 if 2 <= high <= 4 else -0.6)
        + (0.8 if 1 <= latest_overlap <= 3 else -0.8)
        - max(0, consecutive - 2) * 1.0
    )


def history_penalty(ticket: Ticket, history: Sequence[Ticket], limit: int) -> float:
    if not history:
        return 0.0
    overlaps = [overlap(ticket, previous) for previous in history]
    maximum = max(overlaps)
    average = sum(overlaps) / len(overlaps)
    return max(0, maximum - limit) * 75.0 + max(0.0, average - 4.0) * 10.0


def candidate_score(
    ticket: Ticket,
    *,
    rank: int,
    history: Sequence[Ticket],
    latest: Draw,
    history_overlap_limit: int,
    diversity_weight: float,
    first_prize_weight: float,
    base: float = 100.0,
    extra: float = 0.0,
) -> float:
    balance = diversity_score(ticket, latest)
    return (
        base
        - rank * 0.12
        + balance * diversity_weight
        + balance * first_prize_weight * 0.55
        + extra
        - history_penalty(ticket, history, history_overlap_limit)
    )


def model_candidate_pool(
    *,
    genome: Genome,
    source: str,
    source_model: str,
    method: str,
    support: str,
    draws: Sequence[Draw],
    history: Sequence[Ticket],
    candidate_count: int,
    history_overlap_limit: int,
    diversity_weight: float,
    first_prize_weight: float,
    created_at: str,
) -> List[Candidate]:
    raw_tickets = generate_tickets(draws, genome, max(30, candidate_count))
    history_set = set(history)
    latest = draws[-1]
    output: List[Candidate] = []
    seen: Set[Ticket] = set()
    for rank, raw_ticket in enumerate(raw_tickets):
        ticket = ticket_key(raw_ticket)
        if ticket in seen or ticket in history_set:
            continue
        seen.add(ticket)
        output.append(
            Candidate(
                ticket=ticket,
                source=source,
                model_id=genome.id,
                model_score=float(getattr(genome, "score", 0.0)),
                source_model=source_model,
                method=method,
                support=support,
                individual_score=candidate_score(
                    ticket,
                    rank=rank,
                    history=history,
                    latest=latest,
                    history_overlap_limit=history_overlap_limit,
                    diversity_weight=diversity_weight,
                    first_prize_weight=first_prize_weight,
                ),
                raw_rank=rank,
                created_at=created_at,
            )
        )
    output.sort(key=lambda candidate: candidate.individual_score, reverse=True)
    return output[:candidate_count]


def role_allocation(strategy_path: str, candidate_count: int) -> List[Tuple[str, str, int]]:
    requested = load_role_strategy(strategy_path, max(5, min(50, candidate_count)))
    counts = Counter(role for role, _label in requested)
    labels = {role: label for role, label in requested}
    if not counts:
        counts.update(role for role, _label in ROLE_ORDER)
        labels.update(dict(ROLE_ORDER))
    total = sum(counts.values()) or 1
    allocation: List[Tuple[str, str, int]] = []
    assigned = 0
    for index, (role, fallback_label) in enumerate(ROLE_ORDER):
        if index == len(ROLE_ORDER) - 1:
            count = max(1, candidate_count - assigned)
        else:
            count = max(1, int(round(candidate_count * counts.get(role, 0) / total)))
            assigned += count
        allocation.append((role, labels.get(role, fallback_label), count))
    return allocation


def regime_candidate_pool(
    *,
    genome: Genome,
    source_model: str,
    draws: Sequence[Draw],
    history: Sequence[Ticket],
    strategy_path: str,
    candidate_count: int,
    history_overlap_limit: int,
    diversity_weight: float,
    first_prize_weight: float,
    created_at: str,
) -> List[Candidate]:
    latest = draws[-1]
    history_set = set(history)
    output: List[Candidate] = []
    seen: Set[Ticket] = set()
    global_rank = 0
    for role, label, allocated in role_allocation(strategy_path, candidate_count):
        role_genome = clone_genome_for_role(genome, role)
        # Generate a modest oversample once per role, rather than invoking the
        # five-role selector hundreds of times.
        raw_count = max(40, int(math.ceil(allocated * 1.8)))
        try:
            raw_tickets = generate_tickets(draws, role_genome, raw_count)
        except Exception:
            raw_tickets = generate_tickets(draws, genome, raw_count)
        ranked = sorted(
            ((ticket_key(ticket), role_ticket_score(ticket, role, latest)) for ticket in raw_tickets),
            key=lambda item: item[1],
            reverse=True,
        )
        role_added = 0
        for ticket, role_score in ranked:
            if ticket in seen or ticket in history_set:
                continue
            seen.add(ticket)
            score = candidate_score(
                ticket,
                rank=global_rank,
                history=history,
                latest=latest,
                history_overlap_limit=history_overlap_limit,
                diversity_weight=diversity_weight,
                first_prize_weight=first_prize_weight,
                base=98.0,
                extra=float(role_score) * 0.45,
            )
            output.append(
                Candidate(
                    ticket=ticket,
                    source="regime",
                    model_id=f"{genome.id}:{role}",
                    model_score=float(getattr(genome, "score", 0.0)),
                    source_model=source_model,
                    method=f"portfolio_regime_{role}",
                    support=f"Regime役割={label} / 5口セット一括最適化",
                    individual_score=score,
                    raw_rank=global_rank,
                    created_at=created_at,
                )
            )
            global_rank += 1
            role_added += 1
            if role_added >= allocated:
                break
    output.sort(key=lambda candidate: candidate.individual_score, reverse=True)
    return output[:candidate_count]


def _pairs(ticket: Ticket) -> FrozenSet[Pair]:
    return frozenset(combinations(ticket, 2))


def _triples(ticket: Ticket) -> FrozenSet[Triple]:
    return frozenset(combinations(ticket, 3))


def select_portfolio(
    pools: Dict[str, Sequence[Candidate]],
    quotas: Dict[str, int],
    *,
    max_number_usage: int,
    max_pair_overlap: int,
    beam_width: int,
    candidates_per_step: int,
) -> Tuple[List[Candidate], Dict[str, object]]:
    source_sequence: List[str] = []
    active = [(source, count) for source, count in quotas.items() if count > 0]
    for source, count in sorted(active, key=lambda item: (len(pools.get(item[0], [])), item[0])):
        source_sequence.extend([source] * count)
    if not source_sequence:
        raise SystemExit("portfolio quotas selected zero tickets")

    states = [BeamState(0.0, tuple(), Counter(), frozenset(), frozenset())]
    for position, source in enumerate(source_sequence, start=1):
        candidates = list(pools.get(source, []))[:candidates_per_step]
        if not candidates:
            raise SystemExit(f"no candidates available for required source: {source}")
        expanded: List[BeamState] = []
        for state in states:
            existing = [candidate.ticket for candidate in state.selected]
            for candidate in candidates:
                if candidate.ticket in existing:
                    continue
                overlaps = [overlap(candidate.ticket, ticket) for ticket in existing]
                if overlaps and max(overlaps) > max_pair_overlap:
                    continue
                next_usage = state.usage.copy()
                unique_new = 0
                core_penalty = 0.0
                valid = True
                for number in candidate.ticket:
                    if next_usage[number] == 0:
                        unique_new += 1
                    next_usage[number] += 1
                    if next_usage[number] > max_number_usage:
                        valid = False
                        break
                    if next_usage[number] >= 3:
                        core_penalty += (next_usage[number] - 2) * 2.5
                if not valid:
                    continue
                pairs = _pairs(candidate.ticket)
                triples = _triples(candidate.ticket)
                increment = candidate.individual_score
                increment += unique_new * 2.2
                increment += len(pairs - state.pair_coverage) * 0.08
                increment += len(triples - state.triple_coverage) * 0.02
                increment -= sum(value * value for value in overlaps) * 1.4
                increment -= core_penalty
                expanded.append(
                    BeamState(
                        score=state.score + increment,
                        selected=state.selected + (candidate,),
                        usage=next_usage,
                        pair_coverage=state.pair_coverage | pairs,
                        triple_coverage=state.triple_coverage | triples,
                    )
                )
        if not expanded:
            raise SystemExit(f"portfolio search exhausted at position={position} source={source}")
        expanded.sort(key=lambda state: state.score, reverse=True)
        states = expanded[:max(1, beam_width)]

    best = max(states, key=lambda state: state.score)
    selected = sorted(best.selected, key=lambda candidate: candidate.individual_score, reverse=True)
    tickets = [candidate.ticket for candidate in selected]
    overlaps = [overlap(left, right) for left, right in combinations(tickets, 2)]
    summary: Dict[str, object] = {
        "objective_score": round(best.score, 6),
        "source_quotas": quotas,
        "source_counts": dict(Counter(candidate.source for candidate in selected)),
        "candidate_pool_counts": {source: len(pool) for source, pool in pools.items()},
        "unique_number_count": len({number for ticket in tickets for number in ticket}),
        "max_number_usage": max(best.usage.values()) if best.usage else 0,
        "number_usage": {str(number): count for number, count in sorted(best.usage.items())},
        "max_pair_overlap": max(overlaps) if overlaps else 0,
        "average_pair_overlap": round(sum(overlaps) / len(overlaps), 3) if overlaps else 0.0,
        "pair_coverage_count": len(best.pair_coverage),
        "triple_coverage_count": len(best.triple_coverage),
        "post_selection_number_replacements": 0,
    }
    return list(selected), summary


def build_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Genome, str, Dict[str, object]]:
    draws = load_draws(args.csv)
    if not draws:
        raise SystemExit(f"no draws loaded: {args.csv}")
    latest = draws[-1]
    created_at = now_iso()
    history = load_history_tickets(args.prediction_history)

    full_item = load_required_model(args.full_model)
    full_genome: Genome = full_item["genome"]  # type: ignore[assignment]
    recent_item, recent_fallback = load_optional_model(args.recent_model, full_item)
    recent_genome: Genome = recent_item["genome"]  # type: ignore[assignment]
    super_item, super_fallback = load_optional_model(args.super_recent_model, recent_item)
    super_genome: Genome = super_item["genome"]  # type: ignore[assignment]

    super_independent = bool(not super_fallback and super_genome.id != recent_genome.id)
    effective_super_count = args.super_recent_count
    if args.require_super_independent and not super_independent:
        effective_super_count = 0
    regime_count = args.purchase_count - args.full_count - args.recent_count - effective_super_count
    if regime_count < 0:
        raise SystemExit("full/recent/super counts exceed purchase count")
    quotas = {"full": args.full_count, "recent": args.recent_count, "super": effective_super_count, "regime": regime_count}

    pools: Dict[str, Sequence[Candidate]] = {
        "full": model_candidate_pool(
            genome=full_genome,
            source="full",
            source_model=str(full_item.get("path") or args.full_model),
            method="portfolio_full_period",
            support="全期間型候補 / 5口セット一括最適化",
            draws=draws,
            history=history,
            candidate_count=args.candidates_per_source,
            history_overlap_limit=args.history_overlap_limit,
            diversity_weight=args.diversity_weight,
            first_prize_weight=args.first_prize_weight,
            created_at=created_at,
        ),
        "recent": model_candidate_pool(
            genome=recent_genome,
            source="recent",
            source_model=str(recent_item.get("path") or args.recent_model),
            method="portfolio_recent_era",
            support="Recent Era候補 / 5口セット一括最適化" if not recent_fallback else "Recent fallback候補",
            draws=draws,
            history=history,
            candidate_count=args.candidates_per_source,
            history_overlap_limit=args.history_overlap_limit,
            diversity_weight=args.diversity_weight * 1.05,
            first_prize_weight=args.first_prize_weight,
            created_at=created_at,
        ),
        "super": model_candidate_pool(
            genome=super_genome,
            source="super",
            source_model=str(super_item.get("path") or args.super_recent_model),
            method="portfolio_super_recent",
            support="Super Recent独立候補 / 5口セット一括最適化",
            draws=draws,
            history=history,
            candidate_count=args.candidates_per_source,
            history_overlap_limit=args.history_overlap_limit,
            diversity_weight=args.diversity_weight * 1.12,
            first_prize_weight=args.first_prize_weight * 1.15,
            created_at=created_at,
        ) if effective_super_count > 0 else [],
        "regime": regime_candidate_pool(
            genome=full_genome,
            source_model=str(full_item.get("path") or args.full_model),
            draws=draws,
            history=history,
            strategy_path=args.regime_strategy,
            candidate_count=args.candidates_per_source,
            history_overlap_limit=args.history_overlap_limit,
            diversity_weight=args.diversity_weight * 1.10,
            first_prize_weight=args.first_prize_weight,
            created_at=created_at,
        ),
    }

    selected, portfolio_summary = select_portfolio(
        pools,
        quotas,
        max_number_usage=args.max_number_usage,
        max_pair_overlap=args.strict_overlap_limit,
        beam_width=args.beam_width,
        candidates_per_step=args.candidates_per_step,
    )
    rows: List[Dict[str, object]] = []
    for rank, candidate in enumerate(selected, start=1):
        rows.append(
            {
                "confidence_rank": rank,
                "base_latest_draw_no": latest.draw_no,
                "base_latest_date": latest.date,
                "prediction_draw_no": latest.draw_no + 1,
                "combo_index": rank,
                "numbers": fmt_ticket(candidate.ticket),
                "model_id": candidate.model_id,
                "model_score": round(candidate.model_score, 6),
                "source_model": candidate.source_model,
                "prediction_method": candidate.method,
                "ensemble_score": round(candidate.individual_score, 6),
                "support_models": candidate.support + " / post-selection replacement=0",
                "created_at": candidate.created_at,
            }
        )
    portfolio_summary.update(
        {
            "created_at": created_at,
            "prediction_draw_no": latest.draw_no + 1,
            "super_recent_independent": super_independent,
            "effective_super_recent_count": effective_super_count,
            "history_ticket_count": len(history),
            "constraints": {
                "max_number_usage": args.max_number_usage,
                "max_pair_overlap": args.strict_overlap_limit,
                "history_overlap_limit": args.history_overlap_limit,
                "exact_history_duplicates_allowed": False,
            },
        }
    )
    return rows, full_genome, str(full_item.get("path") or args.full_model), portfolio_summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build an optimized five-ticket LOTO7 portfolio.")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--full-model", default="loto7_best_model.json")
    parser.add_argument("--recent-model", default="outputs/recent_era/recent_era_best_model.json")
    parser.add_argument("--super-recent-model", default="outputs/super_recent/super_recent_best_model.json")
    parser.add_argument("--regime-strategy", default="outputs/role_ensemble/regime_strategy.json")
    parser.add_argument("--prediction-history", default="outputs/evolution_prediction_history.csv")
    parser.add_argument("--prediction", default="outputs/evolution_best_prediction.csv")
    parser.add_argument("--prediction-report", default="outputs/holdout/latest_prediction_report.txt")
    parser.add_argument("--portfolio-summary", default="outputs/holdout/latest_portfolio_summary.json")
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--full-count", type=int, default=2)
    parser.add_argument("--recent-count", type=int, default=1)
    parser.add_argument("--super-recent-count", type=int, default=1)
    parser.add_argument("--require-super-independent", action="store_true", default=True)
    parser.add_argument("--allow-super-fallback", dest="require_super_independent", action="store_false")
    parser.add_argument("--overlap-limit", type=int, default=4)
    parser.add_argument("--strict-overlap-limit", type=int, default=None)
    parser.add_argument("--max-number-usage", type=int, default=4)
    parser.add_argument("--history-overlap-limit", type=int, default=6)
    parser.add_argument("--diversity-weight", type=float, default=1.0)
    parser.add_argument("--first-prize-weight", type=float, default=1.0)
    parser.add_argument("--candidates-per-source", type=int, default=240)
    parser.add_argument("--candidates-per-step", type=int, default=100)
    parser.add_argument("--beam-width", type=int, default=320)
    args = parser.parse_args(argv)

    if args.strict_overlap_limit is None:
        args.strict_overlap_limit = args.overlap_limit
    if args.purchase_count <= 0 or min(args.full_count, args.recent_count, args.super_recent_count) < 0:
        raise SystemExit("invalid purchase/source counts")

    rows, full_genome, source_model, portfolio_summary = build_rows(args)
    write_prediction(args.prediction, rows)
    write_prediction_report(
        args.prediction_report,
        rows,
        full_genome,
        source_model,
        model_count=len({str(row.get("model_id")) for row in rows}),
        min_models=1,
        selection_reason="候補全体から5口セットをビームサーチ最適化 / 選択後の数字置換なし",
        prediction_mode="optimized_five_ticket_portfolio",
        role_strategy_path=args.regime_strategy,
    )
    summary_path = Path(args.portfolio_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(portfolio_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(portfolio_summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
