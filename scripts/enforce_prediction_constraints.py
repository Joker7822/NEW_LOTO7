#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-process LOTO7 prediction CSV and enforce cross-ticket constraints."""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

Ticket = Tuple[int, ...]


def parse_ticket(value: object) -> Ticket:
    nums = tuple(sorted(int(x) for x in re.findall(r"\d+", str(value or ""))[:7]))
    if len(nums) != 7 or len(set(nums)) != 7 or any(n < 1 or n > 37 for n in nums):
        raise SystemExit(f"invalid ticket: {value}")
    return nums


def fmt(ticket: Sequence[int]) -> str:
    return " ".join(f"{n:02d}" for n in sorted(ticket))


def usage(tickets: Sequence[Ticket]) -> Counter[int]:
    out: Counter[int] = Counter()
    for ticket in tickets:
        out.update(ticket)
    return out


def overlap(a: Ticket, b: Ticket) -> int:
    return len(set(a) & set(b))


def valid_candidate(candidate: Ticket, index: int, tickets: Sequence[Ticket], max_usage: int, max_overlap: int) -> bool:
    others = [ticket for i, ticket in enumerate(tickets) if i != index]
    if candidate in others:
        return False
    if any(overlap(candidate, other) > max_overlap for other in others):
        return False
    counts = usage(others)
    return all(counts[n] + 1 <= max_usage for n in candidate)


def balance_score(ticket: Ticket, counts: Counter[int]) -> float:
    total = sum(ticket)
    odd = sum(1 for n in ticket if n % 2)
    span = ticket[-1] - ticket[0]
    bands = len({0 if n <= 9 else 1 if n <= 19 else 2 if n <= 29 else 3 for n in ticket})
    return (
        bands * 4.0
        + (4.0 if 105 <= total <= 175 else -abs(total - 140) * 0.05)
        + (3.0 if 3 <= odd <= 4 else -2.0)
        + (2.0 if span >= 24 else -2.0)
        - sum(counts[n] * 3.0 for n in ticket)
    )


def repair(tickets: List[Ticket], max_usage: int, max_overlap: int) -> List[Tuple[int, int, int]]:
    changes: List[Tuple[int, int, int]] = []
    for _ in range(100):
        counts = usage(tickets)
        offenders = [n for n, count in counts.items() if count > max_usage]
        if not offenders:
            return changes
        offender = max(offenders, key=lambda n: (counts[n], n))
        repaired = False
        for index in range(len(tickets) - 1, -1, -1):
            ticket = tickets[index]
            if offender not in ticket:
                continue
            candidates: List[Tuple[float, Ticket, int]] = []
            for replacement in range(1, 38):
                if replacement in ticket:
                    continue
                candidate = tuple(sorted(replacement if n == offender else n for n in ticket))
                if not valid_candidate(candidate, index, tickets, max_usage, max_overlap):
                    continue
                candidates.append((balance_score(candidate, counts), candidate, replacement))
            if not candidates:
                continue
            _score, candidate, replacement = max(candidates, key=lambda item: item[0])
            tickets[index] = candidate
            changes.append((index, offender, replacement))
            repaired = True
            break
        if not repaired:
            raise SystemExit(f"cannot repair global number usage for number {offender}")
    raise SystemExit("constraint repair did not converge")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", default="outputs/evolution_best_prediction.csv")
    parser.add_argument("--report", default="outputs/holdout/latest_prediction_report.txt")
    parser.add_argument("--max-number-usage", type=int, default=4)
    parser.add_argument("--max-pair-overlap", type=int, default=4)
    args = parser.parse_args()

    path = Path(args.prediction)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows: List[Dict[str, str]] = [dict(row) for row in reader]
    if not rows:
        raise SystemExit("prediction CSV is empty")

    tickets = [parse_ticket(row.get("numbers")) for row in rows]
    changes = repair(tickets, args.max_number_usage, args.max_pair_overlap)

    for index, row in enumerate(rows):
        row["numbers"] = fmt(tickets[index])
    for index, old, new in changes:
        method = str(rows[index].get("prediction_method") or "")
        rows[index]["prediction_method"] = f"{method}_usage_guard" if method else "usage_guard"
        support = str(rows[index].get("support_models") or "")
        rows[index]["support_models"] = f"{support} / global-number-usage<={args.max_number_usage} ({old:02d}->{new:02d})"

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    final_counts = usage(tickets)
    violations = {n: count for n, count in final_counts.items() if count > args.max_number_usage}
    if violations:
        raise SystemExit(f"usage violations remain: {violations}")
    for i, ticket in enumerate(tickets):
        if any(overlap(ticket, other) > args.max_pair_overlap for other in tickets[:i]):
            raise SystemExit("pair overlap violation remains")

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("a", encoding="utf-8") as stream:
        stream.write("\n[Global Number Usage Guard]\n")
        stream.write(f"max_number_usage: {args.max_number_usage}\n")
        stream.write(f"max_pair_overlap: {args.max_pair_overlap}\n")
        stream.write(f"changes: {len(changes)}\n")
        for index, old, new in changes:
            stream.write(f"- ticket {index + 1}: {old:02d} -> {new:02d}\n")
        stream.write("final_usage: " + ", ".join(f"{n:02d}={count}" for n, count in sorted(final_counts.items())) + "\n")

    print(f"[OK] enforced prediction constraints; changes={len(changes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
