#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finalize and verify LOTO7 prediction outputs.

The portfolio builder now enforces number usage and pair overlap before a ticket
is selected.  Therefore this script is verification-only by default: a
constraint violation fails loudly instead of silently changing a selected
model ticket.  ``--repair`` remains available only for legacy/manual files.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

Ticket = Tuple[int, ...]


def parse_ticket(value: object) -> Ticket:
    numbers = tuple(sorted(int(item) for item in re.findall(r"\d+", str(value or ""))[:7]))
    if len(numbers) != 7 or len(set(numbers)) != 7 or any(number < 1 or number > 37 for number in numbers):
        raise SystemExit(f"invalid ticket: {value}")
    return numbers


def fmt(ticket: Sequence[int]) -> str:
    return " ".join(f"{number:02d}" for number in sorted(ticket))


def usage(tickets: Sequence[Ticket]) -> Counter[int]:
    output: Counter[int] = Counter()
    for ticket in tickets:
        output.update(ticket)
    return output


def overlap(left: Ticket, right: Ticket) -> int:
    return len(set(left) & set(right))


def row_rank(row: Dict[str, str], fallback: int) -> int:
    for key in ("confidence_rank", "combo_index"):
        try:
            value = int(float(str(row.get(key) or "").strip()))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return fallback


def relative_confidence(index: int) -> float:
    """Shared relative ordering score; it is not a winning probability."""
    return max(0.50, 0.95 - index * 0.05)


def valid_candidate(candidate: Ticket, index: int, tickets: Sequence[Ticket], max_usage: int, max_overlap: int) -> bool:
    others = [ticket for position, ticket in enumerate(tickets) if position != index]
    if candidate in others or any(overlap(candidate, other) > max_overlap for other in others):
        return False
    counts = usage(others)
    return all(counts[number] + 1 <= max_usage for number in candidate)


def balance_score(ticket: Ticket, counts: Counter[int]) -> float:
    total = sum(ticket)
    odd = sum(1 for number in ticket if number % 2)
    span = ticket[-1] - ticket[0]
    bands = len({0 if number <= 9 else 1 if number <= 19 else 2 if number <= 29 else 3 for number in ticket})
    consecutive = sum(1 for left, right in zip(ticket, ticket[1:]) if right == left + 1)
    return (
        bands * 4.0
        + (4.0 if 105 <= total <= 175 else -abs(total - 140) * 0.05)
        + (3.0 if 3 <= odd <= 4 else -2.0)
        + (2.0 if span >= 24 else -2.0)
        - max(0, consecutive - 2) * 1.5
        - sum(counts[number] * 3.0 for number in ticket)
    )


def repair(tickets: List[Ticket], max_usage: int, max_overlap: int) -> List[Tuple[int, int, int]]:
    """Legacy-only repair path. Production workflows must not use it."""
    changes: List[Tuple[int, int, int]] = []
    for _ in range(100):
        counts = usage(tickets)
        offenders = [number for number, count in counts.items() if count > max_usage]
        pair_violation = next(
            (
                (right_index, left_index)
                for right_index, ticket in enumerate(tickets)
                for left_index, other in enumerate(tickets[:right_index])
                if overlap(ticket, other) > max_overlap
            ),
            None,
        )
        if not offenders and pair_violation is None:
            return changes
        offender = max(offenders, key=lambda number: (counts[number], number)) if offenders else None
        target_indices = list(range(len(tickets) - 1, -1, -1))
        if pair_violation is not None:
            target_indices = [pair_violation[0]] + [index for index in target_indices if index != pair_violation[0]]
        repaired = False
        for index in target_indices:
            ticket = tickets[index]
            removable = [offender] if offender is not None and offender in ticket else list(ticket)
            for old in removable:
                candidates: List[Tuple[float, Ticket, int]] = []
                for replacement in range(1, 38):
                    if replacement in ticket:
                        continue
                    candidate = tuple(sorted(replacement if number == old else number for number in ticket))
                    if valid_candidate(candidate, index, tickets, max_usage, max_overlap):
                        candidates.append((balance_score(candidate, counts), candidate, replacement))
                if candidates:
                    _score, candidate, replacement = max(candidates, key=lambda item: item[0])
                    tickets[index] = candidate
                    changes.append((index, int(old), replacement))
                    repaired = True
                    break
            if repaired:
                break
        if not repaired:
            raise SystemExit("cannot repair legacy prediction constraints")
    raise SystemExit("legacy constraint repair did not converge")


def normalize_rows(rows: List[Dict[str, str]], tickets: Sequence[Ticket], changes: Sequence[Tuple[int, int, int]]) -> None:
    changed: Dict[int, List[Tuple[int, int]]] = {}
    for index, old, new in changes:
        changed.setdefault(index, []).append((old, new))
    for index, row in enumerate(rows):
        row["confidence_rank"] = str(index + 1)
        row["combo_index"] = str(index + 1)
        row["numbers"] = fmt(tickets[index])
        row["ensemble_score"] = f"{relative_confidence(index):.2f}"
        replacements = changed.get(index, [])
        if not replacements:
            continue
        method = str(row.get("prediction_method") or "")
        if "usage_guard" not in method:
            row["prediction_method"] = f"{method}_usage_guard" if method else "usage_guard"
        note = ", ".join(f"{old:02d}->{new:02d}" for old, new in replacements)
        support = str(row.get("support_models") or "")
        row["support_models"] = f"{support} / legacy-repair ({note})" if support else f"legacy-repair ({note})"


def ensure_fieldnames(fieldnames: List[str]) -> List[str]:
    required = [
        "confidence_rank", "base_latest_draw_no", "base_latest_date", "prediction_draw_no",
        "combo_index", "numbers", "model_id", "model_score", "source_model",
        "prediction_method", "ensemble_score", "support_models", "created_at",
    ]
    output = list(fieldnames)
    for key in required:
        if key not in output:
            output.append(key)
    return output


def write_prediction_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_final_report(
    path: Path,
    rows: Sequence[Dict[str, str]],
    tickets: Sequence[Ticket],
    changes: Sequence[Tuple[int, int, int]],
    final_counts: Counter[int],
    max_number_usage: int,
    max_pair_overlap: int,
) -> None:
    first = rows[0]
    unique_sources = sorted({str(row.get("source_model") or "").strip() for row in rows if str(row.get("source_model") or "").strip()})
    super_recent_independent = any("super_recent" in str(row.get("prediction_method") or "").lower() for row in rows)
    selection_integrity = "legacy_repaired" if changes else "original_model_candidates"
    lines = [
        "LOTO7 Latest Prediction Report",
        "===============================",
        "",
        f"作成日時(UTC): {first.get('created_at', '')}",
        f"基準最新回: 第{first.get('base_latest_draw_no', '')}回",
        f"基準最新抽せん日: {first.get('base_latest_date', '')}",
        f"予測対象回: 第{first.get('prediction_draw_no', '')}回",
        "",
        "[採用モデル]",
        f"先頭モデルID: {first.get('model_id', '')}",
        f"先頭モデルスコア: {first.get('model_score', '')}",
        f"使用元モデル数: {len(unique_sources)}",
        "予測方式: optimized_five_ticket_portfolio",
        f"selection_integrity: {selection_integrity}",
        "採用基準: 全候補から5口セットを一括最適化 / 選択後の数字置換なし",
        "",
        f"[最新予測 {len(rows)}口: 正規化相対スコア順]",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"{index}位 / {index}口目: {row.get('numbers', '')} / "
            f"相対スコア={row.get('ensemble_score', '')} / 方式={row.get('prediction_method', '')}"
        )
    lines.extend(
        [
            "",
            "[相対スコアの意味]",
            "0.95から順に統一した5口内の相対順位スコアです。",
            "当せん確率や期待収益率を表す数値ではありません。",
            "",
            "[Portfolio Constraint Verification]",
            f"max_number_usage: {max_number_usage}",
            f"max_pair_overlap: {max_pair_overlap}",
            f"super_recent_independent: {str(super_recent_independent).lower()}",
            f"post_selection_changes: {len(changes)}",
            "final_usage: " + ", ".join(f"{number:02d}={count}" for number, count in sorted(final_counts.items())),
        ]
    )
    if not super_recent_independent:
        lines.append("super_recent_note: Recent Eraと異なるモデルIDが採用されるまで独立枠は使用しません。")
    for index, old, new in changes:
        lines.append(f"- legacy repair ticket {index + 1}: {old:02d} -> {new:02d}")
    lines.extend(
        [
            "",
            "[出力ファイル]",
            "CSV: outputs/evolution_best_prediction.csv",
            "TXT: outputs/holdout/latest_prediction_report.txt",
            "",
            "注意: 宝くじはランダム性が高く、この予測は当せんや利益を保証するものではありません。",
        ]
    )
    text = "\n".join(lines) + "\n"
    for ticket in tickets:
        if fmt(ticket) not in text:
            raise SystemExit(f"report synchronization failed for ticket: {fmt(ticket)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", default="outputs/evolution_best_prediction.csv")
    parser.add_argument("--report", default="outputs/holdout/latest_prediction_report.txt")
    parser.add_argument("--max-number-usage", type=int, default=4)
    parser.add_argument("--max-pair-overlap", type=int, default=4)
    parser.add_argument("--repair", action="store_true", help="Legacy-only: modify tickets to repair violations.")
    args = parser.parse_args()

    path = Path(args.prediction)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = ensure_fieldnames(list(reader.fieldnames or []))
        rows: List[Dict[str, str]] = [dict(row) for row in reader]
    if not rows:
        raise SystemExit("prediction CSV is empty")

    rows.sort(key=lambda row: (row_rank(row, len(rows) + 1), str(row.get("numbers") or "")))
    tickets = [parse_ticket(row.get("numbers")) for row in rows]
    changes: List[Tuple[int, int, int]] = []
    counts = usage(tickets)
    usage_violations = {number: count for number, count in counts.items() if count > args.max_number_usage}
    overlap_violations = [
        (left_index, right_index, overlap(tickets[left_index], tickets[right_index]))
        for right_index in range(len(tickets))
        for left_index in range(right_index)
        if overlap(tickets[left_index], tickets[right_index]) > args.max_pair_overlap
    ]
    if usage_violations or overlap_violations:
        if not args.repair:
            raise SystemExit(
                f"optimized portfolio constraint violation; usage={usage_violations} overlap={overlap_violations}"
            )
        changes = repair(tickets, args.max_number_usage, args.max_pair_overlap)

    normalize_rows(rows, tickets, changes)
    final_counts = usage(tickets)
    if any(count > args.max_number_usage for count in final_counts.values()):
        raise SystemExit("usage violation remains")
    for right_index, ticket in enumerate(tickets):
        if any(overlap(ticket, other) > args.max_pair_overlap for other in tickets[:right_index]):
            raise SystemExit("pair overlap violation remains")

    write_prediction_csv(path, fieldnames, rows)
    write_final_report(Path(args.report), rows, tickets, changes, final_counts, args.max_number_usage, args.max_pair_overlap)
    print(
        f"[OK] verified optimized portfolio; changes={len(changes)} "
        f"scores={','.join(row.get('ensemble_score', '') for row in rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
