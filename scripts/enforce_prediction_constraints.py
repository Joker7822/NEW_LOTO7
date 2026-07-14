#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finalize LOTO7 prediction outputs and enforce cross-ticket constraints.

This is the final normalization stage for latest prediction outputs.
It performs all of the following in one place:

- Enforce maximum global number usage across the five tickets.
- Enforce maximum pairwise ticket overlap.
- Normalize every row to one shared relative-confidence scale.
- Rewrite the TXT report from the finalized CSV rows instead of appending to a
  stale pre-repair report.

The normalized confidence is a relative ordering score, not a probability of
winning.
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
    """Return a shared 0-1 relative score: 0.95, 0.90, ...

    The score expresses ordering only and is deliberately not calibrated as a
    winning probability.
    """
    return max(0.50, 0.95 - index * 0.05)


def valid_candidate(
    candidate: Ticket,
    index: int,
    tickets: Sequence[Ticket],
    max_usage: int,
    max_overlap: int,
) -> bool:
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
    consecutive_pairs = sum(1 for a, b in zip(ticket, ticket[1:]) if b == a + 1)
    return (
        bands * 4.0
        + (4.0 if 105 <= total <= 175 else -abs(total - 140) * 0.05)
        + (3.0 if 3 <= odd <= 4 else -2.0)
        + (2.0 if span >= 24 else -2.0)
        - max(0, consecutive_pairs - 2) * 1.5
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


def normalize_rows(rows: List[Dict[str, str]], tickets: Sequence[Ticket], changes: Sequence[Tuple[int, int, int]]) -> None:
    changed_by_index: Dict[int, List[Tuple[int, int]]] = {}
    for index, old, new in changes:
        changed_by_index.setdefault(index, []).append((old, new))

    for index, row in enumerate(rows):
        rank = index + 1
        row["confidence_rank"] = str(rank)
        row["combo_index"] = str(rank)
        row["numbers"] = fmt(tickets[index])
        row["ensemble_score"] = f"{relative_confidence(index):.2f}"

        replacements = changed_by_index.get(index, [])
        if not replacements:
            continue

        method = str(row.get("prediction_method") or "")
        if "usage_guard" not in method:
            row["prediction_method"] = f"{method}_usage_guard" if method else "usage_guard"

        support = str(row.get("support_models") or "")
        notes = ", ".join(f"{old:02d}->{new:02d}" for old, new in replacements)
        marker = f"global-number-usage ({notes})"
        if marker not in support:
            row["support_models"] = f"{support} / {marker}" if support else marker


def ensure_fieldnames(fieldnames: List[str]) -> List[str]:
    required = [
        "confidence_rank",
        "base_latest_draw_no",
        "base_latest_date",
        "prediction_draw_no",
        "combo_index",
        "numbers",
        "model_id",
        "model_score",
        "source_model",
        "prediction_method",
        "ensemble_score",
        "support_models",
        "created_at",
    ]
    output = list(fieldnames)
    for key in required:
        if key not in output:
            output.append(key)
    return output


def write_prediction_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


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
    super_recent_independent = any(
        "dual_super_recent" in str(row.get("prediction_method") or "").lower()
        for row in rows
    )

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
        "予測方式: adaptive_multi_model_finalized",
        "採用基準: 全期間型 + Recent Era + Super Recent独立時のみ + Regime / 重複削減・履歴抑制・多様性補正",
        "",
        f"[最新予測 {len(rows)}口: 正規化相対スコア順]",
    ]

    for index, row in enumerate(rows, start=1):
        lines.append(
            f"{index}位 / {index}口目: {row.get('numbers', '')} / "
            f"相対スコア={row.get('ensemble_score', '')} / "
            f"方式={row.get('prediction_method', '')}"
        )

    lines.extend(
        [
            "",
            "[相対スコアの意味]",
            "0.95から順に統一した5口内の相対順位スコアです。",
            "当せん確率や期待収益率を表す数値ではありません。",
            "CSVのensemble_scoreと予測履歴の信頼度には、この共通尺度を使用します。",
            "",
            "[Global Number Usage Guard]",
            f"max_number_usage: {max_number_usage}",
            f"max_pair_overlap: {max_pair_overlap}",
            f"super_recent_independent: {str(super_recent_independent).lower()}",
        ]
    )
    if not super_recent_independent:
        lines.append("super_recent_note: Recent Eraと異なるモデルIDが採用されるまでSuper Recent独立枠は使用しません。")
    lines.append(f"changes: {len(changes)}")
    for index, old, new in changes:
        lines.append(f"- ticket {index + 1}: {old:02d} -> {new:02d}")
    lines.append("final_usage: " + ", ".join(f"{n:02d}={count}" for n, count in sorted(final_counts.items())))
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
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


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
        fieldnames = ensure_fieldnames(list(reader.fieldnames or []))
        rows: List[Dict[str, str]] = [dict(row) for row in reader]
    if not rows:
        raise SystemExit("prediction CSV is empty")

    rows.sort(key=lambda row: (row_rank(row, len(rows) + 1), str(row.get("numbers") or "")))
    tickets = [parse_ticket(row.get("numbers")) for row in rows]
    changes = repair(tickets, args.max_number_usage, args.max_pair_overlap)
    normalize_rows(rows, tickets, changes)

    final_counts = usage(tickets)
    violations = {n: count for n, count in final_counts.items() if count > args.max_number_usage}
    if violations:
        raise SystemExit(f"usage violations remain: {violations}")
    for i, ticket in enumerate(tickets):
        if any(overlap(ticket, other) > args.max_pair_overlap for other in tickets[:i]):
            raise SystemExit("pair overlap violation remains")

    write_prediction_csv(path, fieldnames, rows)
    write_final_report(
        Path(args.report),
        rows,
        tickets,
        changes,
        final_counts,
        args.max_number_usage,
        args.max_pair_overlap,
    )

    print(
        f"[OK] finalized prediction outputs; changes={len(changes)} "
        f"scores={','.join(row.get('ensemble_score', '') for row in rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
