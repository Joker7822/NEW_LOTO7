#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loto7_logic_predictor.py

NEW_LOTO7 用 Loto7 予測・バックテスト完全版。

実装内容:
    - loto7.csv の過去実績だけで次回候補を生成
    - 予測CSVは loto7_predictions.csv に保存
    - 最新予測TXTは latest_loto7_prediction.txt に保存
    - 最大重複数を4以下に制限
    - 低・中・高番号帯の分散を強化
    - 10口のうち最低2〜3口を別戦略で生成
    - 「信頼度」表記を「スコア正規化値」に変更
    - バックテスト結果を固定3ファイルで出力
        1. loto7_backtest_summary.csv
        2. loto7_backtest_detail.csv
        3. loto7_backtest_report.txt

注意:
    ロト7は独立抽せんのため、的中保証は確認できません。
    このスクリプトは過去実績から候補を生成するロジックです。
    当せん金額は各回で変動するため、バックテスト収支は設定金額ベースの概算です。
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import re
import sys
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_CSV_URL = "loto7.csv"

DEFAULT_OUTPUT_CSV = "loto7_predictions.csv"
DEFAULT_LATEST_TXT = "latest_loto7_prediction.txt"

DEFAULT_BACKTEST_SUMMARY_CSV = "loto7_backtest_summary.csv"
DEFAULT_BACKTEST_DETAIL_CSV = "loto7_backtest_detail.csv"
DEFAULT_BACKTEST_REPORT_TXT = "loto7_backtest_report.txt"

NUM_MIN = 1
NUM_MAX = 37
PICK_SIZE = 7

DEFAULT_TICKETS = 10
DEFAULT_SAVE_COUNT = 10
DEFAULT_UNIT_COST = 300

# GitHub Actions 30分制限対策。
# 通常予測は pool_size=24 でも問題ないが、全件バックテストでは組合せ数が爆発するため、
# バックテスト時だけ候補プールを安全側に自動圧縮する。
DEFAULT_BACKTEST_POOL_CAP = 16

LOW_RANGE = range(1, 13)
MID_RANGE = range(13, 26)
HIGH_RANGE = range(26, 38)

# 実際の当せん金額ではなく、バックテスト用の設定値。
# 正確な回収率を出すには各回の等級別実当せん金額CSVが必要。
DEFAULT_PRIZE_TABLE = {
    1: 700_000_000,
    2: 7_300_000,
    3: 730_000,
    4: 9_100,
    5: 1_400,
    6: 1_000,
}


@dataclass(frozen=True)
class Draw:
    date: str
    main: Tuple[int, ...]
    bonus: Tuple[int, ...]
    draw_no: Optional[int] = None


@dataclass
class TicketScore:
    ticket: Tuple[int, ...]
    score: float
    detail: Dict[str, float]
    strategy: str = "CORE"


@dataclass(frozen=True)
class PrizeResult:
    main_matches: int
    bonus_matches: int
    grade: Optional[int]
    prize: int


def parse_numbers(value: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", str(value or "")))


def parse_draw_no(value: str) -> Optional[int]:
    nums = re.findall(r"\d+", str(value or ""))
    return int(nums[0]) if nums else None


def validate_main_numbers(nums: Sequence[int]) -> Tuple[int, ...]:
    if len(nums) != PICK_SIZE:
        raise ValueError(f"本数字は7個必要です: {nums}")
    if len(set(nums)) != PICK_SIZE:
        raise ValueError(f"本数字に重複があります: {nums}")
    for n in nums:
        if not (NUM_MIN <= n <= NUM_MAX):
            raise ValueError(f"数字が範囲外です: {n}")
    return tuple(sorted(nums))


def read_text_from_source(source: str) -> str:
    if source.startswith(("http://", "https://")):
        req = urllib.request.Request(
            source,
            headers={
                "User-Agent": "Mozilla/5.0 loto7-logic-predictor",
                "Accept": "text/csv,text/plain,*/*",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as res:
            return res.read().decode("utf-8-sig")

    with open(source, "r", encoding="utf-8-sig", newline="") as f:
        return f.read()


def load_draws(source: str = DEFAULT_CSV_URL) -> List[Draw]:
    text = read_text_from_source(source)
    reader = csv.reader(io.StringIO(text))
    draws: List[Draw] = []

    for row in reader:
        if not row or len(row) < 3:
            continue
        if "抽せん" in row[0] or "date" in row[0].lower():
            continue

        date = row[0].strip()
        main_raw = parse_numbers(row[1])
        bonus = tuple(sorted(n for n in parse_numbers(row[2]) if NUM_MIN <= n <= NUM_MAX))
        draw_no = parse_draw_no(row[3]) if len(row) >= 4 else None

        try:
            main = validate_main_numbers(main_raw)
        except ValueError:
            continue

        draws.append(Draw(date=date, main=main, bonus=bonus, draw_no=draw_no))

    draws.sort(key=lambda d: d.date)
    return draws


def format_ticket(ticket: Sequence[int], zero_pad: bool = True) -> str:
    if zero_pad:
        return ", ".join(f"{n:02d}" for n in sorted(ticket))
    return ", ".join(str(n) for n in sorted(ticket))


def recent_draws(draws: Sequence[Draw], window: int) -> Sequence[Draw]:
    return draws[-window:] if window > 0 else draws


def count_numbers(draws: Sequence[Draw], window: int) -> Counter:
    c: Counter = Counter()
    for d in recent_draws(draws, window):
        c.update(d.main)
    return c


def count_bonus_numbers(draws: Sequence[Draw], window: int) -> Counter:
    c: Counter = Counter()
    for d in recent_draws(draws, window):
        c.update(d.bonus)
    return c


def count_combinations(draws: Sequence[Draw], window: int, k: int) -> Counter:
    c: Counter = Counter()
    for d in recent_draws(draws, window):
        for comb in itertools.combinations(sorted(d.main), k):
            c[comb] += 1
    return c


def last_seen_gaps(draws: Sequence[Draw]) -> Dict[int, int]:
    last = {n: None for n in range(NUM_MIN, NUM_MAX + 1)}
    for idx, d in enumerate(draws):
        for n in d.main:
            last[n] = idx

    latest_idx = len(draws) - 1
    return {
        n: (len(draws) + 1 if idx is None else latest_idx - idx)
        for n, idx in last.items()
    }


def normalized(counter: Counter, n: int, max_value: Optional[float] = None) -> float:
    if max_value is None:
        max_value = max(counter.values()) if counter else 1.0
    if max_value <= 0:
        return 0.0
    return counter.get(n, 0) / max_value


def build_number_scores(draws: Sequence[Draw]) -> Dict[int, float]:
    c10 = count_numbers(draws, 10)
    c20 = count_numbers(draws, 20)
    c50 = count_numbers(draws, 50)
    c100 = count_numbers(draws, 100)
    bonus20 = count_bonus_numbers(draws, 20)
    gaps = last_seen_gaps(draws)

    max10 = max(c10.values()) if c10 else 1
    max20 = max(c20.values()) if c20 else 1
    max50 = max(c50.values()) if c50 else 1
    max100 = max(c100.values()) if c100 else 1
    max_bonus20 = max(bonus20.values()) if bonus20 else 1

    scores: Dict[int, float] = {}

    for n in range(NUM_MIN, NUM_MAX + 1):
        hot_score = (
            4.0 * normalized(c10, n, max10)
            + 2.5 * normalized(c20, n, max20)
            + 1.5 * normalized(c50, n, max50)
            + 0.8 * normalized(c100, n, max100)
        )

        bonus_score = 0.15 * normalized(bonus20, n, max_bonus20)

        gap = gaps[n]
        if gap == 0:
            gap_score = -0.15
        elif 3 <= gap <= 18:
            gap_score = 0.15 + min(gap, 18) / 100.0
        elif 19 <= gap <= 25:
            gap_score = 0.15
        else:
            gap_score = 0.0

        scores[n] = hot_score + bonus_score + gap_score

    return scores


def band_counts(ticket: Sequence[int]) -> Tuple[int, int, int]:
    low = sum(1 for n in ticket if n in LOW_RANGE)
    mid = sum(1 for n in ticket if n in MID_RANGE)
    high = sum(1 for n in ticket if n in HIGH_RANGE)
    return low, mid, high


def strategy_type(ticket: Sequence[int]) -> str:
    low, mid, high = band_counts(ticket)
    if high >= 3:
        return "HIGH_BIAS"
    if low >= 3:
        return "LOW_BIAS"
    if low >= 2 and mid >= 2 and high >= 2:
        return "BALANCED"
    return "CORE"


def max_consecutive_run(ticket: Sequence[int]) -> int:
    nums = sorted(ticket)
    best = cur = 1
    for a, b in zip(nums, nums[1:]):
        if b == a + 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def structure_penalty(ticket: Sequence[int], last_main: Sequence[int]) -> float:
    ticket = tuple(sorted(ticket))
    penalty = 0.0

    odd = sum(1 for n in ticket if n % 2 == 1)
    if odd in (3, 4):
        penalty += 0.0
    elif odd in (2, 5):
        penalty += 0.15
    else:
        penalty += 0.60

    low, mid, high = band_counts(ticket)

    # 低・中・高の分散を強化。
    # 基本形は low=2〜3, mid=2〜3, high=1〜3。
    if not (2 <= low <= 3):
        penalty += 0.35 * abs(low - 2.5)
    if not (2 <= mid <= 3):
        penalty += 0.35 * abs(mid - 2.5)
    if not (1 <= high <= 3):
        penalty += 0.30 * abs(high - 2.0)

    total = sum(ticket)
    if total < 110:
        penalty += (110 - total) / 20.0
    elif total > 180:
        penalty += (total - 180) / 20.0
    elif total < 120:
        penalty += (120 - total) / 60.0
    elif total > 170:
        penalty += (total - 170) / 60.0

    run = max_consecutive_run(ticket)
    if run >= 4:
        penalty += 1.00
    elif run == 3:
        penalty += 0.35

    repeat = len(set(ticket) & set(last_main))
    if repeat < 1:
        penalty += 0.70
    elif repeat == 1:
        penalty += 0.20
    elif 2 <= repeat <= 4:
        penalty += 0.0
    elif repeat == 5:
        penalty += 0.35
    else:
        penalty += 0.80

    return penalty


def ticket_score(
    ticket: Sequence[int],
    draws: Sequence[Draw],
    number_scores: Dict[int, float],
    pair10: Counter,
    pair20: Counter,
    pair50: Counter,
    triple20: Counter,
    triple50: Counter,
    strategy: str = "CORE",
) -> TicketScore:
    ticket = tuple(sorted(ticket))
    last_main = draws[-1].main

    single = sum(number_scores[n] for n in ticket)

    pair_score = 0.0
    for pair in itertools.combinations(ticket, 2):
        pair_score += 0.18 * pair10.get(pair, 0)
        pair_score += 0.10 * pair20.get(pair, 0)
        pair_score += 0.04 * pair50.get(pair, 0)

    triple_score = 0.0
    for tri in itertools.combinations(ticket, 3):
        triple_score += 0.12 * triple20.get(tri, 0)
        triple_score += 0.05 * triple50.get(tri, 0)

    penalty = structure_penalty(ticket, last_main)

    low, mid, high = band_counts(ticket)
    strategy_bonus = 0.0
    if strategy == "HIGH_BIAS" and high >= 3:
        strategy_bonus += 0.65
    elif strategy == "LOW_BIAS" and low >= 3:
        strategy_bonus += 0.65
    elif strategy == "BALANCED" and low >= 2 and mid >= 2 and high >= 2:
        strategy_bonus += 0.55
    elif strategy == "COLD_NUMBER":
        gaps = last_seen_gaps(draws)
        strategy_bonus += sum(min(gaps[n], 30) for n in ticket) / 180.0

    total_score = single + pair_score + triple_score + strategy_bonus - penalty

    return TicketScore(
        ticket=ticket,
        score=total_score,
        strategy=strategy,
        detail={
            "single": single,
            "pair": pair_score,
            "triple": triple_score,
            "strategy_bonus": strategy_bonus,
            "penalty": penalty,
            "sum": float(sum(ticket)),
            "odd": float(sum(1 for n in ticket if n % 2 == 1)),
            "low": float(low),
            "mid": float(mid),
            "high": float(high),
            "repeat_last": float(len(set(ticket) & set(last_main))),
        },
    )


def make_candidate_pool(draws: Sequence[Draw], pool_size: int = 24) -> List[int]:
    number_scores = build_number_scores(draws)
    pair20 = count_combinations(draws, 20, 2)
    pair50 = count_combinations(draws, 50, 2)
    tri20 = count_combinations(draws, 20, 3)
    gaps = last_seen_gaps(draws)

    pool = set()

    for n, _ in sorted(number_scores.items(), key=lambda x: (-x[1], x[0]))[:16]:
        pool.add(n)

    for pair, _ in pair20.most_common(12):
        pool.update(pair)

    for pair, _ in pair50.most_common(8):
        pool.update(pair)

    for tri, _ in tri20.most_common(8):
        pool.update(tri)

    # コールド数字も最低限候補へ入れる。
    for n, _ in sorted(gaps.items(), key=lambda x: (-x[1], x[0]))[:6]:
        pool.add(n)

    for n, _ in sorted(number_scores.items(), key=lambda x: (-x[1], x[0])):
        pool.add(n)
        if len(pool) >= pool_size:
            break

    return sorted(pool, key=lambda n: (-number_scores[n], n))[:pool_size]


def rank_tickets(
    draws: Sequence[Draw],
    pool_size: int = 24,
    max_rank: int = 1000,
    strategy: str = "CORE",
) -> List[TicketScore]:
    number_scores = build_number_scores(draws)
    pool = make_candidate_pool(draws, pool_size=pool_size)

    pair10 = count_combinations(draws, 10, 2)
    pair20 = count_combinations(draws, 20, 2)
    pair50 = count_combinations(draws, 50, 2)
    triple20 = count_combinations(draws, 20, 3)
    triple50 = count_combinations(draws, 50, 3)

    ranked: List[TicketScore] = []
    for comb in itertools.combinations(sorted(pool), PICK_SIZE):
        if strategy == "HIGH_BIAS" and band_counts(comb)[2] < 3:
            continue
        if strategy == "LOW_BIAS" and band_counts(comb)[0] < 3:
            continue
        if strategy == "BALANCED":
            low, mid, high = band_counts(comb)
            if not (low >= 2 and mid >= 2 and high >= 2):
                continue

        ranked.append(
            ticket_score(
                comb,
                draws,
                number_scores,
                pair10,
                pair20,
                pair50,
                triple20,
                triple50,
                strategy=strategy,
            )
        )

    ranked.sort(key=lambda x: (-x.score, x.ticket))
    return ranked[:max_rank]


def overlap_ok(ticket: Sequence[int], selected: Sequence[TicketScore], max_overlap: int) -> bool:
    s = set(ticket)
    for item in selected:
        if len(s & set(item.ticket)) > max_overlap:
            return False
    return True


def usage_ok(ticket: Sequence[int], usage: Counter, max_number_usage: int) -> bool:
    return all(usage[n] < max_number_usage for n in ticket)


def append_strategy_ticket(
    selected: List[TicketScore],
    ranked: Sequence[TicketScore],
    usage: Counter,
    max_overlap: int,
    max_number_usage: int,
) -> bool:
    for cand in ranked:
        if cand.ticket in {x.ticket for x in selected}:
            continue
        if not overlap_ok(cand.ticket, selected, max_overlap=max_overlap):
            continue
        if not usage_ok(cand.ticket, usage, max_number_usage=max_number_usage):
            continue
        selected.append(cand)
        usage.update(cand.ticket)
        return True
    return False


def select_diverse_tickets(
    ranked_core: Sequence[TicketScore],
    ranked_balanced: Sequence[TicketScore],
    ranked_high: Sequence[TicketScore],
    ranked_low: Sequence[TicketScore],
    ranked_cold: Sequence[TicketScore],
    num_tickets: int = DEFAULT_TICKETS,
    max_overlap: int = 4,
    max_number_usage: int = 3,
) -> List[TicketScore]:
    """
    多様性強化版:
        - 任意2口間の最大重複数を原則4以下
        - 各数字の使用回数を原則3以下
        - 10口の場合、最低3口を別戦略枠にする
    """
    selected: List[TicketScore] = []
    usage: Counter = Counter()

    strategy_ranked = [
        ranked_balanced,
        ranked_high,
        ranked_low,
        ranked_cold,
    ]

    # 10口なら先に3〜4口を戦略別に確保。
    if num_tickets >= 10:
        for ranked in strategy_ranked:
            if len(selected) >= 4:
                break
            append_strategy_ticket(
                selected,
                ranked,
                usage,
                max_overlap=max_overlap,
                max_number_usage=max_number_usage,
            )

    # 残りはコア順位から選抜。
    for cand in ranked_core:
        if len(selected) >= num_tickets:
            return selected[:num_tickets]

        if cand.ticket in {x.ticket for x in selected}:
            continue

        if not overlap_ok(cand.ticket, selected, max_overlap=max_overlap):
            continue

        if not usage_ok(cand.ticket, usage, max_number_usage=max_number_usage):
            continue

        selected.append(cand)
        usage.update(cand.ticket)

    # 厳格条件で足りない場合は、重複制限のみ維持して補完。
    for pool in [ranked_balanced, ranked_high, ranked_low, ranked_cold, ranked_core]:
        for cand in pool:
            if len(selected) >= num_tickets:
                return selected[:num_tickets]
            if cand.ticket in {x.ticket for x in selected}:
                continue
            if not overlap_ok(cand.ticket, selected, max_overlap=max_overlap):
                continue
            selected.append(cand)
            usage.update(cand.ticket)

    # それでも足りない場合のみ最終補完。
    for cand in ranked_core:
        if len(selected) >= num_tickets:
            break
        if cand.ticket not in {x.ticket for x in selected}:
            selected.append(cand)

    return selected[:num_tickets]


def predict(draws: Sequence[Draw], num_tickets: int = DEFAULT_TICKETS, pool_size: int = 24) -> List[TicketScore]:
    ranked_core = rank_tickets(draws, pool_size=pool_size, max_rank=1000, strategy="CORE")
    ranked_balanced = rank_tickets(draws, pool_size=pool_size, max_rank=500, strategy="BALANCED")
    ranked_high = rank_tickets(draws, pool_size=pool_size, max_rank=500, strategy="HIGH_BIAS")
    ranked_low = rank_tickets(draws, pool_size=pool_size, max_rank=500, strategy="LOW_BIAS")
    ranked_cold = rank_tickets(draws, pool_size=pool_size, max_rank=500, strategy="COLD_NUMBER")

    return select_diverse_tickets(
        ranked_core=ranked_core,
        ranked_balanced=ranked_balanced,
        ranked_high=ranked_high,
        ranked_low=ranked_low,
        ranked_cold=ranked_cold,
        num_tickets=num_tickets,
        max_overlap=4,
        max_number_usage=3,
    )


def score_normalized_values(ranked: Sequence[TicketScore]) -> List[float]:
    """
    旧「信頼度」ではなく、候補順位スコアの正規化値。
    実際の当選確率ではない。
    """
    if not ranked:
        return []
    scores = [x.score for x in ranked]
    max_score = max(scores)
    min_score = min(scores)

    if max_score == min_score:
        return [1.000 for _ in ranked]

    values = []
    for s in scores:
        value = (s - min_score) / (max_score - min_score)
        values.append(round(value, 3))
    return values


def prediction_csv_header(save_count: int = DEFAULT_SAVE_COUNT) -> List[str]:
    header = ["抽せん日"]
    for i in range(1, save_count + 1):
        header.extend([f"予測{i}", f"スコア正規化値{i}", f"戦略{i}"])
    return header


def prediction_row(target_date: str, ranked: Sequence[TicketScore], save_count: int = DEFAULT_SAVE_COUNT) -> Dict[str, str]:
    header = prediction_csv_header(save_count)
    row: Dict[str, str] = {key: "" for key in header}
    row["抽せん日"] = target_date

    top = list(ranked[:save_count])
    scores = score_normalized_values(top)

    for i in range(1, save_count + 1):
        if i <= len(top):
            row[f"予測{i}"] = format_ticket(top[i - 1].ticket, zero_pad=False)
            row[f"スコア正規化値{i}"] = f"{scores[i - 1]:.3f}".rstrip("0").rstrip(".")
            row[f"戦略{i}"] = top[i - 1].strategy
        else:
            row[f"予測{i}"] = ""
            row[f"スコア正規化値{i}"] = ""
            row[f"戦略{i}"] = ""

    return row


def save_predictions_csv(
    output_path: str,
    target_date: str,
    ranked: Sequence[TicketScore],
    save_count: int = DEFAULT_SAVE_COUNT,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = prediction_csv_header(save_count)
    new_row = prediction_row(target_date, ranked, save_count)

    rows: List[Dict[str, str]] = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for old in reader:
                normalized_row = {key: old.get(key, "") for key in header}
                rows.append(normalized_row)

    replaced = False
    for idx, old in enumerate(rows):
        if old.get("抽せん日") == target_date:
            rows[idx] = new_row
            replaced = True
            break

    if not replaced:
        rows.append(new_row)

    rows.sort(key=lambda r: r.get("抽せん日", ""))

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def save_latest_prediction_txt(
    path: str,
    target_date: str,
    tickets: Sequence[TicketScore],
) -> None:
    scores = score_normalized_values(tickets)
    lines: List[str] = []
    lines.append("Loto7 最新予測")
    lines.append("==================")
    lines.append(f"対象抽せん日: {target_date}")
    lines.append("")
    lines.append("予測結果")
    lines.append("------------------")

    for i, t in enumerate(tickets, start=1):
        score = scores[i - 1] if i - 1 < len(scores) else 0.0
        lines.append(
            f"{i:02d}. {format_ticket(t.ticket, zero_pad=False)}"
            f" / スコア正規化値: {score:.3f}".rstrip("0").rstrip(".")
            + f" / 戦略: {t.strategy}"
        )

    lines.append("")
    lines.append("注意")
    lines.append("------------------")
    lines.append("スコア正規化値は候補順位スコアであり、当選確率ではありません。")
    lines.append("この予測は過去実績に基づく候補生成です。")
    lines.append("的中保証は確認できません。")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def classify_loto7_prize(
    ticket: Sequence[int],
    actual_main: Sequence[int],
    actual_bonus: Sequence[int],
    prize_table: Dict[int, int],
) -> PrizeResult:
    ticket_set = set(ticket)
    main_matches = len(ticket_set & set(actual_main))
    bonus_matches = len(ticket_set & set(actual_bonus))

    grade: Optional[int]
    if main_matches == 7:
        grade = 1
    elif main_matches == 6 and bonus_matches >= 1:
        grade = 2
    elif main_matches == 6:
        grade = 3
    elif main_matches == 5:
        grade = 4
    elif main_matches == 4:
        grade = 5
    elif main_matches == 3 and bonus_matches >= 1:
        grade = 6
    else:
        grade = None

    prize = prize_table.get(grade, 0) if grade is not None else 0
    return PrizeResult(main_matches=main_matches, bonus_matches=bonus_matches, grade=grade, prize=prize)


def grade_label(grade: Optional[int]) -> str:
    return "ハズレ" if grade is None else f"{grade}等"


def backtest_detail_header(num_tickets: int) -> List[str]:
    header = [
        "抽せん日",
        "回別",
        "本数字",
        "ボーナス数字",
        "口数",
        "購入金額",
        "当せん金額",
        "収支",
        "回収率",
        "最高等級",
        "最高本数字一致数",
        "最高ボーナス一致数",
        "当せん口数",
    ]
    for i in range(1, num_tickets + 1):
        header.extend(
            [
                f"予測{i}",
                f"予測{i}_戦略",
                f"予測{i}_本数字一致",
                f"予測{i}_ボーナス一致",
                f"予測{i}_等級",
                f"予測{i}_当せん金額",
            ]
        )
    return header


def write_backtest_detail_csv(rows: Sequence[Dict[str, object]], path: str, num_tickets: int) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = backtest_detail_header(num_tickets)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def write_backtest_summary_csv(result: Dict[str, object], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "検証回数",
        "初期学習回数",
        "検証開始回相当",
        "要求バックテスト候補プール",
        "実効バックテスト候補プール",
        "直近検証回数指定",
        "1回あたり口数",
        "1口購入金額",
        "1口目平均一致数",
        "全口ベスト平均一致数",
        "1口目_2個以上率",
        "1口目_3個以上率",
        "1口目_4個以上率",
        "全口ベスト_2個以上率",
        "全口ベスト_3個以上率",
        "全口ベスト_4個以上率",
        "総購入金額",
        "総当せん金額",
        "総収支",
        "総回収率",
        "総当せん口数",
        "1口目購入金額",
        "1口目当せん金額",
        "1口目収支",
        "1口目回収率",
        "全予測口等級分布",
        "各回ベスト等級分布",
        "1口目一致数分布",
        "全口ベスト一致数分布",
    ]

    row = {
        "検証回数": result["trials"],
        "初期学習回数": result["min_train"],
        "検証開始回相当": result["evaluated_from_draw_index"],
        "要求バックテスト候補プール": result.get("requested_backtest_pool_size", ""),
        "実効バックテスト候補プール": result.get("effective_backtest_pool_size", ""),
        "直近検証回数指定": result.get("max_backtest_draws", 0),
        "1回あたり口数": result["tickets_per_draw"],
        "1口購入金額": result["unit_cost"],
        "1口目平均一致数": round(float(result["top1_avg"]), 6),
        "全口ベスト平均一致数": round(float(result["best_all_avg"]), 6),
        "1口目_2個以上率": round(float(result["top1_ge2"]), 6),
        "1口目_3個以上率": round(float(result["top1_ge3"]), 6),
        "1口目_4個以上率": round(float(result["top1_ge4"]), 6),
        "全口ベスト_2個以上率": round(float(result["best_all_ge2"]), 6),
        "全口ベスト_3個以上率": round(float(result["best_all_ge3"]), 6),
        "全口ベスト_4個以上率": round(float(result["best_all_ge4"]), 6),
        "総購入金額": result["total_purchase"],
        "総当せん金額": result["total_prize"],
        "総収支": result["total_profit"],
        "総回収率": round(float(result["total_return_rate"]), 6),
        "総当せん口数": result["total_winning_tickets"],
        "1口目購入金額": result["top1_purchase"],
        "1口目当せん金額": result["top1_prize"],
        "1口目収支": result["top1_profit"],
        "1口目回収率": round(float(result["top1_return_rate"]), 6),
        "全予測口等級分布": result["all_ticket_grade_dist"],
        "各回ベスト等級分布": result["best_grade_dist"],
        "1口目一致数分布": result["top1_dist"],
        "全口ベスト一致数分布": result["best_all_dist"],
    }

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerow(row)


def yen(value: int) -> str:
    return f"{value:,}円"


def write_backtest_report_txt(result: Dict[str, object], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("Loto7 バックテストレポート")
    lines.append("============================")
    lines.append("")
    lines.append("基本条件")
    lines.append("----------------------------")
    lines.append(f"検証回数: {result['trials']}")
    lines.append(f"初期学習回数: {result['min_train']}")
    lines.append(f"検証開始回相当: 第{result['evaluated_from_draw_index']}回")
    lines.append(f"要求バックテスト候補プール: {result.get('requested_backtest_pool_size', '')}")
    lines.append(f"実効バックテスト候補プール: {result.get('effective_backtest_pool_size', '')}")
    lines.append(f"直近検証回数指定: {result.get('max_backtest_draws', 0)}")
    lines.append(f"1回あたり口数: {result['tickets_per_draw']}")
    lines.append(f"1口購入金額: {yen(int(result['unit_cost']))}")
    lines.append("")
    lines.append("一致数")
    lines.append("----------------------------")
    lines.append(f"1口目 平均一致数: {float(result['top1_avg']):.3f}")
    lines.append(f"全口ベスト 平均一致数: {float(result['best_all_avg']):.3f}")
    lines.append("")
    lines.append("1口目")
    lines.append(f"2個以上: {float(result['top1_ge2']) * 100:.2f}%")
    lines.append(f"3個以上: {float(result['top1_ge3']) * 100:.2f}%")
    lines.append(f"4個以上: {float(result['top1_ge4']) * 100:.2f}%")
    lines.append(f"一致数分布: {result['top1_dist']}")
    lines.append("")
    lines.append("全口ベスト")
    lines.append(f"2個以上: {float(result['best_all_ge2']) * 100:.2f}%")
    lines.append(f"3個以上: {float(result['best_all_ge3']) * 100:.2f}%")
    lines.append(f"4個以上: {float(result['best_all_ge4']) * 100:.2f}%")
    lines.append(f"一致数分布: {result['best_all_dist']}")
    lines.append("")
    lines.append("等級・収支")
    lines.append("----------------------------")
    lines.append(f"総購入金額: {yen(int(result['total_purchase']))}")
    lines.append(f"総当せん金額: {yen(int(result['total_prize']))}")
    lines.append(f"総収支: {yen(int(result['total_profit']))}")
    lines.append(f"総回収率: {float(result['total_return_rate']) * 100:.2f}%")
    lines.append(f"総当せん口数: {result['total_winning_tickets']}")
    lines.append(f"全予測口 等級分布: {result['all_ticket_grade_dist']}")
    lines.append(f"各回ベスト等級分布: {result['best_grade_dist']}")
    lines.append("")
    lines.append("注意")
    lines.append("----------------------------")
    lines.append("当せん金額は設定値ベースの概算です。")
    lines.append("正確な回収率には各回の実当せん金額データが必要です。")
    lines.append("的中保証は確認できません。")


    lines.append("")
    lines.append("============================")
    lines.append("当選履歴一覧")
    lines.append("============================")

    history = result.get("winning_history", [])
    if not history:
        lines.append("当選履歴なし")
    else:
        for hit in history:
            lines.append(f"{hit['date']} 第{hit['draw_no']}回")
            lines.append(f"予測{hit['ticket_no']} / 戦略:{hit['strategy']}")
            lines.append(f"{hit['grade']}等")
            lines.append(f"組合せ: {hit['ticket']}")
            lines.append(f"本数字一致: {hit['main_matches']}")
            lines.append(f"ボーナス一致: {hit['bonus_matches']}")
            lines.append(f"当選金額: {hit['prize']:,}円")
            lines.append("--------------------------------")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def backtest(
    draws: Sequence[Draw],
    min_train: int = 100,
    num_tickets: int = DEFAULT_TICKETS,
    pool_size: int = 22,
    unit_cost: int = DEFAULT_UNIT_COST,
    prize_table: Optional[Dict[int, int]] = None,
    detail_output_csv: str = DEFAULT_BACKTEST_DETAIL_CSV,
    summary_output_csv: str = DEFAULT_BACKTEST_SUMMARY_CSV,
    report_output_txt: str = DEFAULT_BACKTEST_REPORT_TXT,
    pool_cap: int = DEFAULT_BACKTEST_POOL_CAP,
    max_backtest_draws: int = 0,
) -> Dict[str, object]:
    if len(draws) <= min_train:
        raise ValueError("バックテストには min_train より多いデータが必要です。")

    if min_train < 1:
        raise ValueError("min_train は1以上にしてください。第2回から検証する場合は --min-train 1 です。")

    if max_backtest_draws and max_backtest_draws > 0:
        # 直近N回だけ検証する場合も、学習に必要な過去データは残す。
        start_index = max(min_train, len(draws) - max_backtest_draws)
    else:
        start_index = min_train

    effective_pool_size = min(pool_size, pool_cap) if pool_cap and pool_cap > 0 else pool_size

    if prize_table is None:
        prize_table = dict(DEFAULT_PRIZE_TABLE)

    top1_hits: List[int] = []
    best_all_hits: List[int] = []
    hit_dist_top1: Counter = Counter()
    hit_dist_best_all: Counter = Counter()

    total_purchase = 0
    total_prize = 0
    total_winning_tickets = 0

    top1_purchase = 0
    top1_prize = 0

    all_ticket_grade_dist: Counter = Counter()
    best_grade_dist: Counter = Counter()

    detail_rows: List[Dict[str, object]] = []
    winning_history: List[Dict[str, object]] = []

    for i in range(start_index, len(draws)):
        train = draws[:i]
        actual = draws[i]
        tickets = predict(train, num_tickets=num_tickets, pool_size=effective_pool_size)

        results = [
            classify_loto7_prize(
                ticket=t.ticket,
                actual_main=actual.main,
                actual_bonus=actual.bonus,
                prize_table=prize_table,
            )
            for t in tickets
        ]

        hits = [r.main_matches for r in results]
        top1 = hits[0] if hits else 0
        best_all = max(hits) if hits else 0

        top1_hits.append(top1)
        best_all_hits.append(best_all)
        hit_dist_top1[top1] += 1
        hit_dist_best_all[best_all] += 1

        draw_purchase = len(tickets) * unit_cost
        draw_prize = sum(r.prize for r in results)
        draw_profit = draw_prize - draw_purchase
        draw_return_rate = draw_prize / draw_purchase if draw_purchase else 0.0

        total_purchase += draw_purchase
        total_prize += draw_prize
        total_winning_tickets += sum(1 for r in results if r.grade is not None)

        top1_purchase += unit_cost
        top1_prize += results[0].prize if results else 0

        grades = [r.grade for r in results if r.grade is not None]
        for r in results:
            all_ticket_grade_dist[grade_label(r.grade)] += 1

        best_grade: Optional[int] = min(grades) if grades else None
        best_grade_dist[grade_label(best_grade)] += 1

        row: Dict[str, object] = {
            "抽せん日": actual.date,
            "回別": actual.draw_no if actual.draw_no is not None else "",
            "本数字": format_ticket(actual.main),
            "ボーナス数字": format_ticket(actual.bonus),
            "口数": len(tickets),
            "購入金額": draw_purchase,
            "当せん金額": draw_prize,
            "収支": draw_profit,
            "回収率": round(draw_return_rate, 6),
            "最高等級": grade_label(best_grade),
            "最高本数字一致数": best_all,
            "最高ボーナス一致数": max((r.bonus_matches for r in results), default=0),
            "当せん口数": sum(1 for r in results if r.grade is not None),
        }

        for idx, (ticket, result) in enumerate(zip(tickets, results), start=1):
            row[f"予測{idx}"] = format_ticket(ticket.ticket)
            row[f"予測{idx}_戦略"] = ticket.strategy
            row[f"予測{idx}_本数字一致"] = result.main_matches
            row[f"予測{idx}_ボーナス一致"] = result.bonus_matches
            row[f"予測{idx}_等級"] = grade_label(result.grade)
            row[f"予測{idx}_当せん金額"] = result.prize

            if result.grade is not None:
                winning_history.append({
                    "date": actual.date,
                    "draw_no": actual.draw_no,
                    "ticket_no": idx,
                    "grade": result.grade,
                    "main_matches": result.main_matches,
                    "bonus_matches": result.bonus_matches,
                    "prize": result.prize,
                    "ticket": format_ticket(ticket.ticket),
                    "strategy": ticket.strategy,
                })

        detail_rows.append(row)

    def rate(values: Sequence[int], threshold: int) -> float:
        return sum(1 for x in values if x >= threshold) / len(values) if values else 0.0

    total_profit = total_prize - total_purchase
    total_return_rate = total_prize / total_purchase if total_purchase else 0.0
    top1_profit = top1_prize - top1_purchase
    top1_return_rate = top1_prize / top1_purchase if top1_purchase else 0.0

    result = {
        "trials": len(top1_hits),
        "min_train": min_train,
        "evaluated_from_draw_index": start_index + 1,
        "requested_backtest_pool_size": pool_size,
        "effective_backtest_pool_size": effective_pool_size,
        "max_backtest_draws": max_backtest_draws,
        "tickets_per_draw": num_tickets,
        "unit_cost": unit_cost,
        "prize_table": prize_table,
        "top1_avg": sum(top1_hits) / len(top1_hits) if top1_hits else 0.0,
        "best_all_avg": sum(best_all_hits) / len(best_all_hits) if best_all_hits else 0.0,
        "top1_ge2": rate(top1_hits, 2),
        "top1_ge3": rate(top1_hits, 3),
        "top1_ge4": rate(top1_hits, 4),
        "best_all_ge2": rate(best_all_hits, 2),
        "best_all_ge3": rate(best_all_hits, 3),
        "best_all_ge4": rate(best_all_hits, 4),
        "top1_dist": dict(sorted(hit_dist_top1.items())),
        "best_all_dist": dict(sorted(hit_dist_best_all.items())),
        "total_purchase": total_purchase,
        "total_prize": total_prize,
        "total_profit": total_profit,
        "total_return_rate": total_return_rate,
        "total_winning_tickets": total_winning_tickets,
        "top1_purchase": top1_purchase,
        "top1_prize": top1_prize,
        "top1_profit": top1_profit,
        "top1_return_rate": top1_return_rate,
        "all_ticket_grade_dist": dict(sorted(all_ticket_grade_dist.items())),
        "best_grade_dist": dict(sorted(best_grade_dist.items())),
        "detail_output_csv": detail_output_csv,
        "summary_output_csv": summary_output_csv,
        "report_output_txt": report_output_txt,
        "winning_history": winning_history,
    }

    write_backtest_detail_csv(detail_rows, detail_output_csv, num_tickets=num_tickets)
    write_backtest_summary_csv(result, summary_output_csv)
    write_backtest_report_txt(result, report_output_txt)

    return result


def next_friday_after(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d").date()
    return (dt + timedelta(days=7)).isoformat()


def print_recent_summary(draws: Sequence[Draw]) -> None:
    latest = draws[-1]
    print("=== 最新データ ===")
    print(f"抽せん日: {latest.date}")
    print(f"回別: {latest.draw_no if latest.draw_no is not None else '不明'}")
    print(f"本数字: {format_ticket(latest.main)}")
    print(f"ボーナス: {format_ticket(latest.bonus)}")
    print(f"次回想定日: {next_friday_after(latest.date)}")
    print()

    print("=== 直近10回の出現回数 ===")
    c10 = count_numbers(draws, 10)
    for n, cnt in sorted(c10.items(), key=lambda x: (-x[1], x[0])):
        print(f"{n:02d}: {cnt}回")
    print()

    print("=== 直近20回の強ペア TOP15 ===")
    p20 = count_combinations(draws, 20, 2)
    for pair, cnt in p20.most_common(15):
        print(f"{pair[0]:02d}-{pair[1]:02d}: {cnt}回")
    print()


def print_predictions(tickets: Sequence[TicketScore]) -> None:
    print("=== 次回予測 ===")
    scores = score_normalized_values(tickets)
    for i, t in enumerate(tickets, start=1):
        d = t.detail
        score = scores[i - 1] if i - 1 < len(scores) else 0.0
        print(
            f"{i:02d}. {format_ticket(t.ticket)}"
            f" | スコア正規化値={score:.3f}"
            f" | strategy={t.strategy}"
            f" | raw_score={t.score:.3f}"
            f" | sum={int(d['sum'])}"
            f" | odd={int(d['odd'])}"
            f" | low/mid/high={int(d['low'])}/{int(d['mid'])}/{int(d['high'])}"
            f" | repeat_last={int(d['repeat_last'])}"
        )
    print()


def print_prize_table(prize_table: Dict[int, int], unit_cost: int) -> None:
    print("=== バックテスト用 設定金額 ===")
    print(f"1口購入金額: {yen(unit_cost)}")
    for grade in range(1, 7):
        print(f"{grade}等: {yen(prize_table.get(grade, 0))}")
    print("※実際の当せん金額は各回で変動します。正確な収支には各回の実当せん金額データが必要です。")
    print()


def print_backtest_summary(result: Dict[str, object]) -> None:
    print("=== バックテスト結果 ===")
    print(f"検証回数: {result['trials']}")
    print(f"開始条件: min_train={result['min_train']} / 第{result['evaluated_from_draw_index']}回相当から検証")
    print(f"要求バックテスト候補プール: {result.get('requested_backtest_pool_size', '')}")
    print(f"実効バックテスト候補プール: {result.get('effective_backtest_pool_size', '')}")
    print(f"直近検証回数指定: {result.get('max_backtest_draws', 0)}")
    print(f"1回あたり口数: {result['tickets_per_draw']}")
    print(f"1口購入金額: {yen(int(result['unit_cost']))}")
    print()
    print("一致数:")
    print(f"  1口目 平均一致数: {float(result['top1_avg']):.3f}")
    print(f"  全口ベスト 平均一致数: {float(result['best_all_avg']):.3f}")
    print("  1口目:")
    print(f"    2個以上: {float(result['top1_ge2']):.1%}")
    print(f"    3個以上: {float(result['top1_ge3']):.1%}")
    print(f"    4個以上: {float(result['top1_ge4']):.1%}")
    print(f"    分布: {result['top1_dist']}")
    print("  全口ベスト:")
    print(f"    2個以上: {float(result['best_all_ge2']):.1%}")
    print(f"    3個以上: {float(result['best_all_ge3']):.1%}")
    print(f"    4個以上: {float(result['best_all_ge4']):.1%}")
    print(f"    分布: {result['best_all_dist']}")
    print()
    print("等級・収支:")
    print(f"  総購入金額: {yen(int(result['total_purchase']))}")
    print(f"  総当せん金額: {yen(int(result['total_prize']))}")
    print(f"  総収支: {yen(int(result['total_profit']))}")
    print(f"  総回収率: {float(result['total_return_rate']) * 100:.2f}%")
    print(f"  総当せん口数: {result['total_winning_tickets']}")
    print(f"  全予測口 等級分布: {result['all_ticket_grade_dist']}")
    print(f"  各回ベスト等級分布: {result['best_grade_dist']}")
    print()
    print("出力ファイル:")
    print(f"  明細CSV: {result['detail_output_csv']}")
    print(f"  サマリーCSV: {result['summary_output_csv']}")
    print(f"  レポートTXT: {result['report_output_txt']}")
    print()


def build_prize_table_from_args(args: argparse.Namespace) -> Dict[int, int]:
    return {
        1: int(args.prize1),
        2: int(args.prize2),
        3: int(args.prize3),
        4: int(args.prize4),
        5: int(args.prize5),
        6: int(args.prize6),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="loto7.csv だけでロト7予測とバックテストを出力します。"
    )
    parser.add_argument("--csv", default=DEFAULT_CSV_URL, help="loto7.csv のURLまたはローカルパス。既定: loto7.csv")
    parser.add_argument("--tickets", type=int, default=DEFAULT_TICKETS, help="画面表示・バックテストする口数。既定: 10")
    parser.add_argument("--pool-size", type=int, default=24, help="候補プールサイズ。大きいほど遅くなる。既定: 24")
    parser.add_argument("--backtest", action="store_true", help="バックテストも実行する")
    parser.add_argument("--min-train", type=int, default=100, help="バックテストの初期学習回数。第2回から検証するなら1")
    parser.add_argument("--backtest-pool-size", type=int, default=16, help="バックテスト時の候補プールサイズ。既定: 16")
    parser.add_argument("--backtest-pool-cap", type=int, default=DEFAULT_BACKTEST_POOL_CAP, help="Actionsタイムアウト防止用のバックテスト候補プール上限。既定: 16")
    parser.add_argument("--max-backtest-draws", type=int, default=0, help="直近N回だけバックテストする。0なら全件")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="予測CSV保存先")
    parser.add_argument("--latest-txt", default=DEFAULT_LATEST_TXT, help="最新予測TXT保存先")
    parser.add_argument("--save-count", type=int, default=DEFAULT_SAVE_COUNT, help="保存する予測数。既定: 10")
    parser.add_argument("--no-save", action="store_true", help="予測CSV/TXTを保存しない")
    parser.add_argument("--unit-cost", type=int, default=DEFAULT_UNIT_COST, help="1口の購入金額。既定: 300")
    parser.add_argument("--backtest-summary-csv", default=DEFAULT_BACKTEST_SUMMARY_CSV, help="バックテストサマリーCSV")
    parser.add_argument("--backtest-detail-csv", default=DEFAULT_BACKTEST_DETAIL_CSV, help="バックテスト明細CSV")
    parser.add_argument("--backtest-report-txt", default=DEFAULT_BACKTEST_REPORT_TXT, help="バックテストレポートTXT")
    parser.add_argument("--prize1", type=int, default=DEFAULT_PRIZE_TABLE[1], help="1等の設定当せん金額")
    parser.add_argument("--prize2", type=int, default=DEFAULT_PRIZE_TABLE[2], help="2等の設定当せん金額")
    parser.add_argument("--prize3", type=int, default=DEFAULT_PRIZE_TABLE[3], help="3等の設定当せん金額")
    parser.add_argument("--prize4", type=int, default=DEFAULT_PRIZE_TABLE[4], help="4等の設定当せん金額")
    parser.add_argument("--prize5", type=int, default=DEFAULT_PRIZE_TABLE[5], help="5等の設定当せん金額")
    parser.add_argument("--prize6", type=int, default=DEFAULT_PRIZE_TABLE[6], help="6等の設定当せん金額")

    parser.add_argument(
        "--backtest-output-csv",
        default=None,
        help="旧workflow互換オプション"
    )

    args = parser.parse_args(argv)

    
    if args.backtest_output_csv:
        args.backtest_summary_csv = args.backtest_output_csv


    draws = load_draws(args.csv)
    if not draws:
        print("抽せんデータを読み込めませんでした。", file=sys.stderr)
        return 1

    latest = draws[-1]
    target_date = next_friday_after(latest.date)

    print_recent_summary(draws)

    display_tickets = predict(draws, num_tickets=args.tickets, pool_size=args.pool_size)
    print_predictions(display_tickets)

    if not args.no_save:
        save_predictions_csv(
            output_path=args.output_csv,
            target_date=target_date,
            ranked=display_tickets,
            save_count=args.save_count,
        )
        save_latest_prediction_txt(
            path=args.latest_txt,
            target_date=target_date,
            tickets=display_tickets[: args.save_count],
        )
        print(f"保存完了: {args.output_csv}")
        print(f"保存完了: {args.latest_txt}")
        print(f"保存形式: 抽せん日, 予測1, スコア正規化値1, 戦略1 ...")
        print(f"対象抽せん日: {target_date}")
        print()

    if args.backtest:
        prize_table = build_prize_table_from_args(args)
        print_prize_table(prize_table, args.unit_cost)
        bt = backtest(
            draws,
            min_train=args.min_train,
            num_tickets=args.tickets,
            pool_size=args.backtest_pool_size,
            unit_cost=args.unit_cost,
            prize_table=prize_table,
            detail_output_csv=args.backtest_detail_csv,
            summary_output_csv=args.backtest_summary_csv,
            report_output_txt=args.backtest_report_txt,
            pool_cap=args.backtest_pool_cap,
            max_backtest_draws=args.max_backtest_draws,
        )
        print_backtest_summary(bt)

    print("注意: スコア正規化値は当選確率ではありません。")
    print("注意: 的中保証は確認できません。過去実績ベースの候補生成です。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
