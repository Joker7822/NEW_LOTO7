#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loto7_logic_predictor.py

loto7_predictions.csv を使わず、loto7.csv の過去実績だけで
ロト7の次回候補を生成し、loto7_predictions.csv 互換形式で保存する。

保存形式:
    抽せん日,予測1,信頼度1,...,予測25,信頼度25

注意:
    宝くじの当せんはランダム性が強く、的中保証は確認できません。
    このコードは過去実績に基づく候補生成・検証ツールです。
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
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_CSV_URL = "https://raw.githubusercontent.com/Joker7822/loto7/main/loto7.csv"
DEFAULT_OUTPUT_CSV = "loto7_predictions.csv"
NUM_MIN = 1
NUM_MAX = 37
PICK_SIZE = 7
DEFAULT_SAVE_COUNT = 25


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
    gaps = last_seen_gaps(draws)

    max10 = max(c10.values()) if c10 else 1
    max20 = max(c20.values()) if c20 else 1
    max50 = max(c50.values()) if c50 else 1
    max100 = max(c100.values()) if c100 else 1

    scores: Dict[int, float] = {}

    for n in range(NUM_MIN, NUM_MAX + 1):
        hot_score = (
            4.0 * normalized(c10, n, max10)
            + 2.5 * normalized(c20, n, max20)
            + 1.5 * normalized(c50, n, max50)
            + 0.8 * normalized(c100, n, max100)
        )

        gap = gaps[n]
        if gap == 0:
            gap_score = -0.15
        elif 3 <= gap <= 18:
            gap_score = 0.15 + min(gap, 18) / 100.0
        elif 19 <= gap <= 25:
            gap_score = 0.15
        else:
            gap_score = 0.0

        scores[n] = hot_score + gap_score

    return scores


def band_counts(ticket: Sequence[int]) -> Tuple[int, int, int]:
    low = sum(1 for n in ticket if 1 <= n <= 12)
    mid = sum(1 for n in ticket if 13 <= n <= 25)
    high = sum(1 for n in ticket if 26 <= n <= 37)
    return low, mid, high


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
    if not (2 <= low <= 3):
        penalty += 0.30 * abs(low - 2.5)
    if not (2 <= mid <= 3):
        penalty += 0.30 * abs(mid - 2.5)
    if not (1 <= high <= 2):
        penalty += 0.35 * abs(high - 1.5)

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
    total_score = single + pair_score + triple_score - penalty

    return TicketScore(
        ticket=ticket,
        score=total_score,
        detail={
            "single": single,
            "pair": pair_score,
            "triple": triple_score,
            "penalty": penalty,
            "sum": float(sum(ticket)),
            "odd": float(sum(1 for n in ticket if n % 2 == 1)),
            "repeat_last": float(len(set(ticket) & set(last_main))),
        },
    )


def make_candidate_pool(draws: Sequence[Draw], pool_size: int = 21) -> List[int]:
    number_scores = build_number_scores(draws)
    pair20 = count_combinations(draws, 20, 2)
    pair50 = count_combinations(draws, 50, 2)
    tri20 = count_combinations(draws, 20, 3)

    pool = set()

    for n, _ in sorted(number_scores.items(), key=lambda x: (-x[1], x[0]))[:16]:
        pool.add(n)
    for pair, _ in pair20.most_common(12):
        pool.update(pair)
    for pair, _ in pair50.most_common(8):
        pool.update(pair)
    for tri, _ in tri20.most_common(8):
        pool.update(tri)

    for n, _ in sorted(number_scores.items(), key=lambda x: (-x[1], x[0])):
        pool.add(n)
        if len(pool) >= pool_size:
            break

    return sorted(pool, key=lambda n: (-number_scores[n], n))[:pool_size]


def rank_tickets(draws: Sequence[Draw], pool_size: int = 21, max_rank: int = 500) -> List[TicketScore]:
    number_scores = build_number_scores(draws)
    pool = make_candidate_pool(draws, pool_size=pool_size)

    pair10 = count_combinations(draws, 10, 2)
    pair20 = count_combinations(draws, 20, 2)
    pair50 = count_combinations(draws, 50, 2)
    triple20 = count_combinations(draws, 20, 3)
    triple50 = count_combinations(draws, 50, 3)

    ranked: List[TicketScore] = []
    for comb in itertools.combinations(sorted(pool), PICK_SIZE):
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
            )
        )

    ranked.sort(key=lambda x: (-x.score, x.ticket))
    return ranked[:max_rank]


def select_diverse_tickets(
    ranked: Sequence[TicketScore],
    num_tickets: int = 5,
    max_overlap: int = 5,
    max_number_usage: int = 4,
) -> List[TicketScore]:
    selected: List[TicketScore] = []
    usage: Counter = Counter()

    for cand in ranked:
        s = set(cand.ticket)
        if selected:
            max_ov = max(len(s & set(x.ticket)) for x in selected)
            if max_ov > max_overlap:
                continue
        if any(usage[n] >= max_number_usage for n in cand.ticket):
            continue

        selected.append(cand)
        usage.update(cand.ticket)
        if len(selected) >= num_tickets:
            return selected

    for cand in ranked:
        if cand not in selected:
            selected.append(cand)
        if len(selected) >= num_tickets:
            break

    return selected[:num_tickets]


def predict(draws: Sequence[Draw], num_tickets: int = 5, pool_size: int = 21) -> List[TicketScore]:
    ranked = rank_tickets(draws, pool_size=pool_size, max_rank=500)
    return select_diverse_tickets(ranked, num_tickets=num_tickets)


def confidence_values(ranked: Sequence[TicketScore]) -> List[float]:
    """
    loto7_predictions.csv の既存レンジに合わせて、
    1位=0.970、最下位=0.720付近になるよう正規化する。
    """
    if not ranked:
        return []
    scores = [x.score for x in ranked]
    max_score = max(scores)
    min_score = min(scores)

    if max_score == min_score:
        return [0.970 for _ in ranked]

    values = []
    for s in scores:
        conf = 0.720 + ((s - min_score) / (max_score - min_score)) * 0.250
        values.append(round(conf, 3))
    return values


def prediction_csv_header(save_count: int = DEFAULT_SAVE_COUNT) -> List[str]:
    header = ["抽せん日"]
    for i in range(1, save_count + 1):
        header.extend([f"予測{i}", f"信頼度{i}"])
    return header


def prediction_row(target_date: str, ranked: Sequence[TicketScore], save_count: int = DEFAULT_SAVE_COUNT) -> Dict[str, str]:
    header = prediction_csv_header(save_count)
    row: Dict[str, str] = {key: "" for key in header}
    row["抽せん日"] = target_date

    top = list(ranked[:save_count])
    confs = confidence_values(top)

    for i in range(1, save_count + 1):
        if i <= len(top):
            row[f"予測{i}"] = format_ticket(top[i - 1].ticket, zero_pad=False)
            row[f"信頼度{i}"] = f"{confs[i - 1]:.3f}".rstrip("0").rstrip(".")
        else:
            row[f"予測{i}"] = ""
            row[f"信頼度{i}"] = ""

    return row


def save_predictions_csv(
    output_path: str,
    target_date: str,
    ranked: Sequence[TicketScore],
    save_count: int = DEFAULT_SAVE_COUNT,
) -> None:
    """
    loto7_predictions.csv 互換形式で保存する。
    同じ抽せん日が既に存在する場合は上書きし、なければ追記する。
    """
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


def evaluate_prediction(ticket: Sequence[int], actual_main: Sequence[int]) -> int:
    return len(set(ticket) & set(actual_main))


def backtest(
    draws: Sequence[Draw],
    min_train: int = 100,
    num_tickets: int = 5,
    pool_size: int = 19,
) -> Dict[str, object]:
    if len(draws) <= min_train:
        raise ValueError("バックテストには min_train より多いデータが必要です。")

    top1_hits: List[int] = []
    best5_hits: List[int] = []
    hit_dist_top1: Counter = Counter()
    hit_dist_best5: Counter = Counter()

    for i in range(min_train, len(draws)):
        train = draws[:i]
        actual = draws[i]
        tickets = predict(train, num_tickets=num_tickets, pool_size=pool_size)
        hits = [evaluate_prediction(t.ticket, actual.main) for t in tickets]

        top1 = hits[0]
        best5 = max(hits)
        top1_hits.append(top1)
        best5_hits.append(best5)
        hit_dist_top1[top1] += 1
        hit_dist_best5[best5] += 1

    def rate(values: Sequence[int], threshold: int) -> float:
        return sum(1 for x in values if x >= threshold) / len(values) if values else 0.0

    return {
        "trials": len(top1_hits),
        "top1_avg": sum(top1_hits) / len(top1_hits),
        "best5_avg": sum(best5_hits) / len(best5_hits),
        "top1_ge2": rate(top1_hits, 2),
        "top1_ge3": rate(top1_hits, 3),
        "top1_ge4": rate(top1_hits, 4),
        "best5_ge2": rate(best5_hits, 2),
        "best5_ge3": rate(best5_hits, 3),
        "best5_ge4": rate(best5_hits, 4),
        "top1_dist": dict(sorted(hit_dist_top1.items())),
        "best5_dist": dict(sorted(hit_dist_best5.items())),
    }


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

    print("=== 直近50回の強ペア TOP15 ===")
    p50 = count_combinations(draws, 50, 2)
    for pair, cnt in p50.most_common(15):
        print(f"{pair[0]:02d}-{pair[1]:02d}: {cnt}回")
    print()


def print_predictions(tickets: Sequence[TicketScore]) -> None:
    print("=== 次回予測 ===")
    for i, t in enumerate(tickets, start=1):
        d = t.detail
        print(
            f"{i}. {format_ticket(t.ticket)}"
            f" | score={t.score:.3f}"
            f" | sum={int(d['sum'])}"
            f" | odd={int(d['odd'])}"
            f" | repeat_last={int(d['repeat_last'])}"
        )
    print()


def print_backtest_summary(result: Dict[str, object]) -> None:
    print("=== バックテスト結果 ===")
    print(f"検証回数: {result['trials']}")
    print(f"1口目 平均一致数: {result['top1_avg']:.3f}")
    print(f"5口内ベスト 平均一致数: {result['best5_avg']:.3f}")
    print()
    print("1口目:")
    print(f"  2個以上: {result['top1_ge2']:.1%}")
    print(f"  3個以上: {result['top1_ge3']:.1%}")
    print(f"  4個以上: {result['top1_ge4']:.1%}")
    print(f"  分布: {result['top1_dist']}")
    print()
    print("5口内ベスト:")
    print(f"  2個以上: {result['best5_ge2']:.1%}")
    print(f"  3個以上: {result['best5_ge3']:.1%}")
    print(f"  4個以上: {result['best5_ge4']:.1%}")
    print(f"  分布: {result['best5_dist']}")
    print()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="loto7.csv の実績だけでロト7予測とバックテストを出力し、loto7_predictions.csv形式で保存します。"
    )
    parser.add_argument("--csv", default=DEFAULT_CSV_URL, help="loto7.csv のURLまたはローカルパス")
    parser.add_argument("--tickets", type=int, default=5, help="画面に表示する口数")
    parser.add_argument("--pool-size", type=int, default=21, help="候補プールサイズ。大きいほど遅くなる")
    parser.add_argument("--backtest", action="store_true", help="バックテストも実行する")
    parser.add_argument("--min-train", type=int, default=100, help="バックテストの初期学習回数")
    parser.add_argument("--backtest-pool-size", type=int, default=19, help="バックテスト時の候補プールサイズ")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="保存先CSV。既定: loto7_predictions.csv")
    parser.add_argument("--save-count", type=int, default=DEFAULT_SAVE_COUNT, help="保存する予測数。既定: 25")
    parser.add_argument("--no-save", action="store_true", help="予測CSVを保存しない")
    args = parser.parse_args(argv)

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
        ranked_for_csv = rank_tickets(
            draws,
            pool_size=args.pool_size,
            max_rank=max(args.save_count, DEFAULT_SAVE_COUNT),
        )
        save_predictions_csv(
            output_path=args.output_csv,
            target_date=target_date,
            ranked=ranked_for_csv,
            save_count=args.save_count,
        )
        print(f"保存完了: {args.output_csv}")
        print(f"保存形式: 抽せん日, 予測1, 信頼度1 ... 予測{args.save_count}, 信頼度{args.save_count}")
        print(f"対象抽せん日: {target_date}")
        print()

    if args.backtest:
        bt = backtest(
            draws,
            min_train=args.min_train,
            num_tickets=args.tickets,
            pool_size=args.backtest_pool_size,
        )
        print_backtest_summary(bt)

    print("注意: 的中保証は確認できません。過去実績ベースの候補生成です。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
