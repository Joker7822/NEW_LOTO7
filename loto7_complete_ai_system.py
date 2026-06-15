#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loto7_complete_ai_system.py

LOTO7 完全版統合システム。

実装範囲:
  - Meta Ensemble
  - Master Champion Tournament
  - ROI Optimizer
  - Self Evolution Loop controller
  - Bayesian Update
  - Monte Carlo simulation
  - MCTS-like constructive search
  - Diversity Optimizer
  - Feature Store export
  - Explainability dashboard JSON/Markdown

注意:
  ロト7はランダム抽せんであり、的中・収益を保証しない。
  本システムは過去バックテスト上の相対評価と探索を自動化する。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

NUMBERS = tuple(range(1, 38))
PAYOUT_ESTIMATE = {
    "1等": 600_000_000,
    "2等": 7_000_000,
    "3等": 700_000,
    "4等": 12_500,
    "5等": 1_800,
    "6等": 1_000,
    "外れ": 0,
}
TICKET_COST = 300


@dataclass(frozen=True)
class Draw:
    draw_no: int
    date: str
    main: Tuple[int, ...]
    bonus: Tuple[int, ...]


def parse_nums(text: object) -> Tuple[int, ...]:
    return tuple(int(x) for x in str(text or "").replace(",", " ").split() if x.isdigit())


def draw_no_int(text: object) -> Optional[int]:
    import re
    m = re.search(r"\d+", str(text or ""))
    return int(m.group(0)) if m else None


def load_draws(csv_path: str) -> List[Draw]:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(csv_path)
    rows: List[Draw] = []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            no = draw_no_int(row.get("回別"))
            main = parse_nums(row.get("本数字"))
            bonus = parse_nums(row.get("ボーナス数字"))
            date = str(row.get("抽せん日") or "").strip()
            if no is None or len(main) != 7 or len(set(main)) != 7:
                continue
            if len(bonus) != 2:
                bonus = tuple()
            rows.append(Draw(no, date, tuple(sorted(main)), tuple(sorted(bonus))))
    return sorted(rows, key=lambda d: d.draw_no)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: str, data: Dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default


def prize_rank(main_match: int, bonus_match: int) -> str:
    if main_match == 7:
        return "1等"
    if main_match == 6 and bonus_match >= 1:
        return "2等"
    if main_match == 6:
        return "3等"
    if main_match == 5:
        return "4等"
    if main_match == 4:
        return "5等"
    if main_match == 3 and bonus_match >= 1:
        return "6等"
    return "外れ"


def evaluate(ticket: Sequence[int], target: Draw) -> Tuple[int, int, str, int]:
    s = set(ticket)
    main = len(s & set(target.main))
    bonus = len(s & set(target.bonus)) if target.bonus else 0
    rank = prize_rank(main, bonus)
    return main, bonus, rank, PAYOUT_ESTIMATE[rank]


def frequency_scores(draws: Sequence[Draw], decay: float = 0.985) -> Dict[int, float]:
    scores = {n: 1.0 for n in NUMBERS}
    total = len(draws)
    for i, d in enumerate(draws):
        w = decay ** (total - i - 1)
        for n in d.main:
            scores[n] += w
    return scores


def pair_scores(draws: Sequence[Draw], decay: float = 0.990) -> Dict[Tuple[int, int], float]:
    scores: Dict[Tuple[int, int], float] = {}
    total = len(draws)
    for i, d in enumerate(draws):
        w = decay ** (total - i - 1)
        for a, b in itertools.combinations(d.main, 2):
            k = (min(a, b), max(a, b))
            scores[k] = scores.get(k, 0.0) + w
    return scores


def bayesian_update(draws: Sequence[Draw], alpha: float = 1.0) -> Dict[int, float]:
    # Beta-Bernoulli posterior for each number appearing in one draw.
    total = len(draws)
    counts = {n: 0 for n in NUMBERS}
    for d in draws:
        for n in d.main:
            counts[n] += 1
    # prior centered at 7/37
    prior_a = alpha * 7 / 37
    prior_b = alpha * 30 / 37
    posterior = {n: (prior_a + counts[n]) / (prior_a + prior_b + total) for n in NUMBERS}
    return posterior


def ticket_features(ticket: Sequence[int], draws: Sequence[Draw]) -> Dict[str, float]:
    t = tuple(sorted(ticket))
    freq_full = frequency_scores(draws, 0.990)
    freq_240 = frequency_scores(draws[-240:], 0.985)
    freq_120 = frequency_scores(draws[-120:], 0.980)
    freq_60 = frequency_scores(draws[-60:], 0.970)
    pair = pair_scores(draws[-240:], 0.990)
    bayes = bayesian_update(draws[-240:] if len(draws) > 240 else draws)
    odd = sum(n % 2 for n in t)
    low = sum(1 for n in t if n <= 18)
    consecutive = sum(1 for a, b in zip(t, t[1:]) if b == a + 1)
    return {
        "sum": float(sum(t)),
        "odd": float(odd),
        "low": float(low),
        "consecutive": float(consecutive),
        "range": float(max(t) - min(t)),
        "freq_full": sum(freq_full[n] for n in t),
        "freq_240": sum(freq_240[n] for n in t),
        "freq_120": sum(freq_120[n] for n in t),
        "freq_60": sum(freq_60[n] for n in t),
        "pair_score": sum(pair.get((min(a, b), max(a, b)), 0.0) for a, b in itertools.combinations(t, 2)),
        "bayes_score": sum(bayes[n] for n in t),
        "zone_1_9": float(sum(1 for n in t if 1 <= n <= 9)),
        "zone_10_18": float(sum(1 for n in t if 10 <= n <= 18)),
        "zone_19_27": float(sum(1 for n in t if 19 <= n <= 27)),
        "zone_28_37": float(sum(1 for n in t if 28 <= n <= 37)),
    }


def structural_score(features: Dict[str, float]) -> float:
    score = 0.0
    score += 3.0 if features["odd"] in (3, 4) else -1.0
    score += 2.0 if features["low"] in (3, 4) else -1.0
    score += 3.0 if 85 <= features["sum"] <= 190 else -abs(features["sum"] - 137) / 30
    score -= max(0.0, features["consecutive"] - 2) * 1.5
    score += features["freq_240"] * 0.25 + features["freq_120"] * 0.25 + features["freq_60"] * 0.20
    score += features["pair_score"] * 0.12
    score += features["bayes_score"] * 20.0
    return score


def generate_candidates(draws: Sequence[Draw], n: int = 2000, seed: int = 777) -> List[Tuple[int, ...]]:
    rng = random.Random(seed + len(draws))
    freq = frequency_scores(draws[-240:] if len(draws) > 240 else draws)
    weighted = list(NUMBERS)
    weights = [max(0.001, freq[x]) for x in weighted]
    out = set()
    # deterministic top pool combinations
    top_pool = [x for x, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:22]]
    for combo in itertools.combinations(sorted(top_pool), 7):
        odd = sum(x % 2 for x in combo)
        s = sum(combo)
        if 80 <= s <= 195 and 2 <= odd <= 5:
            out.add(tuple(combo))
        if len(out) >= n // 2:
            break
    # weighted random candidates
    while len(out) < n:
        ticket = set()
        while len(ticket) < 7:
            ticket.add(rng.choices(weighted, weights=weights, k=1)[0])
        out.add(tuple(sorted(ticket)))
    return list(out)


def monte_carlo_score(ticket: Sequence[int], draws: Sequence[Draw], sims: int = 20000, seed: int = 777) -> Dict[str, float]:
    rng = random.Random(seed + sum(ticket))
    bayes = bayesian_update(draws[-240:] if len(draws) > 240 else draws)
    nums = list(NUMBERS)
    weights = [bayes[n] for n in nums]
    rank_counts = {r: 0 for r in PAYOUT_ESTIMATE}
    payout_total = 0
    for _ in range(sims):
        main = set()
        while len(main) < 7:
            main.add(rng.choices(nums, weights=weights, k=1)[0])
        remain = [n for n in nums if n not in main]
        bonus = set(rng.sample(remain, 2))
        fake = Draw(0, "", tuple(sorted(main)), tuple(sorted(bonus)))
        _, _, rank, payout = evaluate(ticket, fake)
        rank_counts[rank] += 1
        payout_total += payout
    expected_payout = payout_total / max(1, sims)
    return {
        "mc_expected_payout": expected_payout,
        "mc_expected_roi": (expected_payout - TICKET_COST) / TICKET_COST,
        **{f"mc_prob_{k}": v / max(1, sims) for k, v in rank_counts.items()},
    }


def mcts_construct(draws: Sequence[Draw], iterations: int = 2000, seed: int = 777) -> List[Tuple[int, ...]]:
    rng = random.Random(seed + len(draws) * 13)
    bayes = bayesian_update(draws[-240:] if len(draws) > 240 else draws)
    pair = pair_scores(draws[-240:] if len(draws) > 240 else draws)
    candidates = set()
    for _ in range(iterations):
        current: List[int] = []
        remaining = set(NUMBERS)
        while len(current) < 7:
            best = None
            best_score = -1e18
            sample = rng.sample(list(remaining), min(len(remaining), 10))
            for n in sample:
                tmp = tuple(sorted(current + [n]))
                s = bayes[n] * 100
                for a in current:
                    s += pair.get((min(a, n), max(a, n)), 0.0) * 0.25
                if len(tmp) == 7:
                    f = ticket_features(tmp, draws)
                    s += structural_score(f)
                s += rng.random() * 0.25
                if s > best_score:
                    best_score = s
                    best = n
            current.append(best)  # type: ignore[arg-type]
            remaining.remove(best)  # type: ignore[arg-type]
        candidates.add(tuple(sorted(current)))
    return list(candidates)


def parse_prediction_files() -> List[Tuple[int, ...]]:
    tickets = set()
    for path in glob.glob("outputs/evolution_best_prediction*.csv"):
        for row in read_csv_rows(path):
            nums = parse_nums(row.get("numbers"))
            if len(nums) == 7:
                tickets.add(tuple(sorted(nums)))
    return list(tickets)


def load_champions() -> List[Dict[str, object]]:
    champs = []
    for path in glob.glob("loto7_best_model*.json"):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            genome = data.get("genome", data)
            champs.append({"source_file": path, "id": genome.get("id"), "score": safe_float(genome.get("score")), "generation": genome.get("generation"), "genome": genome})
        except Exception as exc:
            champs.append({"source_file": path, "error": str(exc), "score": 0.0})
    return sorted(champs, key=lambda x: safe_float(x.get("score")), reverse=True)


def meta_ensemble_score(ticket: Sequence[int], draws: Sequence[Draw], ml_report_path: str) -> Dict[str, float]:
    features = ticket_features(ticket, draws)
    base = structural_score(features)
    reports = read_csv_rows(ml_report_path)
    weights = {}
    total_auc = 0.0
    for r in reports:
        model = str(r.get("model") or "unknown")
        auc = max(0.0, safe_float(r.get("auc"), 0.0) - 0.5)
        if auc > 0:
            weights[model] = auc
            total_auc += auc
    if not weights:
        weights = {"structural": 1.0}
        total_auc = 1.0
    norm = {k: v / total_auc for k, v in weights.items()}
    # Without serialized models, use report-derived model weights against shared robust feature score.
    ensemble = base * sum(norm.values())
    return {"meta_ensemble_score": ensemble, **{f"ensemble_weight_{k}": v for k, v in norm.items()}}


def diversity_select(rows: List[Dict[str, object]], count: int = 5, overlap_limit: int = 4) -> List[Dict[str, object]]:
    rows = sorted(rows, key=lambda r: safe_float(r.get("final_score")), reverse=True)
    selected: List[Dict[str, object]] = []
    for row in rows:
        nums = set(parse_nums(row.get("numbers")))
        if len(nums) != 7:
            continue
        if all(len(nums & set(parse_nums(prev.get("numbers")))) <= overlap_limit for prev in selected):
            selected.append(row)
        if len(selected) >= count:
            return selected
    for row in rows:
        if row not in selected:
            selected.append(row)
        if len(selected) >= count:
            break
    return selected


def champion_tournament(draws: Sequence[Draw], tickets: List[Tuple[int, ...]], out_dir: str, sims: int) -> Dict[str, object]:
    candidates = set(tickets)
    candidates.update(generate_candidates(draws, n=1500))
    candidates.update(mcts_construct(draws, iterations=1200))

    ml_report = str(Path(out_dir) / "ml_stack" / "ml_model_report.csv")
    rows: List[Dict[str, object]] = []
    for idx, ticket in enumerate(candidates):
        feats = ticket_features(ticket, draws)
        structural = structural_score(feats)
        ensemble = meta_ensemble_score(ticket, draws, ml_report)
        # lightweight MC for all, heavier only top later would be ideal. Keep bounded.
        mc = monte_carlo_score(ticket, draws, sims=max(500, sims // 20), seed=idx)
        roi_score = mc["mc_expected_roi"] * 10.0
        final = structural + ensemble["meta_ensemble_score"] + roi_score
        rows.append({
            "numbers": " ".join(f"{n:02d}" for n in ticket),
            "structural_score": round(structural, 8),
            "meta_ensemble_score": round(ensemble["meta_ensemble_score"], 8),
            "roi_score": round(roi_score, 8),
            "final_score": round(final, 8),
            **{k: round(v, 8) for k, v in feats.items()},
            **{k: round(v, 8) for k, v in mc.items()},
            **{k: round(v, 8) for k, v in ensemble.items() if k != "meta_ensemble_score"},
        })

    ranked = sorted(rows, key=lambda r: safe_float(r.get("final_score")), reverse=True)
    selected = diversity_select(ranked, 5, overlap_limit=4)
    write_csv(str(Path(out_dir) / "complete_ai_candidates.csv"), ranked[:1000])
    write_csv(str(Path(out_dir) / "complete_ai_prediction.csv"), selected)
    return {"candidate_count": len(rows), "selected": selected, "best": ranked[0] if ranked else None}


def roi_backtest(draws: Sequence[Draw], selected: List[Dict[str, object]], window: int = 100) -> Dict[str, object]:
    tickets = [parse_nums(r.get("numbers")) for r in selected]
    targets = draws[-window:] if len(draws) > window else draws
    total_cost = 0
    total_payout = 0
    rank_counts = {k: 0 for k in PAYOUT_ESTIMATE}
    max_match = 0
    for target in targets:
        for ticket in tickets:
            main, bonus, rank, payout = evaluate(ticket, target)
            rank_counts[rank] += 1
            total_cost += TICKET_COST
            total_payout += payout
            max_match = max(max_match, main)
    roi = (total_payout - total_cost) / total_cost if total_cost else 0.0
    return {"window": len(targets), "tickets_per_draw": len(tickets), "total_cost": total_cost, "total_payout": total_payout, "roi": roi, "max_match": max_match, "rank_counts": rank_counts}


def feature_store_export(draws: Sequence[Draw], out_dir: str) -> None:
    rows = []
    latest = draws[-1]
    for n in NUMBERS:
        freq60 = sum(1 for d in draws[-60:] if n in d.main)
        freq120 = sum(1 for d in draws[-120:] if n in d.main)
        freq240 = sum(1 for d in draws[-240:] if n in d.main)
        last_seen = next((i for i, d in enumerate(reversed(draws)) if n in d.main), len(draws))
        rows.append({"base_draw_no": latest.draw_no, "number": n, "freq60": freq60, "freq120": freq120, "freq240": freq240, "sleep": last_seen, "bayes": bayesian_update(draws).get(n, 0.0)})
    write_csv(str(Path(out_dir) / "feature_store_numbers.csv"), rows)


def make_dashboard(out_dir: str, result: Dict[str, object], roi: Dict[str, object], champions: List[Dict[str, object]]) -> None:
    path = Path(out_dir) / "complete_ai_dashboard.md"
    lines = [
        "# LOTO7 Complete AI Dashboard",
        "",
        f"- generated_at: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        f"- candidates: {result.get('candidate_count')}",
        f"- ROI window: {roi.get('window')}",
        f"- ROI: {roi.get('roi')}",
        f"- max_match: {roi.get('max_match')}",
        "",
        "## Selected 5 tickets",
    ]
    for i, row in enumerate(result.get("selected", []), start=1):
        lines.append(f"{i}. `{row.get('numbers')}` score={row.get('final_score')} expected_roi={row.get('mc_expected_roi')}")
    lines.extend(["", "## Master Champions"])
    for c in champions[:10]:
        lines.append(f"- {c.get('source_file')} id={c.get('id')} score={c.get('score')} generation={c.get('generation')}")
    lines.extend(["", "## ROI Backtest", "```json", json.dumps(roi, ensure_ascii=False, indent=2), "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_evolution_plan(champions: List[Dict[str, object]], roi: Dict[str, object]) -> Dict[str, object]:
    best_score = safe_float(champions[0].get("score")) if champions else 0.0
    roi_value = safe_float(roi.get("roi"))
    if roi_value < -0.75:
        next_mode = "explore_more_diversity"
        recommendation = {"population_multiplier": 1.25, "mutation_bias": "increase_diversity", "target_stride": 2}
    elif best_score > 0 and roi_value > -0.50:
        next_mode = "exploit_champion"
        recommendation = {"population_multiplier": 1.0, "mutation_bias": "around_champion", "target_stride": 1}
    else:
        next_mode = "balanced"
        recommendation = {"population_multiplier": 1.0, "mutation_bias": "balanced", "target_stride": 2}
    return {"next_mode": next_mode, "best_score": best_score, "roi": roi_value, "recommendation": recommendation}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="LOTO7 Complete AI System")
    ap.add_argument("--csv", default="loto7.csv")
    ap.add_argument("--output-dir", default="outputs/complete_ai")
    ap.add_argument("--monte-carlo-sims", type=int, default=20000)
    ap.add_argument("--roi-window", type=int, default=100)
    args = ap.parse_args(argv)

    out_dir = args.output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    draws = load_draws(args.csv)
    if len(draws) < 80:
        raise SystemExit("Not enough draws")

    feature_store_export(draws, out_dir)
    champions = load_champions()
    prediction_tickets = parse_prediction_files()
    result = champion_tournament(draws, prediction_tickets, out_dir, args.monte_carlo_sims)
    roi = roi_backtest(draws, result.get("selected", []), args.roi_window)
    plan = self_evolution_plan(champions, roi)
    payload = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "champions": champions[:20], "tournament": result, "roi_backtest": roi, "self_evolution_plan": plan}
    save_json(str(Path(out_dir) / "complete_ai_summary.json"), payload)
    make_dashboard(out_dir, result, roi, champions)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
