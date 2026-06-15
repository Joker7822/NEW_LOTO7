#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loto7_evolution_trainer.py

LOTO7用の進化型AutoMLトレーナー。

目的:
  - 固定ルールではなく、候補モデルの重み・制約を世代ごとに変異させる
  - 各候補を未来リークなし walk-forward バックテストで評価する
  - 成績の良い候補だけを残し、交叉・突然変異で次世代を作る
  - 最良モデルを loto7_best_model.json と outputs/evolution_history.csv に保存する

注意:
  宝くじ抽せんはランダム性が高く、的中・利益は保証しない。
  このスクリプトは「過去検証で相対的に良い戦略」を探索するためのもの。

標準ライブラリのみで動作。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

NUMBERS = tuple(range(1, 38))
RANK_ORDER = ["1等", "2等", "3等", "4等", "5等", "6等", "外れ"]


@dataclass(frozen=True)
class Draw:
    draw_no: int
    date: str
    main: Tuple[int, ...]
    bonus: Tuple[int, ...]


@dataclass
class Genome:
    id: str
    generation: int
    full_weight: float
    recent240_weight: float
    recent120_weight: float
    recent60_weight: float
    pair_weight: float
    pair_recency_weight: float
    pair_stability_weight: float
    triple_weight: float
    dormancy_weight: float
    odd_bonus: float
    sum_bonus: float
    low_high_bonus: float
    consecutive_penalty: float
    overlap_limit: int
    pool_size: int
    target_sum_min: int
    target_sum_max: int
    max_consecutive_pairs: int
    score: float = 0.0
    max_main_match: int = 0
    best_rank_count: int = 0


def parse_nums(text: object) -> Tuple[int, ...]:
    return tuple(int(x) for x in str(text or "").replace(",", " ").split() if x.isdigit())


def draw_no_int(text: object) -> Optional[int]:
    import re

    m = re.search(r"\d+", str(text or ""))
    return int(m.group(0)) if m else None


def load_draws(csv_path: str) -> List[Draw]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    draws: List[Draw] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            draw_no = draw_no_int(row.get("回別"))
            main = parse_nums(row.get("本数字"))
            bonus = parse_nums(row.get("ボーナス数字"))
            date = str(row.get("抽せん日") or "").strip()
            if draw_no is None or len(main) != 7 or len(set(main)) != 7:
                continue
            if len(bonus) != 2:
                bonus = tuple()
            if any(n < 1 or n > 37 for n in main + bonus):
                continue
            draws.append(Draw(draw_no=draw_no, date=date, main=tuple(sorted(main)), bonus=tuple(sorted(bonus))))
    draws.sort(key=lambda d: d.draw_no)
    return draws


def normalize_weights(values: Sequence[float]) -> Tuple[float, ...]:
    clipped = [max(0.01, float(v)) for v in values]
    s = sum(clipped)
    return tuple(v / s for v in clipped)


def random_genome(generation: int, idx: int, rng: random.Random) -> Genome:
    w = normalize_weights([rng.random(), rng.random(), rng.random(), rng.random()])
    return Genome(
        id=f"g{generation:03d}_{idx:04d}_{rng.randint(1000, 9999)}",
        generation=generation,
        full_weight=w[0],
        recent240_weight=w[1],
        recent120_weight=w[2],
        recent60_weight=w[3],
        pair_weight=rng.uniform(0.02, 0.20),
        pair_recency_weight=rng.uniform(0.02, 0.25),
        pair_stability_weight=rng.uniform(0.02, 0.20),
        triple_weight=rng.uniform(0.00, 0.10),
        dormancy_weight=rng.uniform(0.005, 0.050),
        odd_bonus=rng.uniform(0.10, 0.65),
        sum_bonus=rng.uniform(0.10, 0.65),
        low_high_bonus=rng.uniform(0.05, 0.45),
        consecutive_penalty=rng.uniform(0.10, 0.60),
        overlap_limit=rng.choice([4, 5, 5, 6]),
        pool_size=rng.randint(14, 22),
        target_sum_min=rng.randint(80, 105),
        target_sum_max=rng.randint(160, 190),
        max_consecutive_pairs=rng.choice([1, 1, 2]),
    )


def mutate_value(value: float, rng: random.Random, scale: float, low: float, high: float) -> float:
    return min(high, max(low, value + rng.gauss(0, scale)))


def crossover(parent_a: Genome, parent_b: Genome, generation: int, idx: int, rng: random.Random) -> Genome:
    data: Dict[str, object] = {}
    for key in asdict(parent_a).keys():
        if key in {"id", "generation", "score", "max_main_match", "best_rank_count"}:
            continue
        data[key] = getattr(parent_a if rng.random() < 0.5 else parent_b, key)

    w = normalize_weights([
        float(data["full_weight"]),
        float(data["recent240_weight"]),
        float(data["recent120_weight"]),
        float(data["recent60_weight"]),
    ])
    genome = Genome(
        id=f"g{generation:03d}_{idx:04d}_{rng.randint(1000, 9999)}",
        generation=generation,
        full_weight=w[0],
        recent240_weight=w[1],
        recent120_weight=w[2],
        recent60_weight=w[3],
        pair_weight=float(data["pair_weight"]),
        pair_recency_weight=float(data["pair_recency_weight"]),
        pair_stability_weight=float(data["pair_stability_weight"]),
        triple_weight=float(data["triple_weight"]),
        dormancy_weight=float(data["dormancy_weight"]),
        odd_bonus=float(data["odd_bonus"]),
        sum_bonus=float(data["sum_bonus"]),
        low_high_bonus=float(data["low_high_bonus"]),
        consecutive_penalty=float(data["consecutive_penalty"]),
        overlap_limit=int(data["overlap_limit"]),
        pool_size=int(data["pool_size"]),
        target_sum_min=int(data["target_sum_min"]),
        target_sum_max=int(data["target_sum_max"]),
        max_consecutive_pairs=int(data["max_consecutive_pairs"]),
    )
    return mutate(genome, generation, idx, rng)


def mutate(genome: Genome, generation: int, idx: int, rng: random.Random) -> Genome:
    w = normalize_weights([
        mutate_value(genome.full_weight, rng, 0.08, 0.01, 1.0),
        mutate_value(genome.recent240_weight, rng, 0.08, 0.01, 1.0),
        mutate_value(genome.recent120_weight, rng, 0.08, 0.01, 1.0),
        mutate_value(genome.recent60_weight, rng, 0.08, 0.01, 1.0),
    ])
    out = Genome(
        id=f"g{generation:03d}_{idx:04d}_{rng.randint(1000, 9999)}",
        generation=generation,
        full_weight=w[0],
        recent240_weight=w[1],
        recent120_weight=w[2],
        recent60_weight=w[3],
        pair_weight=mutate_value(genome.pair_weight, rng, 0.025, 0.00, 0.35),
        pair_recency_weight=mutate_value(genome.pair_recency_weight, rng, 0.030, 0.00, 0.40),
        pair_stability_weight=mutate_value(genome.pair_stability_weight, rng, 0.025, 0.00, 0.35),
        triple_weight=mutate_value(genome.triple_weight, rng, 0.018, 0.00, 0.18),
        dormancy_weight=mutate_value(genome.dormancy_weight, rng, 0.008, 0.00, 0.08),
        odd_bonus=mutate_value(genome.odd_bonus, rng, 0.06, 0.00, 1.0),
        sum_bonus=mutate_value(genome.sum_bonus, rng, 0.06, 0.00, 1.0),
        low_high_bonus=mutate_value(genome.low_high_bonus, rng, 0.05, 0.00, 0.8),
        consecutive_penalty=mutate_value(genome.consecutive_penalty, rng, 0.06, 0.00, 1.0),
        overlap_limit=max(4, min(6, genome.overlap_limit + rng.choice([-1, 0, 0, 1]))),
        pool_size=max(12, min(26, genome.pool_size + rng.choice([-2, -1, 0, 1, 2]))),
        target_sum_min=max(65, min(120, genome.target_sum_min + rng.choice([-5, -3, 0, 3, 5]))),
        target_sum_max=max(140, min(210, genome.target_sum_max + rng.choice([-5, -3, 0, 3, 5]))),
        max_consecutive_pairs=max(0, min(3, genome.max_consecutive_pairs + rng.choice([-1, 0, 0, 1]))),
    )
    if out.target_sum_min >= out.target_sum_max:
        out.target_sum_min, out.target_sum_max = 90, 175
    return out


def number_scores(draws: Sequence[Draw], decay: float, dormancy_weight: float) -> Dict[int, float]:
    score = {n: 0.0 for n in NUMBERS}
    last_seen = {n: -1 for n in NUMBERS}
    total = len(draws)
    for idx, draw in enumerate(draws):
        age = total - idx - 1
        weight = decay ** age
        for n in draw.main:
            score[n] += weight
            last_seen[n] = idx
    for n in NUMBERS:
        gap = total - last_seen[n] - 1 if last_seen[n] >= 0 else total
        score[n] += min(gap, 40) * dormancy_weight
    return score


def pair_scores(draws: Sequence[Draw]) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    freq: Dict[Tuple[int, int], float] = {}
    recency: Dict[Tuple[int, int], float] = {}
    stability: Dict[Tuple[int, int], float] = {}
    total = len(draws)
    if not draws:
        return freq, recency, stability

    # windows for stability: pair that appears across multiple recent windows is preferred.
    windows = [60, 120, 240]
    window_sets: Dict[int, set] = {}
    for w in windows:
        subset = draws[-w:] if len(draws) > w else draws
        s = set()
        for draw in subset:
            for a, b in itertools.combinations(draw.main, 2):
                s.add((min(a, b), max(a, b)))
        window_sets[w] = s

    for idx, draw in enumerate(draws):
        age = total - idx - 1
        f_weight = 0.992 ** age
        r_weight = 0.970 ** age
        for a, b in itertools.combinations(draw.main, 2):
            key = (min(a, b), max(a, b))
            freq[key] = freq.get(key, 0.0) + f_weight
            recency[key] = recency.get(key, 0.0) + r_weight

    all_pairs = set(freq) | set(recency)
    for key in all_pairs:
        stability[key] = sum(1.0 for w in windows if key in window_sets[w]) / len(windows)
    return freq, recency, stability


def triple_scores(draws: Sequence[Draw]) -> Dict[Tuple[int, int, int], float]:
    out: Dict[Tuple[int, int, int], float] = {}
    total = len(draws)
    for idx, draw in enumerate(draws):
        age = total - idx - 1
        weight = 0.990 ** age
        for combo in itertools.combinations(draw.main, 3):
            out[tuple(sorted(combo))] = out.get(tuple(sorted(combo)), 0.0) + weight
    return out


def blend_number_scores(train: Sequence[Draw], genome: Genome) -> Dict[int, float]:
    full = number_scores(train, 0.986, genome.dormancy_weight)
    r240 = number_scores(train[-240:], 0.982, genome.dormancy_weight) if len(train) >= 2 else full
    r120 = number_scores(train[-120:], 0.976, genome.dormancy_weight) if len(train) >= 2 else full
    r60 = number_scores(train[-60:], 0.965, genome.dormancy_weight) if len(train) >= 2 else full
    out: Dict[int, float] = {}
    for n in NUMBERS:
        out[n] = (
            full[n] * genome.full_weight
            + r240[n] * genome.recent240_weight
            + r120[n] * genome.recent120_weight
            + r60[n] * genome.recent60_weight
        )
    return out


def structural_score(combo: Sequence[int], genome: Genome) -> float:
    odd = sum(1 for n in combo if n % 2)
    low = sum(1 for n in combo if n <= 18)
    total = sum(combo)
    consecutive_pairs = sum(1 for a, b in zip(combo, combo[1:]) if b == a + 1)

    score = 0.0
    if odd in (3, 4):
        score += genome.odd_bonus
    else:
        score -= genome.odd_bonus * 0.70

    if low in (3, 4):
        score += genome.low_high_bonus
    else:
        score -= genome.low_high_bonus * 0.70

    if genome.target_sum_min <= total <= genome.target_sum_max:
        score += genome.sum_bonus
    else:
        # distance-based penalty
        if total < genome.target_sum_min:
            dist = genome.target_sum_min - total
        else:
            dist = total - genome.target_sum_max
        score -= genome.sum_bonus * min(2.0, dist / 20.0)

    if consecutive_pairs <= genome.max_consecutive_pairs:
        score += 0.10
    else:
        score -= genome.consecutive_penalty * (consecutive_pairs - genome.max_consecutive_pairs)
    return score


def combo_score(
    combo: Sequence[int],
    number_score: Dict[int, float],
    pair_freq: Dict[Tuple[int, int], float],
    pair_recency: Dict[Tuple[int, int], float],
    pair_stability: Dict[Tuple[int, int], float],
    triple: Dict[Tuple[int, int, int], float],
    genome: Genome,
) -> float:
    score = sum(number_score[n] for n in combo)
    for a, b in itertools.combinations(combo, 2):
        key = (min(a, b), max(a, b))
        score += pair_freq.get(key, 0.0) * genome.pair_weight
        score += pair_recency.get(key, 0.0) * genome.pair_recency_weight
        score += pair_stability.get(key, 0.0) * genome.pair_stability_weight
    for tri in itertools.combinations(combo, 3):
        score += triple.get(tuple(sorted(tri)), 0.0) * genome.triple_weight
    score += structural_score(combo, genome)
    return score


def generate_tickets(train: Sequence[Draw], genome: Genome, purchase_count: int) -> List[Tuple[int, ...]]:
    number_score = blend_number_scores(train, genome)
    pair_freq, pair_recency, pair_stability = pair_scores(train[-240:] if len(train) > 240 else train)
    triple = triple_scores(train[-180:] if len(train) > 180 else train)
    pool = [n for n, _ in sorted(number_score.items(), key=lambda kv: kv[1], reverse=True)[: genome.pool_size]]
    pool = sorted(pool)

    scored: List[Tuple[float, Tuple[int, ...]]] = []
    for combo in itertools.combinations(pool, 7):
        scored.append((combo_score(combo, number_score, pair_freq, pair_recency, pair_stability, triple, genome), tuple(combo)))
    scored.sort(reverse=True, key=lambda x: x[0])

    selected: List[Tuple[int, ...]] = []
    for _, combo in scored:
        if all(len(set(combo) & set(prev)) <= genome.overlap_limit for prev in selected):
            selected.append(combo)
        if len(selected) >= purchase_count:
            break
    if len(selected) < purchase_count:
        for _, combo in scored:
            if combo not in selected:
                selected.append(combo)
            if len(selected) >= purchase_count:
                break
    return selected[:purchase_count]


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


def evaluate_ticket(ticket: Sequence[int], target: Draw) -> Tuple[int, int, str]:
    s = set(ticket)
    main_match = len(s & set(target.main))
    bonus_match = len(s & set(target.bonus)) if target.bonus else 0
    return main_match, bonus_match, prize_rank(main_match, bonus_match)


def rank_score(rank: str, main_match: int, bonus_match: int) -> float:
    # 3等以上を強く、4等/5等/6等も探索シグナルとして加点。
    if rank == "1等":
        return 20000.0
    if rank == "2等":
        return 8000.0
    if rank == "3等":
        return 3500.0
    if rank == "4等":
        return 650.0
    if rank == "5等":
        return 90.0
    if rank == "6等":
        return 30.0
    return main_match * 2.0 + bonus_match * 0.5


def evaluate_genome(
    genome: Genome,
    draws: Sequence[Draw],
    purchase_count: int,
    min_train_draws: int,
    max_targets: Optional[int],
    target_stride: int,
) -> Tuple[Genome, Dict[str, object]]:
    target_indices = list(range(min_train_draws, len(draws), max(1, target_stride)))
    if max_targets is not None:
        target_indices = target_indices[-max_targets:]

    rank_counts = {rank: 0 for rank in RANK_ORDER}
    total_score = 0.0
    max_main_match = 0
    tickets_total = 0
    targets_total = 0

    for idx in target_indices:
        target = draws[idx]
        train = draws[:idx]
        tickets = generate_tickets(train, genome, purchase_count)
        targets_total += 1
        for ticket in tickets:
            main_match, bonus_match, rank = evaluate_ticket(ticket, target)
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
            total_score += rank_score(rank, main_match, bonus_match)
            max_main_match = max(max_main_match, main_match)
            tickets_total += 1

    # 過学習/偏り対策: 高得点に対して、外れ過多と極端制約を軽く減点。
    miss_rate = rank_counts.get("外れ", 0) / tickets_total if tickets_total else 1.0
    regularization = 0.0
    regularization += abs(genome.recent60_weight - genome.full_weight) * 2.0
    regularization += max(0, genome.pool_size - 22) * 0.5
    regularization += max(0, 14 - genome.pool_size) * 0.5
    final_score = total_score - miss_rate * 10.0 - regularization

    genome.score = final_score
    genome.max_main_match = max_main_match
    genome.best_rank_count = sum(rank_counts.get(r, 0) for r in ["1等", "2等", "3等", "4等"])
    stats: Dict[str, object] = {
        "genome_id": genome.id,
        "generation": genome.generation,
        "score": round(final_score, 6),
        "targets": targets_total,
        "tickets": tickets_total,
        "max_main_match": max_main_match,
        **{f"rank_{rank}": rank_counts.get(rank, 0) for rank in RANK_ORDER},
    }
    return genome, stats


def write_csv(path: str, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]], append: bool = False) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    exists = p.exists() and p.stat().st_size > 0
    mode = "a" if append else "w"
    with p.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not append or not exists:
            writer.writeheader()
        writer.writerows(rows)


def save_json(path: str, payload: Dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def load_best_model(path: str) -> Optional[Genome]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        genome_data = data.get("genome", data)
        return Genome(**{k: genome_data[k] for k in Genome.__dataclass_fields__.keys() if k in genome_data})
    except Exception:
        return None


def git_commit_push(message: str, paths: Sequence[str], retries: int = 3) -> bool:
    if os.environ.get("DISABLE_GIT_PUSH", "").lower() in {"1", "true", "yes"}:
        print("[GIT] DISABLE_GIT_PUSH is set. skip.")
        return False
    if not Path(".git").exists():
        print("[GIT] .git not found. skip.")
        return False
    existing = [p for p in paths if Path(p).exists()]
    if not existing:
        return False
    try:
        subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=False)
        subprocess.run(["git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"], check=False)
        subprocess.run(["git", "add", *existing], check=True)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False)
        if diff.returncode == 0:
            print("[GIT] no changes.")
            return False
        subprocess.run(["git", "commit", "-m", message], check=True)
        for attempt in range(1, retries + 1):
            subprocess.run(["git", "pull", "--rebase", "--autostash"], check=False)
            push = subprocess.run(["git", "push"], check=False)
            if push.returncode == 0:
                return True
            time.sleep(2 * attempt)
    except Exception as exc:
        print(f"[GIT] failed: {exc}", file=sys.stderr)
    return False


def make_next_generation(evaluated: List[Genome], generation: int, population: int, elite_count: int, rng: random.Random) -> List[Genome]:
    evaluated = sorted(evaluated, key=lambda g: g.score, reverse=True)
    elites = evaluated[: max(1, elite_count)]
    next_pop: List[Genome] = []

    # Carry elites with new IDs/generation but identical parameters.
    for idx, elite in enumerate(elites):
        d = asdict(elite)
        d.update({"id": f"g{generation:03d}_elite_{idx:04d}", "generation": generation, "score": 0.0, "max_main_match": 0, "best_rank_count": 0})
        next_pop.append(Genome(**d))

    while len(next_pop) < population:
        if rng.random() < 0.20:
            next_pop.append(random_genome(generation, len(next_pop), rng))
            continue
        parent_a = rng.choice(elites)
        parent_b = rng.choice(evaluated[: max(len(elites), min(len(evaluated), elite_count * 3))])
        next_pop.append(crossover(parent_a, parent_b, generation, len(next_pop), rng))
    return next_pop[:population]


def predict_with_best(draws: Sequence[Draw], genome: Genome, purchase_count: int, output_path: str) -> None:
    tickets = generate_tickets(draws, genome, purchase_count)
    latest = draws[-1]
    rows = []
    for idx, ticket in enumerate(tickets, start=1):
        rows.append(
            {
                "base_latest_draw_no": latest.draw_no,
                "base_latest_date": latest.date,
                "prediction_draw_no": latest.draw_no + 1,
                "combo_index": idx,
                "numbers": " ".join(f"{n:02d}" for n in ticket),
                "model_id": genome.id,
                "model_score": round(genome.score, 6),
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )
    write_csv(output_path, ["base_latest_draw_no", "base_latest_date", "prediction_draw_no", "combo_index", "numbers", "model_id", "model_score", "created_at"], rows)



def parse_max_targets_runtime(value: object) -> Optional[int]:
    text = str(value or "").strip().lower()
    if text in {"", "none", "all", "null", "-1", "999999"}:
        return None
    return int(text)


def genome_to_dict(genome: Genome) -> Dict[str, object]:
    return asdict(genome)


def genome_from_dict(data: Dict[str, object]) -> Genome:
    fields = Genome.__dataclass_fields__.keys()
    return Genome(**{k: data[k] for k in fields if k in data})


def atomic_save_json(path: str, payload: Dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(p)


def load_json_file(path: str, default: object) -> object:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_evolution_state(
    state_path: str,
    generation: int,
    population: List[Genome],
    evaluated: List[Genome],
    completed_ids: Sequence[str],
    best: Optional[Genome],
    args: argparse.Namespace,
) -> None:
    atomic_save_json(
        state_path,
        {
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "generation": generation,
            "population": [genome_to_dict(g) for g in population],
            "evaluated": [genome_to_dict(g) for g in evaluated],
            "completed_ids": list(completed_ids),
            "best": genome_to_dict(best) if best else None,
            "args": {
                "generations": args.generations,
                "population": args.population,
                "elite_count": args.elite_count,
                "purchase_count": args.purchase_count,
                "min_train_draws": args.min_train_draws,
                "max_targets": args.max_targets,
                "target_stride": args.target_stride,
                "seed": args.seed,
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
                "max_runtime_minutes": args.max_runtime_minutes,
                "safe_exit_minutes": args.safe_exit_minutes,
            },
        },
    )


def load_evolution_state(state_path: str) -> Optional[Dict[str, object]]:
    state = load_json_file(state_path, None)
    if not isinstance(state, dict):
        return None
    if "generation" not in state or "population" not in state:
        return None
    return state


def should_safe_exit(start_time: float, args: argparse.Namespace) -> bool:
    max_runtime = max(1, int(args.max_runtime_minutes)) * 60
    safe_exit = max(0, int(args.safe_exit_minutes)) * 60
    return (time.time() - start_time) >= max(0, max_runtime - safe_exit)


def save_runtime_state(
    runtime_state_path: str,
    reason: str,
    generation: int,
    args: argparse.Namespace,
    start_time: float,
    best: Optional[Genome],
    state_path: str,
) -> None:
    elapsed = max(0.0, time.time() - start_time)
    atomic_save_json(
        runtime_state_path,
        {
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "reason": reason,
            "generation": generation,
            "elapsed_seconds": round(elapsed, 3),
            "elapsed_minutes": round(elapsed / 60.0, 3),
            "max_runtime_minutes": args.max_runtime_minutes,
            "safe_exit_minutes": args.safe_exit_minutes,
            "resume": True,
            "state_path": state_path,
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "best_id": best.id if best else None,
            "best_score": best.score if best else None,
        },
    )

def run_evolution(args: argparse.Namespace) -> int:
    start_time = time.time()
    rng = random.Random(args.seed)
    draws = load_draws(args.csv)
    if len(draws) <= args.min_train_draws + 5:
        raise SystemExit(f"draws too small: {len(draws)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_suffix = f"shard{args.shard_id:02d}_of_{args.num_shards:02d}"
    history_csv = str(out_dir / f"evolution_history_{shard_suffix}.csv")
    best_summary_csv = str(out_dir / f"evolution_best_summary_{shard_suffix}.csv")
    state_path = args.state_path or str(out_dir / f"evolution_state_{shard_suffix}.json")
    best_model_json = args.best_model if args.num_shards == 1 else str(Path(args.best_model).with_name(f"{Path(args.best_model).stem}_{shard_suffix}.json"))
    best_prediction_csv = str(out_dir / f"evolution_best_prediction_{shard_suffix}.csv")
    runtime_state_json = str(out_dir / f"evolution_runtime_state_{shard_suffix}.json")

    fieldnames = [
        "generation", "genome_id", "score", "targets", "tickets", "max_main_match",
        "rank_1等", "rank_2等", "rank_3等", "rank_4等", "rank_5等", "rank_6等", "rank_外れ",
        "full_weight", "recent240_weight", "recent120_weight", "recent60_weight", "pair_weight", "pair_recency_weight", "pair_stability_weight",
        "triple_weight", "dormancy_weight", "odd_bonus", "sum_bonus", "low_high_bonus", "consecutive_penalty",
        "overlap_limit", "pool_size", "target_sum_min", "target_sum_max", "max_consecutive_pairs",
        "shard_id", "num_shards", "completed_at",
    ]

    state = load_evolution_state(state_path) if args.resume else None
    if state:
        start_generation = int(state.get("generation", 0))
        population = [genome_from_dict(x) for x in state.get("population", [])]
        evaluated = [genome_from_dict(x) for x in state.get("evaluated", [])]
        completed_ids = set(str(x) for x in state.get("completed_ids", []))
        best_data = state.get("best")
        best = genome_from_dict(best_data) if isinstance(best_data, dict) else None
        print(f"[RESUME] generation={start_generation} completed={len(completed_ids)} shard={args.shard_id}/{args.num_shards}")
    else:
        start_generation = 0
        population = [random_genome(0, i, rng) for i in range(args.population)]
        previous_best = load_best_model(best_model_json)
        if previous_best is not None:
            previous_best.id = "g000_previous_best"
            previous_best.generation = 0
            population[0] = previous_best
        evaluated = []
        completed_ids = set()
        best = None
        save_evolution_state(state_path, start_generation, population, evaluated, completed_ids, best, args)

    max_targets = parse_max_targets_runtime(args.max_targets)

    for generation in range(start_generation, args.generations):
        if should_safe_exit(start_time, args):
            save_runtime_state(runtime_state_json, "safe_timeout_exit_before_generation", generation, args, start_time, best, state_path)
            save_evolution_state(state_path, generation, population, evaluated, completed_ids, best, args)
            git_commit_push(
                f"Safe timeout exit LOTO7 evolution shard {args.shard_id} generation {generation}",
                [history_csv, best_summary_csv, best_model_json, best_prediction_csv, state_path, runtime_state_json],
            )
            print(f"[SAFE EXIT] before generation={generation} shard={args.shard_id}/{args.num_shards}")
            return 0

        if generation != start_generation or not state:
            evaluated = []
            completed_ids = set()
            save_evolution_state(state_path, generation, population, evaluated, completed_ids, best, args)

        shard_population = [g for idx, g in enumerate(population) if idx % args.num_shards == args.shard_id]
        pending = [g for g in shard_population if g.id not in completed_ids]
        print(
            f"[EVOLVE] generation={generation} shard={args.shard_id}/{args.num_shards} "
            f"population_total={len(population)} shard_population={len(shard_population)} "
            f"pending={len(pending)} workers={args.workers}"
        )

        for genome in pending:
            evaluated_genome, stats = evaluate_genome(
                genome,
                draws=draws,
                purchase_count=args.purchase_count,
                min_train_draws=args.min_train_draws,
                max_targets=max_targets,
                target_stride=args.target_stride,
            )
            evaluated.append(evaluated_genome)
            completed_ids.add(evaluated_genome.id)

            row = dict(stats)
            row.update({k: v for k, v in asdict(evaluated_genome).items() if k not in {"id", "generation", "score", "max_main_match", "best_rank_count"}})
            row.update({"shard_id": args.shard_id, "num_shards": args.num_shards, "completed_at": dt.datetime.now(dt.timezone.utc).isoformat()})
            write_csv(history_csv, fieldnames, [row], append=True)

            if best is None or evaluated_genome.score > best.score:
                best = evaluated_genome
                save_json(
                    best_model_json,
                    {
                        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "csv": args.csv,
                        "purchase_count": args.purchase_count,
                        "min_train_draws": args.min_train_draws,
                        "max_targets": args.max_targets,
                        "target_stride": args.target_stride,
                        "shard_id": args.shard_id,
                        "num_shards": args.num_shards,
                        "genome": asdict(best),
                    },
                )
                predict_with_best(draws, best, args.purchase_count, best_prediction_csv)
                print(f"[BEST] generation={generation} shard={args.shard_id} score={best.score:.3f} id={best.id}")

            save_evolution_state(state_path, generation, population, evaluated, completed_ids, best, args)

            if args.push_every_genome > 0 and len(completed_ids) % args.push_every_genome == 0:
                git_commit_push(
                    f"Update LOTO7 evolution shard {args.shard_id} generation {generation} genome {len(completed_ids)}",
                    [history_csv, best_summary_csv, best_model_json, best_prediction_csv, state_path, runtime_state_json],
                )

            if should_safe_exit(start_time, args):
                save_runtime_state(runtime_state_json, "safe_timeout_exit_during_generation", generation, args, start_time, best, state_path)
                save_evolution_state(state_path, generation, population, evaluated, completed_ids, best, args)
                git_commit_push(
                    f"Safe timeout exit LOTO7 evolution shard {args.shard_id} generation {generation}",
                    [history_csv, best_summary_csv, best_model_json, best_prediction_csv, state_path, runtime_state_json],
                )
                print(
                    f"[SAFE EXIT] during generation={generation} shard={args.shard_id}/{args.num_shards} "
                    f"completed={len(completed_ids)}/{len(shard_population)}"
                )
                return 0

        evaluated_sorted = sorted(evaluated, key=lambda g: g.score, reverse=True)
        summary_rows = []
        for rank, genome in enumerate(evaluated_sorted[: args.elite_count], start=1):
            summary_rows.append(
                {
                    "rank": rank,
                    "generation": generation,
                    "genome_id": genome.id,
                    "score": round(genome.score, 6),
                    "max_main_match": genome.max_main_match,
                    "best_rank_count": genome.best_rank_count,
                    "shard_id": args.shard_id,
                    "num_shards": args.num_shards,
                    "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
            )
        write_csv(
            best_summary_csv,
            ["rank", "generation", "genome_id", "score", "max_main_match", "best_rank_count", "shard_id", "num_shards", "updated_at"],
            summary_rows,
        )

        if args.push_every_generation > 0 and (generation + 1) % args.push_every_generation == 0:
            git_commit_push(
                f"Update LOTO7 evolution shard {args.shard_id} generation {generation + 1}",
                [history_csv, best_summary_csv, best_model_json, best_prediction_csv, state_path, runtime_state_json],
            )

        if should_safe_exit(start_time, args):
            save_runtime_state(runtime_state_json, "safe_timeout_exit_after_generation", generation + 1, args, start_time, best, state_path)
            save_evolution_state(state_path, generation, population, evaluated, completed_ids, best, args)
            git_commit_push(
                f"Safe timeout exit LOTO7 evolution shard {args.shard_id} generation {generation + 1}",
                [history_csv, best_summary_csv, best_model_json, best_prediction_csv, state_path, runtime_state_json],
            )
            print(f"[SAFE EXIT] after generation={generation + 1} shard={args.shard_id}/{args.num_shards}")
            return 0

        if generation + 1 < args.generations:
            if evaluated_sorted:
                population = make_next_generation(evaluated_sorted, generation + 1, args.population, min(args.elite_count, len(evaluated_sorted)), rng)
            else:
                population = [random_genome(generation + 1, i, rng) for i in range(args.population)]
            evaluated = []
            completed_ids = set()
            save_evolution_state(state_path, generation + 1, population, evaluated, completed_ids, best, args)

    if best is not None:
        print(f"[FINAL BEST] shard={args.shard_id}/{args.num_shards} score={best.score:.3f} id={best.id}")
        print(json.dumps(asdict(best), ensure_ascii=False, indent=2))

    if args.push_final:
        git_commit_push("Update LOTO7 evolutionary best model", [history_csv, best_summary_csv, best_model_json, best_prediction_csv, state_path, runtime_state_json])

    return 0



def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LOTO7 evolutionary walk-forward trainer with resume/shards/adaptive timeout")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--best-model", default="loto7_best_model.json")
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=max(1, min(2, os.cpu_count() or 1)), help="互換用。現行版では逐次評価。")
    parser.add_argument("--resume", action="store_true", help="outputs/evolution_state_*.json から再開")
    parser.add_argument("--push-every-genome", type=int, default=0, help="N個体ごとにcommit/push。0で無効")
    parser.add_argument("--max-runtime-minutes", type=int, default=330, help="安全終了を含む最大実行時間。GitHub Actions 355分制限なら330推奨。")
    parser.add_argument("--safe-exit-minutes", type=int, default=20, help="最大実行時間の何分前に保存・pushして正常終了するか。")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--elite-count", type=int, default=10)
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--min-train-draws", type=int, default=60)
    parser.add_argument("--max-targets", default="all", help="評価対象回数。allなら全対象。")
    parser.add_argument("--target-stride", type=int, default=1, help="1なら全対象、2なら1回おきに評価。")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--push-every-generation", type=int, default=1)
    parser.add_argument("--push-final", action="store_true")
    args = parser.parse_args(argv)

    if args.population < 4:
        raise SystemExit("--population must be >= 4")
    if args.elite_count < 1 or args.elite_count >= args.population:
        raise SystemExit("--elite-count must be >=1 and < population")
    if args.purchase_count <= 0:
        raise SystemExit("--purchase-count must be positive")
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise SystemExit("--shard-id must satisfy 0 <= shard_id < num_shards")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.max_runtime_minutes <= 0:
        raise SystemExit("--max-runtime-minutes must be positive")
    if args.safe_exit_minutes < 0:
        raise SystemExit("--safe-exit-minutes must be >= 0")
    if args.safe_exit_minutes >= args.max_runtime_minutes:
        raise SystemExit("--safe-exit-minutes must be smaller than --max-runtime-minutes")
    return run_evolution(args)


if __name__ == "__main__":
    raise SystemExit(main())
