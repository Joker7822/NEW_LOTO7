#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loto7_self_evolution_engine.py

LOTO7 真の自己進化エンジン。

実装内容:
  1. 真の自己進化
     - complete_ai_summary / progress_summary / evolution_history を読み取り、次回の探索方針を自動生成
     - self_evolution_policy.json と self_evolution_next_config.json を更新
  2. Championの遺伝子交配
     - shard別 best_model を集約
     - 上位Champion同士を交叉・突然変異
     - breeder_seed_genomes.json を生成
  3. 分散学習統合
     - outputs/evolution_history_*.csv を統合
     - shard横断ランキング・安定性・世代別成績を生成
  4. 強化学習
     - 軽量Policy Gradient風の腕選択（explore / exploit / diversity / roi）
     - ROI・score改善・最大一致を報酬化して方策重みを更新

注意:
  ロト7はランダム抽せんであり、的中・利益を保証しない。
  ここでの強化学習は外部依存なしの軽量ポリシー更新であり、PPO等の本格RLではない。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ACTIONS = ["explore", "exploit", "diversity", "roi"]
GENOME_FLOAT_KEYS = [
    "full_weight", "recent240_weight", "recent120_weight", "recent60_weight",
    "pair_weight", "pair_recency_weight", "pair_stability_weight", "triple_weight",
    "dormancy_weight", "odd_bonus", "sum_bonus", "low_high_bonus", "consecutive_penalty",
]
GENOME_INT_KEYS = ["overlap_limit", "pool_size", "target_sum_min", "target_sum_max", "max_consecutive_pairs"]


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
        keys: List[str] = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: str, default=None):
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: str, data: Dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v, default=0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def softmax(weights: Dict[str, float]) -> Dict[str, float]:
    m = max(weights.values()) if weights else 0.0
    exps = {k: math.exp(v - m) for k, v in weights.items()}
    s = sum(exps.values()) or 1.0
    return {k: v / s for k, v in exps.items()}


def normalize_weight_block(genome: Dict[str, object]) -> Dict[str, object]:
    keys = ["full_weight", "recent240_weight", "recent120_weight", "recent60_weight"]
    vals = [max(0.001, safe_float(genome.get(k), 0.25)) for k in keys]
    s = sum(vals) or 1.0
    for k, v in zip(keys, vals):
        genome[k] = v / s
    return genome


def load_champion_genomes() -> List[Dict[str, object]]:
    champions = []
    for path in sorted(glob.glob("loto7_best_model*.json")):
        data = load_json(path, {}) or {}
        genome = data.get("genome", data)
        if not isinstance(genome, dict):
            continue
        genome = dict(genome)
        genome["source_file"] = path
        genome["score"] = safe_float(genome.get("score"), 0.0)
        champions.append(genome)
    champions.sort(key=lambda g: safe_float(g.get("score")), reverse=True)
    return champions


def merge_distributed_history(out_dir: str) -> Dict[str, object]:
    files = sorted(set(glob.glob("outputs/evolution_history_*.csv") + glob.glob("outputs/evolution_history.csv")))
    rows: List[Dict[str, object]] = []
    for path in files:
        for row in read_csv_rows(path):
            row = dict(row)
            row["source_file"] = path
            row["score"] = safe_float(row.get("score"))
            row["generation"] = safe_int(row.get("generation"), -1)
            row["max_main_match"] = safe_int(row.get("max_main_match"), 0)
            rows.append(row)

    rows.sort(key=lambda r: safe_float(r.get("score")), reverse=True)
    top_rows = rows[:500]
    write_csv(str(Path(out_dir) / "distributed_history_top500.csv"), top_rows)

    generation_stats: Dict[int, Dict[str, float]] = {}
    for row in rows:
        gen = safe_int(row.get("generation"), -1)
        if gen < 0:
            continue
        st = generation_stats.setdefault(gen, {"count": 0, "score_sum": 0.0, "score_max": -1e18, "max_match": 0})
        st["count"] += 1
        st["score_sum"] += safe_float(row.get("score"))
        st["score_max"] = max(st["score_max"], safe_float(row.get("score")))
        st["max_match"] = max(st["max_match"], safe_int(row.get("max_main_match"), 0))

    gen_rows = []
    for gen, st in sorted(generation_stats.items()):
        gen_rows.append({
            "generation": gen,
            "count": int(st["count"]),
            "score_avg": st["score_sum"] / max(1, st["count"]),
            "score_max": st["score_max"],
            "max_match": int(st["max_match"]),
        })
    write_csv(str(Path(out_dir) / "distributed_generation_stats.csv"), gen_rows)
    return {"history_files": files, "row_count": len(rows), "top_score": safe_float(rows[0].get("score")) if rows else 0.0, "max_generation": max([safe_int(r.get("generation"), -1) for r in rows], default=-1), "generation_stats": gen_rows[-20:]}


def crossover(a: Dict[str, object], b: Dict[str, object], rng: random.Random, idx: int) -> Dict[str, object]:
    child: Dict[str, object] = {}
    for k in GENOME_FLOAT_KEYS:
        av = safe_float(a.get(k), 0.0)
        bv = safe_float(b.get(k), 0.0)
        base = av if rng.random() < 0.5 else bv
        blend = (av + bv) / 2.0
        v = base * 0.65 + blend * 0.35
        v += rng.gauss(0, max(0.005, abs(v) * 0.08))
        child[k] = max(0.0, v)
    for k in GENOME_INT_KEYS:
        av = safe_int(a.get(k), 0)
        bv = safe_int(b.get(k), 0)
        v = av if rng.random() < 0.5 else bv
        if rng.random() < 0.35:
            v += rng.choice([-2, -1, 1, 2])
        child[k] = v

    child["overlap_limit"] = max(3, min(6, safe_int(child.get("overlap_limit"), 5)))
    child["pool_size"] = max(12, min(30, safe_int(child.get("pool_size"), 20)))
    child["target_sum_min"] = max(60, min(130, safe_int(child.get("target_sum_min"), 85)))
    child["target_sum_max"] = max(130, min(220, safe_int(child.get("target_sum_max"), 190)))
    if child["target_sum_min"] >= child["target_sum_max"]:
        child["target_sum_min"] = 85
        child["target_sum_max"] = 190
    child["max_consecutive_pairs"] = max(0, min(4, safe_int(child.get("max_consecutive_pairs"), 2)))
    child["id"] = f"breed_{idx:04d}_{rng.randint(1000, 9999)}"
    child["generation"] = max(safe_int(a.get("generation"), 0), safe_int(b.get("generation"), 0)) + 1
    child["score"] = 0.0
    child["parents"] = [a.get("id"), b.get("id")]
    return normalize_weight_block(child)


def breed_champions(champions: List[Dict[str, object]], out_dir: str, children: int, seed: int = 777) -> Dict[str, object]:
    rng = random.Random(seed)
    parents = champions[: max(2, min(16, len(champions)))]
    bred: List[Dict[str, object]] = []
    if len(parents) >= 2:
        for i in range(children):
            a = rng.choice(parents[: max(2, min(8, len(parents)))])
            b = rng.choice(parents)
            if a is b and len(parents) > 1:
                b = parents[(parents.index(a) + 1) % len(parents)]
            bred.append(crossover(a, b, rng, i))
    save_json(str(Path(out_dir) / "breeder_seed_genomes.json"), {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "parent_count": len(parents), "children": bred})
    write_csv(str(Path(out_dir) / "breeder_seed_genomes.csv"), bred)
    return {"parent_count": len(parents), "child_count": len(bred), "top_parent_score": safe_float(parents[0].get("score")) if parents else 0.0}


def compute_reward(distributed: Dict[str, object], complete_summary: Dict[str, object]) -> Dict[str, float]:
    top_score = safe_float(distributed.get("top_score"), 0.0)
    roi = safe_float(((complete_summary.get("roi_backtest") or {}) if isinstance(complete_summary, dict) else {}).get("roi"), -1.0)
    max_match = safe_float(((complete_summary.get("roi_backtest") or {}) if isinstance(complete_summary, dict) else {}).get("max_match"), 0.0)
    # scale to stable ranges
    score_reward = math.tanh(top_score / 25000.0)
    roi_reward = max(-1.0, min(1.0, roi))
    match_reward = max_match / 7.0
    total = 0.50 * score_reward + 0.30 * roi_reward + 0.20 * match_reward
    return {"total_reward": total, "score_reward": score_reward, "roi_reward": roi_reward, "match_reward": match_reward, "top_score": top_score, "roi": roi, "max_match": max_match}


def update_policy(out_dir: str, reward: Dict[str, float]) -> Dict[str, object]:
    path = Path(out_dir) / "self_evolution_policy.json"
    policy = load_json(str(path), None)
    if not policy:
        policy = {"weights": {a: 0.0 for a in ACTIONS}, "counts": {a: 0 for a in ACTIONS}, "history": []}

    weights = {a: safe_float((policy.get("weights") or {}).get(a), 0.0) for a in ACTIONS}
    probs = softmax(weights)

    # choose pseudo action from observed reward pattern
    if reward["roi_reward"] > -0.45:
        action = "roi"
    elif reward["match_reward"] >= 0.75:
        action = "exploit"
    elif reward["score_reward"] < 0.65:
        action = "explore"
    else:
        action = "diversity"

    lr = 0.12
    baseline = sum(probs[a] * weights[a] for a in ACTIONS) / max(1, len(ACTIONS))
    advantage = reward["total_reward"] - baseline
    for a in ACTIONS:
        grad = (1.0 if a == action else 0.0) - probs[a]
        weights[a] = weights[a] + lr * advantage * grad

    counts = {a: safe_int((policy.get("counts") or {}).get(a), 0) for a in ACTIONS}
    counts[action] += 1
    history = list(policy.get("history") or [])[-100:]
    history.append({"time": dt.datetime.now(dt.timezone.utc).isoformat(), "action": action, **reward, "probs_before": probs})
    updated = {"weights": weights, "probs": softmax(weights), "counts": counts, "last_action": action, "last_reward": reward, "history": history}
    save_json(str(path), updated)
    return updated


def next_config(policy: Dict[str, object], reward: Dict[str, float], bred: Dict[str, object]) -> Dict[str, object]:
    probs = policy.get("probs") or {}
    best_action = max(ACTIONS, key=lambda a: safe_float(probs.get(a), 0.0))
    if best_action == "explore":
        cfg = {"evolution_mode": "custom", "generations": 120, "population": 320, "max_targets": 240, "target_stride": 2, "mutation_mode": "wide"}
    elif best_action == "exploit":
        cfg = {"evolution_mode": "custom", "generations": 80, "population": 240, "max_targets": "all", "target_stride": 1, "mutation_mode": "champion_local"}
    elif best_action == "roi":
        cfg = {"evolution_mode": "custom", "generations": 100, "population": 260, "max_targets": "all", "target_stride": 2, "mutation_mode": "roi_weighted"}
    else:
        cfg = {"evolution_mode": "recommended", "generations": 100, "population": 240, "max_targets": 240, "target_stride": 2, "mutation_mode": "diversity"}
    cfg.update({"seed_genome_file": "outputs/self_evolution/breeder_seed_genomes.json", "bred_child_count": bred.get("child_count"), "policy_action": best_action, "reward": reward})
    return cfg


def make_report(out_dir: str, distributed: Dict[str, object], bred: Dict[str, object], policy: Dict[str, object], config: Dict[str, object]) -> None:
    lines = [
        "# LOTO7 Self Evolution Report",
        "",
        f"generated_at: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        "",
        "## Distributed Integration",
        f"- history rows: {distributed.get('row_count')}",
        f"- top score: {distributed.get('top_score')}",
        f"- max generation: {distributed.get('max_generation')}",
        "",
        "## Champion Breeding",
        f"- parent count: {bred.get('parent_count')}",
        f"- child count: {bred.get('child_count')}",
        f"- top parent score: {bred.get('top_parent_score')}",
        "",
        "## Reinforcement Policy",
        f"- last action: {policy.get('last_action')}",
        f"- probabilities: `{json.dumps(policy.get('probs'), ensure_ascii=False)}`",
        f"- reward: `{json.dumps(policy.get('last_reward'), ensure_ascii=False)}`",
        "",
        "## Next Config",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2),
        "```",
    ]
    Path(out_dir, "self_evolution_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="LOTO7 self evolution engine")
    ap.add_argument("--output-dir", default="outputs/self_evolution")
    ap.add_argument("--children", type=int, default=64)
    args = ap.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    distributed = merge_distributed_history(str(out))
    champions = load_champion_genomes()
    bred = breed_champions(champions, str(out), args.children)
    complete_summary = load_json("outputs/complete_ai/complete_ai_summary.json", {}) or {}
    reward = compute_reward(distributed, complete_summary)
    policy = update_policy(str(out), reward)
    cfg = next_config(policy, reward, bred)
    save_json(str(out / "self_evolution_next_config.json"), cfg)
    make_report(str(out), distributed, bred, policy, cfg)
    summary = {"distributed": distributed, "breeding": bred, "reward": reward, "policy": policy, "next_config": cfg}
    save_json(str(out / "self_evolution_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
