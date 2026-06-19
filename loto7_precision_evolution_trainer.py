#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loto7_precision_evolution_trainer.py

既存の loto7_evolution_trainer.py を壊さずに、学習スコアだけを
4等以上・最大一致重視へ差し替えて実行するラッパー。

狙い:
  - 最終採用基準 holdout_balanced と学習時の評価基準を近づける
  - 本数字6個一致、5個一致、4個一致をより強く評価する
  - 5口全体の高一致を増やす方向へ進化させる
  - 旧スコアのresume stateと混ざらないよう、precision専用stateを使う
  - loto7_self_evolution_config.json の自動調整スコアを読み込む

注意:
  宝くじはランダム性が高く、将来の当せんや利益を保証するものではありません。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, List

import loto7_evolution_trainer as base

DEFAULT_PRECISION_SCORING: Dict[str, float] = {
    "1等": 100000.0,
    "2等": 60000.0,
    "3等": 38000.0,
    "4等": 8500.0,
    "5等": 850.0,
    "6等": 420.0,
    "near_miss_main5": 4200.0,
    "near_miss_main5_bonus": 250.0,
    "near_miss_main4": 360.0,
    "near_miss_main4_bonus": 80.0,
    "near_miss_main3": 55.0,
    "near_miss_main3_bonus": 45.0,
    "near_miss_main2": 8.0,
    "near_miss_main2_bonus": 4.0,
    "fallback_main": 1.5,
    "fallback_bonus": 0.75,
}

PRECISION_SCORING: Dict[str, float] = dict(DEFAULT_PRECISION_SCORING)


def load_precision_scoring() -> Dict[str, float]:
    path = Path(os.environ.get("LOTO7_SELF_EVOLUTION_CONFIG", "loto7_self_evolution_config.json"))
    scoring = dict(DEFAULT_PRECISION_SCORING)
    if not path.exists():
        return scoring
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data.get("precision_scoring", {})
        if not isinstance(raw, dict):
            return scoring
        for key, default in DEFAULT_PRECISION_SCORING.items():
            try:
                scoring[key] = float(raw.get(key, default))
            except Exception:
                scoring[key] = default
    except Exception as exc:
        print(f"[PRECISION] failed to read self evolution config: {path} ({exc})", flush=True)
    return scoring


def score_value(key: str) -> float:
    return float(PRECISION_SCORING.get(key, DEFAULT_PRECISION_SCORING[key]))


def high_grade_rank_score(rank: str, main_match: int, bonus_match: int) -> float:
    """4等以上・最大一致を強く評価する学習スコア。

    loto7_self_evolution_config.json の precision_scoring を読むため、
    自己進化AIが評価結果に応じてスコア配分を調整できる。
    """
    if rank in {"1等", "2等", "3等", "4等", "5等", "6等"}:
        return score_value(rank)

    # 外れでも高一致に近い候補は強い探索シグナルとして残す。
    if main_match == 5:
        return score_value("near_miss_main5") + bonus_match * score_value("near_miss_main5_bonus")
    if main_match == 4:
        return score_value("near_miss_main4") + bonus_match * score_value("near_miss_main4_bonus")
    if main_match == 3:
        return score_value("near_miss_main3") + bonus_match * score_value("near_miss_main3_bonus")
    if main_match == 2:
        return score_value("near_miss_main2") + bonus_match * score_value("near_miss_main2_bonus")
    return main_match * score_value("fallback_main") + bonus_match * score_value("fallback_bonus")


def install_precision_scoring() -> None:
    global PRECISION_SCORING
    PRECISION_SCORING = load_precision_scoring()
    base.rank_score = high_grade_rank_score


def arg_value(argv: List[str], name: str, default: str) -> str:
    if name not in argv:
        return default
    idx = argv.index(name)
    if idx + 1 >= len(argv):
        return default
    return argv[idx + 1]


def inject_precision_state_path(argv: Optional[List[str]]) -> List[str]:
    args = list(argv or [])
    if "--state-path" in args:
        return args
    shard_id = int(arg_value(args, "--shard-id", "0"))
    num_shards = int(arg_value(args, "--num-shards", "1"))
    state_path = f"outputs/evolution_state_precision_shard{shard_id:02d}_of_{num_shards:02d}.json"
    return args + ["--state-path", state_path]


def main(argv: Optional[List[str]] = None) -> int:
    install_precision_scoring()
    patched_argv = inject_precision_state_path(argv)
    print("[PRECISION] high-grade focused rank_score enabled", flush=True)
    print("[PRECISION] self evolution config connected", flush=True)
    print("[PRECISION] isolated precision resume state enabled", flush=True)
    return base.main(patched_argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
