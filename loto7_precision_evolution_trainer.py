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

注意:
  宝くじはランダム性が高く、将来の当せんや利益を保証するものではありません。
"""

from __future__ import annotations

import sys
from typing import Optional, List

import loto7_evolution_trainer as base


def high_grade_rank_score(rank: str, main_match: int, bonus_match: int) -> float:
    """4等以上・最大一致を強く評価する学習スコア。

    旧scoreは等級ごとの固定点が中心だった。
    新scoreでは、同じ外れでも本数字一致数を強く評価し、
    4等以上の探索圧を高める。
    """
    if rank == "1等":
        return 100000.0
    if rank == "2等":
        return 60000.0
    if rank == "3等":
        return 38000.0
    if rank == "4等":
        return 8500.0
    if rank == "5等":
        return 850.0
    if rank == "6等":
        return 420.0

    # 外れでも高一致に近い候補は強い探索シグナルとして残す。
    if main_match == 5:
        return 4200.0 + bonus_match * 250.0
    if main_match == 4:
        return 360.0 + bonus_match * 80.0
    if main_match == 3:
        return 55.0 + bonus_match * 45.0
    if main_match == 2:
        return 8.0 + bonus_match * 4.0
    return main_match * 1.5 + bonus_match * 0.75


def install_precision_scoring() -> None:
    base.rank_score = high_grade_rank_score


def main(argv: Optional[List[str]] = None) -> int:
    install_precision_scoring()
    print("[PRECISION] high-grade focused rank_score enabled", flush=True)
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
