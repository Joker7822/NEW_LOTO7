#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rewrite the final TXT report with Generation 4 evidence and diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize the Generation 4 prediction report.")
    parser.add_argument("--prediction", default="outputs/evolution_best_prediction.csv")
    parser.add_argument("--summary", default="outputs/generation4/latest_generation4_summary.json")
    parser.add_argument("--report", default="outputs/holdout/latest_prediction_report.txt")
    args = parser.parse_args()

    rows = load_rows(Path(args.prediction))
    if not rows:
        raise SystemExit("prediction CSV is empty")
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    first = rows[0]
    portfolio = summary.get("generation4_portfolio", {}) if isinstance(summary.get("generation4_portfolio"), dict) else {}
    components = portfolio.get("objective_components", {}) if isinstance(portfolio.get("objective_components"), dict) else {}
    conformal = summary.get("conformal_number_pool", {}) if isinstance(summary.get("conformal_number_pool"), dict) else {}
    change = summary.get("change_point", {}) if isinstance(summary.get("change_point"), dict) else {}
    null_league = summary.get("null_strategy_league", {}) if isinstance(summary.get("null_strategy_league"), dict) else {}
    source_weights = summary.get("dynamic_source_weights", {}) if isinstance(summary.get("dynamic_source_weights"), dict) else {}

    lines = [
        "LOTO7 Generation 4 Latest Prediction",
        "====================================",
        "",
        f"基準最新回: 第{first.get('base_latest_draw_no', '')}回",
        f"基準最新抽せん日: {first.get('base_latest_date', '')}",
        f"予測対象回: 第{first.get('prediction_draw_no', '')}回",
        "予測方式: generation4_complete",
        "採用構成: Bayesian DMA + Change-Point + Rolling Conformal + DPP + Hypergraph",
        "selection_integrity: original_model_candidates / post-selection replacement=0",
        "",
        f"[最新予測 {len(rows)}口]",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"{index}位 / {index}口目: {row.get('numbers', '')} / "
            f"相対スコア={row.get('ensemble_score', '')} / "
            f"方式={row.get('prediction_method', '')}"
        )
    lines.extend([
        "",
        "[Generation 4 Evidence]",
        f"source_quotas: {summary.get('source_quotas')}",
        f"source_weights: {source_weights}",
        f"change_point_score: {change.get('score')} / level={change.get('level')}",
        f"conformal_pool_size: {conformal.get('pool_size')}",
        f"conformal_pool: {conformal.get('numbers')}",
        f"conformal_empirical_coverage: {conformal.get('empirical_main_number_coverage')}",
        f"null_league_passed: {null_league.get('passed')}",
        f"null_model_percentile: {null_league.get('model_percentile')}",
        f"null_pbo: {null_league.get('pbo')}",
        "",
        "[Portfolio Objective]",
        f"objective_score: {portfolio.get('objective_score')}",
        f"dpp_logdet: {components.get('dpp_logdet')}",
        f"hypergraph_score: {components.get('hypergraph')}",
        f"conformal_hits: {components.get('conformal')}",
        f"unique_number_count: {portfolio.get('unique_number_count')}",
        f"max_number_usage: {portfolio.get('max_number_usage')}",
        f"max_pair_overlap: {portfolio.get('max_pair_overlap')}",
        f"average_pair_overlap: {portfolio.get('average_pair_overlap')}",
        "",
        "[相対スコアの意味]",
        "0.95から順に統一した5口内の順位です。当せん確率ではありません。",
        "",
        "[出力ファイル]",
        "CSV: outputs/evolution_best_prediction.csv",
        "Generation 4 summary: outputs/generation4/latest_generation4_summary.json",
        "Shadow history: outputs/generation4/shadow_history.csv",
        "Sealed manifest: outputs/generation4/latest_sealed_manifest.json",
        "",
        "注意: 宝くじはランダム性が高く、過去検証・Conformal集合・e-processは将来の当せんや利益を保証しません。",
    ])
    text = "\n".join(lines) + "\n"
    for row in rows:
        if str(row.get("numbers") or "") not in text:
            raise SystemExit(f"report synchronization failed: {row.get('numbers')}")
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"[OK] generation4 report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
