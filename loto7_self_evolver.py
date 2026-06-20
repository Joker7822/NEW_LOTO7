#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loto7_self_evolver.py

LOTO7自己進化AIコントローラ。

Lv1: 既存の進化型モデル学習を前提に評価結果を読む
Lv2: 評価基準・precision scoringを自動調整する
Lv3: 弱点診断と改善案を自動生成する
Lv4: 安全な設定変更として改善候補を自動適用する
Lv5: workflow側でsmoke検証できる出力を作る
Lv6: workflow側で改善ブランチ/PR化できるレポートを作る

重要:
  - このスクリプトはmainへ直接pushしない
  - 任意コード生成はしない。変更対象は安全なJSON設定に限定する
  - 宝くじの当せんや利益を保証しない
  - 実質差分がない重複PRを作らない
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_CONFIG: Dict[str, Any] = {
    "version": 1,
    "profile": "precision_high_grade_v1",
    "safety": {
        "allow_main_direct_push": False,
        "auto_create_pull_request": True,
        "max_config_change_ratio": 0.25,
        "require_smoke_test": True,
        "require_positive_or_neutral_regression_check": True,
    },
    "targets": {
        "roi_percent": 40.0,
        "high_grade_hit_count": 8,
        "max_main_match": 6,
        "grade_hit_count": 155,
        "min_total_target_draws": 600,
    },
    "precision_scoring": {
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
    },
    "ensemble": {
        "prediction_mode": "ensemble",
        "candidates_per_model": 10,
        "overlap_limit": 4,
        "roi_weight": 2.0,
        "max_match_weight": 0.15,
        "high_grade_weight": 0.08,
        "fifth_weight": 0.004,
        "sixth_weight": 0.002,
        "consensus_bonus": 0.012,
        "diversity_bonus": 0.03,
    },
    "adoption_rules": {
        "min_roi_delta_percent": 3.0,
        "min_high_grade_delta": 0,
        "min_max_main_match_delta": 0,
        "allow_roi_drop_if_high_grade_improves": False,
    },
}

VOLATILE_CONFIG_KEYS = {"last_self_evolved_at"}
REPORT_FILES = [
    "diagnosis.json",
    "proposal.json",
    "adoption_decision.json",
    "comparison_report.txt",
    "applied_config.json",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def normalize_config_for_diff(config: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(config)
    for key in VOLATILE_CONFIG_KEYS:
        out.pop(key, None)
    return out


def has_substantive_config_change(base_config: Dict[str, Any], proposed_config: Dict[str, Any]) -> bool:
    return normalize_config_for_diff(base_config) != normalize_config_for_diff(proposed_config)


def changed_config_paths(base_config: Dict[str, Any], proposed_config: Dict[str, Any]) -> List[str]:
    changes: List[str] = []
    for key in sorted(set(base_config.keys()) | set(proposed_config.keys())):
        if key in VOLATILE_CONFIG_KEYS:
            continue
        if normalize_config_for_diff({key: base_config.get(key)}).get(key) != normalize_config_for_diff({key: proposed_config.get(key)}).get(key):
            changes.append(key)
    return changes


def numeric(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def int_numeric(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def rank_counts(summary: Dict[str, Any]) -> Dict[str, int]:
    raw = summary.get("rank_counts", {})
    if not isinstance(raw, dict):
        return {}
    return {str(k): int_numeric(v) for k, v in raw.items()}


def extract_metrics(holdout_summary: Dict[str, Any], model_selection_summary: Dict[str, Any], progress_summary: Dict[str, Any]) -> Dict[str, Any]:
    counts = rank_counts(holdout_summary)
    high_grade = sum(counts.get(r, 0) for r in ["1等", "2等", "3等", "4等"])
    grade = sum(counts.get(r, 0) for r in ["1等", "2等", "3等", "4等", "5等", "6等"])
    selected_holdout = model_selection_summary.get("selected_holdout", {})
    if not isinstance(selected_holdout, dict):
        selected_holdout = {}
    return {
        "created_at": utc_now(),
        "holdout_complete": bool(holdout_summary.get("complete", False)),
        "target_draws": int_numeric(holdout_summary.get("target_draws")),
        "total_target_draws": int_numeric(holdout_summary.get("total_target_draws")),
        "remaining_target_draws": int_numeric(holdout_summary.get("remaining_target_draws")),
        "roi_percent": numeric(holdout_summary.get("roi_percent")),
        "profit": int_numeric(holdout_summary.get("profit")),
        "total_cost": int_numeric(holdout_summary.get("total_cost")),
        "total_payout": int_numeric(holdout_summary.get("total_payout")),
        "max_main_match": int_numeric(holdout_summary.get("max_main_match")),
        "rank_counts": counts,
        "high_grade_hit_count": high_grade,
        "grade_hit_count": grade,
        "selected_model": model_selection_summary.get("selected_model"),
        "selection_mode": model_selection_summary.get("selection_mode"),
        "prediction_mode": model_selection_summary.get("prediction_mode"),
        "model_count": int_numeric(model_selection_summary.get("model_count")),
        "selected_holdout_roi_percent": numeric(selected_holdout.get("roi_percent")),
        "progress_best_score": numeric(progress_summary.get("best_score")),
        "progress_max_generation_seen": int_numeric(progress_summary.get("max_generation_seen")),
    }


def diagnose(metrics: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    targets = config.get("targets", {}) if isinstance(config.get("targets", {}), dict) else {}
    issues: List[Dict[str, Any]] = []

    if not metrics.get("holdout_complete"):
        issues.append({"severity": "high", "code": "holdout_incomplete", "message": "holdoutが未完了です。完全評価前の自動採用は避けます。"})

    if int_numeric(metrics.get("target_draws")) < int_numeric(targets.get("min_total_target_draws"), 600):
        issues.append({"severity": "high", "code": "insufficient_holdout_draws", "message": "検証対象回数が不足しています。"})

    roi = numeric(metrics.get("roi_percent"))
    target_roi = numeric(targets.get("roi_percent"), 40.0)
    if roi < target_roi:
        issues.append({"severity": "medium", "code": "roi_below_target", "message": f"ROI {roi:.3f}% が目標 {target_roi:.3f}% を下回っています。"})

    high_grade = int_numeric(metrics.get("high_grade_hit_count"))
    target_high = int_numeric(targets.get("high_grade_hit_count"), 8)
    if high_grade < target_high:
        issues.append({"severity": "medium", "code": "high_grade_below_target", "message": f"4等以上 {high_grade}口 が目標 {target_high}口 を下回っています。"})

    max_match = int_numeric(metrics.get("max_main_match"))
    target_match = int_numeric(targets.get("max_main_match"), 6)
    if max_match < target_match:
        issues.append({"severity": "medium", "code": "max_match_below_target", "message": f"最大本数字一致 {max_match} が目標 {target_match} を下回っています。"})

    if str(metrics.get("prediction_mode") or "") != "ensemble":
        issues.append({"severity": "low", "code": "ensemble_not_confirmed", "message": "最新サマリーでensemble予測がまだ確認できません。"})

    return issues


def clamp_change(old: float, new: float, max_ratio: float) -> float:
    if old == 0:
        return new
    upper = old * (1.0 + max_ratio)
    lower = old * (1.0 - max_ratio)
    return min(upper, max(lower, new))


def propose_config(metrics: Dict[str, Any], config: Dict[str, Any], issues: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    proposed = copy.deepcopy(config)
    reasons: List[str] = []
    scoring = proposed.setdefault("precision_scoring", {})
    ensemble = proposed.setdefault("ensemble", {})
    safety = proposed.get("safety", {}) if isinstance(proposed.get("safety", {}), dict) else {}
    max_ratio = numeric(safety.get("max_config_change_ratio"), 0.25)
    issue_codes = {str(i.get("code")) for i in issues}

    def bump(key: str, ratio: float, reason: str) -> None:
        old = numeric(scoring.get(key), numeric(DEFAULT_CONFIG["precision_scoring"].get(key, 1.0)))
        scoring[key] = round(clamp_change(old, old * (1.0 + ratio), max_ratio), 6)
        reasons.append(reason)

    if "max_match_below_target" in issue_codes or int_numeric(metrics.get("max_main_match")) <= 5:
        bump("2等", 0.10, "最大一致を伸ばすため2等スコアを上げる")
        bump("3等", 0.08, "本数字6個一致を増やすため3等スコアを上げる")
        bump("near_miss_main5", 0.12, "本数字5個近辺の候補を残しやすくする")
        bump("near_miss_main5_bonus", 0.08, "5個一致+ボーナス候補を強める")

    if "high_grade_below_target" in issue_codes:
        target_high = int_numeric(config.get("targets", {}).get("high_grade_hit_count", 8))
        current_high = int_numeric(metrics.get("high_grade_hit_count"))
        if target_high - current_high >= 2:
            bump("4等", 0.10, "4等以上件数を増やすため4等スコアを上げる")
            bump("near_miss_main4", 0.08, "本数字4個一致近辺の候補を強める")
        old_candidates = int_numeric(ensemble.get("candidates_per_model"), 10)
        if old_candidates < 12:
            ensemble["candidates_per_model"] = min(18, old_candidates + 2)
            reasons.append("合議制候補数を増やして4等以上候補の探索幅を広げる")

    if "roi_below_target" in issue_codes:
        bump("5等", 0.04, "ROI底上げのため5等スコアを少し上げる")
        bump("6等", 0.03, "ROI底上げのため6等スコアを少し上げる")
        ensemble["roi_weight"] = round(clamp_change(numeric(ensemble.get("roi_weight"), 2.0), numeric(ensemble.get("roi_weight"), 2.0) * 1.06, max_ratio), 6)
        reasons.append("合議制でROI実績の高いモデル重みを強める")

    if int_numeric(metrics.get("high_grade_hit_count")) < int_numeric(config.get("targets", {}).get("high_grade_hit_count", 8)):
        old_overlap = int_numeric(ensemble.get("overlap_limit"), 4)
        ensemble["overlap_limit"] = max(3, min(5, old_overlap))

    if has_substantive_config_change(config, proposed):
        proposed["version"] = int_numeric(proposed.get("version"), 1) + 1
        proposed["profile"] = "self_evolved_precision"
        proposed["last_self_evolved_at"] = utc_now()
    else:
        proposed = copy.deepcopy(config)
        reasons = []
    return proposed, reasons


def adoption_decision(metrics: Dict[str, Any], config: Dict[str, Any], proposed_config: Dict[str, Any], issues: List[Dict[str, Any]], proposal_reasons: List[str]) -> Dict[str, Any]:
    high_severity = [i for i in issues if i.get("severity") == "high"]
    substantive_change = has_substantive_config_change(config, proposed_config)
    ready_for_pr = substantive_change and len(proposal_reasons) > 0 and not high_severity
    if not substantive_change:
        decision = "no_op"
    elif ready_for_pr:
        decision = "propose_pr"
    else:
        decision = "diagnose_only"
    return {
        "created_at": utc_now(),
        "ready_for_branch": substantive_change,
        "ready_for_pull_request": ready_for_pr,
        "auto_main_push_allowed": bool(config.get("safety", {}).get("allow_main_direct_push", False)),
        "decision": decision,
        "blocking_issues": high_severity,
        "proposal_reason_count": len(proposal_reasons),
        "substantive_config_change": substantive_change,
        "changed_config_paths": changed_config_paths(config, proposed_config),
        "notes": [
            "この判定は安全な候補生成用です。mainへの直接反映は行いません。",
            "実質差分がない場合は重複PRを作りません。",
            "宝くじの将来当せんや利益を保証しません。",
        ],
    }


def report_text(metrics: Dict[str, Any], issues: List[Dict[str, Any]], reasons: List[str], decision: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("LOTO7 Self Evolution Report")
    lines.append("=" * 28)
    lines.append("")
    lines.append(f"作成日時(UTC): {utc_now()}")
    lines.append("")
    lines.append("[現状指標]")
    lines.append(f"holdout完了: {metrics.get('holdout_complete')}")
    lines.append(f"検証回数: {metrics.get('target_draws')} / {metrics.get('total_target_draws')}")
    lines.append(f"ROI: {metrics.get('roi_percent')}%")
    lines.append(f"収支: {metrics.get('profit')}円")
    lines.append(f"最大本数字一致: {metrics.get('max_main_match')}")
    lines.append(f"4等以上件数: {metrics.get('high_grade_hit_count')}")
    lines.append(f"6等以上件数: {metrics.get('grade_hit_count')}")
    lines.append(f"予測方式: {metrics.get('prediction_mode')}")
    lines.append("")
    lines.append("[診断]")
    if issues:
        for issue in issues:
            lines.append(f"- {issue.get('severity')}: {issue.get('code')} / {issue.get('message')}")
    else:
        lines.append("- 大きな問題は検出されませんでした。")
    lines.append("")
    lines.append("[改善案]")
    if reasons:
        for reason in reasons:
            lines.append(f"- {reason}")
    else:
        lines.append("- 実質差分なし。重複PRは作成しません。")
    lines.append("")
    lines.append("[採用判定]")
    lines.append(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("")
    lines.append("注意: 自己進化は過去検証上の改善候補を作る仕組みであり、将来の当せんや利益を保証しません。")
    return "\n".join(lines) + "\n"


def should_write_reports(out_dir: Path, decision: Dict[str, Any]) -> bool:
    if decision.get("decision") != "no_op":
        return True
    return not all((out_dir / name).exists() for name in REPORT_FILES)


def main() -> int:
    parser = argparse.ArgumentParser(description="LOTO7 self-evolving AI controller")
    parser.add_argument("--config", default="loto7_self_evolution_config.json")
    parser.add_argument("--holdout-summary", default="outputs/holdout/holdout_summary.json")
    parser.add_argument("--model-selection-summary", default="outputs/holdout/model_selection_summary.json")
    parser.add_argument("--progress-summary", default="outputs/loto7_progress_summary.json")
    parser.add_argument("--output-dir", default="outputs/self_evolution")
    parser.add_argument("--apply", action="store_true", help="改善候補を設定ファイルへ安全適用する")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = deep_merge(DEFAULT_CONFIG, read_json(config_path, {}))
    holdout_summary = read_json(Path(args.holdout_summary), {})
    model_selection_summary = read_json(Path(args.model_selection_summary), {})
    progress_summary = read_json(Path(args.progress_summary), {})

    metrics = extract_metrics(holdout_summary, model_selection_summary, progress_summary)
    issues = diagnose(metrics, config)
    proposed_config, proposal_reasons = propose_config(metrics, config, issues)
    decision = adoption_decision(metrics, config, proposed_config, issues, proposal_reasons)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnosis = {
        "created_at": utc_now(),
        "level": "Lv1-Lv6",
        "metrics": metrics,
        "issues": issues,
    }
    proposal = {
        "created_at": utc_now(),
        "base_config": config,
        "proposed_config": proposed_config,
        "reasons": proposal_reasons,
        "changed": bool(decision.get("substantive_config_change")),
        "changed_config_paths": decision.get("changed_config_paths", []),
    }

    if should_write_reports(out_dir, decision):
        write_json(out_dir / "diagnosis.json", diagnosis)
        write_json(out_dir / "proposal.json", proposal)
        write_json(out_dir / "adoption_decision.json", decision)
        (out_dir / "comparison_report.txt").write_text(report_text(metrics, issues, proposal_reasons, decision), encoding="utf-8")

        if args.apply and decision.get("substantive_config_change"):
            write_json(config_path, proposed_config)
            write_json(out_dir / "applied_config.json", proposed_config)
        elif args.apply:
            write_json(out_dir / "applied_config.json", config)
    else:
        print("[SELF-EVOLVER] no substantive config change; keeping existing self-evolution reports", flush=True)

    print(json.dumps({"metrics": metrics, "issues": issues, "proposal_reasons": proposal_reasons, "decision": decision}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
