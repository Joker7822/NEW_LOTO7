#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/collect_txt_reports.py

LOTO7 の主要テキストレポートを outputs/txt_reports/ に再構成して集約する。

目的:
  - 最新予測、履歴照合、holdout、自己進化、Role/Regime系を1フォルダで確認できるようにする
  - 欠損レポートはINDEXとmanifestにだけ記録し、MISSING専用の空ファイルは作らない
  - 00_INDEX.txt と 99_combined_report.txt を自動生成する

出力:
  outputs/txt_reports/00_INDEX.txt
  outputs/txt_reports/01_latest_prediction.txt
  outputs/txt_reports/02_prediction_history_result.txt
  outputs/txt_reports/03_holdout_report.txt
  outputs/txt_reports/04_model_selection_report.txt
  outputs/txt_reports/05_model_self_evolution_standalone.txt
  outputs/txt_reports/06_model_self_evolution_integrated.txt
  outputs/txt_reports/07_role_ensemble_backtest.txt
  outputs/txt_reports/08_role_strategy_optimizer.txt
  outputs/txt_reports/09_regime_strategy.txt  # source exists only
  outputs/txt_reports/10_progress_summary.md
  outputs/txt_reports/11_json_summary_digest.txt
  outputs/txt_reports/99_combined_report.txt
  outputs/txt_reports/manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Sequence

REPORTS = [
    {
        "order": "01",
        "title": "Latest Prediction",
        "source": "outputs/holdout/latest_prediction_report.txt",
        "target": "01_latest_prediction.txt",
        "description": "最新5口予測。role/regime strategyが適用されている場合は役割も表示。",
    },
    {
        "order": "02",
        "title": "Prediction History Result",
        "source": "outputs/evolution_prediction_history_result.txt",
        "target": "02_prediction_history_result.txt",
        "description": "累積予測履歴の照合結果。実抽せんとの一致・収支・等級を確認。",
    },
    {
        "order": "03",
        "title": "Full Holdout Report",
        "source": "outputs/holdout/holdout_report.txt",
        "target": "03_holdout_report.txt",
        "description": "採用モデルの全期間holdout検証。ROI・収支・等級別回数。",
    },
    {
        "order": "04",
        "title": "Model Selection Report",
        "source": "outputs/holdout/model_selection_report.txt",
        "target": "04_model_selection_report.txt",
        "description": "候補モデルの再ランキングと採用理由。",
    },
    {
        "order": "05",
        "title": "Model Self Evolution Standalone",
        "source": "outputs/model_self_evolution/standalone_report.txt",
        "target": "05_model_self_evolution_standalone.txt",
        "description": "専用workflowの自己進化レポート。長時間探索の最新状態。",
    },
    {
        "order": "06",
        "title": "Model Self Evolution Integrated",
        "source": "outputs/model_self_evolution/report.txt",
        "target": "06_model_self_evolution_integrated.txt",
        "description": "統合workflow内の軽量自己進化レポート。",
    },
    {
        "order": "07",
        "title": "Role Ensemble Backtest",
        "source": "outputs/role_ensemble/role_ensemble_report.txt",
        "target": "07_role_ensemble_backtest.txt",
        "description": "role_ensemble と best_model top5 の比較、役割別成績。",
    },
    {
        "order": "08",
        "title": "Role Strategy Optimizer",
        "source": "outputs/role_ensemble/role_strategy_report.txt",
        "target": "08_role_strategy_optimizer.txt",
        "description": "役割別成績から生成された5口配分 strategy。",
    },
    {
        "order": "09",
        "title": "Regime Strategy",
        "source": "outputs/role_ensemble/regime_strategy_report.txt",
        "target": "09_regime_strategy.txt",
        "description": "直近レジーム判定と、状態別に補正された5口配分。",
    },
    {
        "order": "10",
        "title": "Progress Summary",
        "source": "outputs/loto7_progress_summary.md",
        "target": "10_progress_summary.md",
        "description": "全体進捗サマリー。",
    },
]

JSON_SUMMARIES = [
    "outputs/evolution_merged_summary.json",
    "outputs/run_manifest.json",
    "outputs/holdout/holdout_summary.json",
    "outputs/holdout/model_selection_summary.json",
    "outputs/model_self_evolution/standalone_summary.json",
    "outputs/role_ensemble/role_ensemble_summary.json",
    "outputs/role_ensemble/role_strategy.json",
    "outputs/role_ensemble/regime_state.json",
    "outputs/role_ensemble/regime_strategy.json",
    "outputs/loto7_progress_summary.json",
]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def clean_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in out_dir.iterdir():
        if item.is_file() and item.suffix.lower() in {".txt", ".md", ".json"}:
            item.unlink()


def header(title: str, source: str, created_at: str, exists: bool) -> str:
    status = "FOUND" if exists else "MISSING"
    return "\n".join(
        [
            title,
            "=" * len(title),
            "",
            f"created_at: {created_at}",
            f"source: {source}",
            f"status: {status}",
            "",
        ]
    )


def copy_report(report: Dict[str, str], out_dir: Path, created_at: str) -> Dict[str, object]:
    src = Path(report["source"])
    dst = out_dir / report["target"]
    exists = src.exists() and src.is_file() and src.stat().st_size > 0
    if exists:
        body = read_text(src).rstrip() + "\n"
        content = header(str(report["title"]), str(report["source"]), created_at, True) + body
        write_text(dst, content)
    elif dst.exists():
        dst.unlink()
    return {
        "order": report["order"],
        "title": report["title"],
        "source": report["source"],
        "target": str(dst),
        "description": report["description"],
        "exists": exists,
        "bytes": dst.stat().st_size if dst.exists() else 0,
    }


def compact_json_value(value: object, max_len: int = 120) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
    text = text.replace("\n", " ")
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def summarize_json(path: Path) -> Dict[str, object]:
    if not path.exists() or path.stat().st_size <= 0:
        return {"source": str(path), "exists": False}
    try:
        payload = json.loads(read_text(path))
    except Exception as exc:
        return {"source": str(path), "exists": True, "error": str(exc)}
    if not isinstance(payload, dict):
        return {"source": str(path), "exists": True, "type": type(payload).__name__}

    keys = [
        "created_at",
        "updated_at",
        "status",
        "kind",
        "prediction_mode",
        "regime",
        "regime_strength",
        "selected_genome_id",
        "genome_id",
        "roi_percent",
        "profit",
        "target_draws",
        "target_draws_total",
        "completed_target_draws",
        "latest_draw_no",
        "prediction_draw_no",
    ]
    compact = {k: payload.get(k) for k in keys if k in payload}
    for nested_key in ["selected_holdout", "role_ensemble", "best_model", "comparison", "strategy_counts", "final_counts"]:
        if nested_key in payload:
            compact[nested_key] = payload.get(nested_key)
    return {"source": str(path), "exists": True, "summary": compact}


def write_json_digest(out_dir: Path, created_at: str) -> List[Dict[str, object]]:
    items = [summarize_json(Path(path)) for path in JSON_SUMMARIES]
    lines = [
        "JSON Summary Digest",
        "===================",
        "",
        f"created_at: {created_at}",
        "",
    ]
    for item in items:
        source = item.get("source")
        exists = item.get("exists")
        lines.append(f"## {source}")
        lines.append(f"status: {'FOUND' if exists else 'MISSING'}")
        if item.get("error"):
            lines.append(f"error: {item['error']}")
        summary = item.get("summary")
        if isinstance(summary, dict):
            for key, value in summary.items():
                lines.append(f"{key}: {compact_json_value(value)}")
        lines.append("")
    write_text(out_dir / "11_json_summary_digest.txt", "\n".join(lines).rstrip() + "\n")
    return items


def write_index(out_dir: Path, created_at: str, entries: Sequence[Dict[str, object]], json_items: Sequence[Dict[str, object]]) -> None:
    lines = [
        "LOTO7 TXT Reports Index",
        "=======================",
        "",
        f"created_at: {created_at}",
        "directory: outputs/txt_reports",
        "",
        "[Reports]",
    ]
    for entry in entries:
        status = "FOUND" if entry.get("exists") else "MISSING_SOURCE_NO_FILE"
        target = entry["target"] if entry.get("exists") else "(not generated)"
        lines.append(f"{entry['order']}. {entry['title']} / {target} / {status}")
        lines.append(f"    source: {entry['source']}")
        lines.append(f"    note: {entry['description']}")
    lines.extend(["", "[JSON Summaries]"])
    for item in json_items:
        status = "FOUND" if item.get("exists") else "MISSING"
        lines.append(f"- {item.get('source')} / {status}")
    lines.extend(
        [
            "",
            "[Recommended Reading Order]",
            "1. 01_latest_prediction.txt",
            "2. 09_regime_strategy.txt / generated only after regime report exists",
            "3. 08_role_strategy_optimizer.txt",
            "4. 07_role_ensemble_backtest.txt",
            "5. 03_holdout_report.txt",
            "6. 05_model_self_evolution_standalone.txt",
            "7. 02_prediction_history_result.txt",
            "",
            "[Combined]",
            "- 99_combined_report.txt includes only generated report files.",
        ]
    )
    write_text(out_dir / "00_INDEX.txt", "\n".join(lines).rstrip() + "\n")


def write_combined(out_dir: Path, entries: Sequence[Dict[str, object]], created_at: str) -> None:
    parts = [
        "LOTO7 Combined TXT Report",
        "=========================",
        "",
        f"created_at: {created_at}",
        "",
    ]
    for entry in entries:
        if not entry.get("exists"):
            continue
        target = Path(str(entry["target"]))
        if not target.exists():
            continue
        parts.append("\n" + "#" * 80)
        parts.append(f"# {entry['order']} {entry['title']}")
        parts.append("#" * 80)
        parts.append("")
        parts.append(read_text(target))
    json_digest = out_dir / "11_json_summary_digest.txt"
    if json_digest.exists():
        parts.append("\n" + "#" * 80)
        parts.append("# 11 JSON Summary Digest")
        parts.append("#" * 80)
        parts.append("")
        parts.append(read_text(json_digest))
    write_text(out_dir / "99_combined_report.txt", "\n".join(parts).rstrip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect LOTO7 text reports into outputs/txt_reports.")
    parser.add_argument("--output-dir", default="outputs/txt_reports")
    parser.add_argument("--clean", action="store_true", default=True)
    parser.add_argument("--no-clean", dest="clean", action="store_false")
    args = parser.parse_args()

    created_at = now_iso()
    out_dir = Path(args.output_dir)
    if args.clean:
        clean_output_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    entries = [copy_report(report, out_dir, created_at) for report in REPORTS]
    json_items = write_json_digest(out_dir, created_at)
    write_index(out_dir, created_at, entries, json_items)
    write_combined(out_dir, entries, created_at)

    manifest = {
        "created_at": created_at,
        "kind": "loto7_txt_reports_manifest",
        "output_dir": str(out_dir),
        "reports": entries,
        "json_summaries": json_items,
        "index": str(out_dir / "00_INDEX.txt"),
        "combined": str(out_dir / "99_combined_report.txt"),
        "policy": "missing_source_reports_are_not_materialized_as_placeholder_files",
    }
    write_text(out_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
