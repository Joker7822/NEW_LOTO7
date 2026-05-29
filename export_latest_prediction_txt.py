#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_latest_prediction_txt.py

loto7_predictions.csv の最新行だけを読み取り、
最新予測のみを latest_loto7_prediction.txt に出力する。

入力形式:
    抽せん日,予測1,信頼度1,...,予測25,信頼度25

出力形式:
    最新予測のみの読みやすいテキスト。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


DEFAULT_INPUT_CSV = "loto7_predictions.csv"
DEFAULT_OUTPUT_TXT = "latest_loto7_prediction.txt"


def read_rows(path: str) -> List[Dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSVが見つかりません: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row.get("抽せん日")]

    if not rows:
        raise ValueError(f"有効な予測行がありません: {csv_path}")

    rows.sort(key=lambda r: r.get("抽せん日", ""))
    return rows


def build_latest_text(row: Dict[str, str], max_predictions: int = 25) -> str:
    target_date = row.get("抽せん日", "不明")

    lines: List[str] = []
    lines.append("Loto7 最新予測")
    lines.append("=" * 18)
    lines.append(f"対象抽せん日: {target_date}")
    lines.append("")
    lines.append("予測結果")
    lines.append("-" * 18)

    count = 0
    for i in range(1, max_predictions + 1):
        pred = (row.get(f"予測{i}") or "").strip()
        conf = (row.get(f"信頼度{i}") or "").strip()
        if not pred:
            continue

        count += 1
        if conf:
            lines.append(f"{i:02d}. {pred}  / 信頼度: {conf}")
        else:
            lines.append(f"{i:02d}. {pred}")

    if count == 0:
        lines.append("予測データなし")

    lines.append("")
    lines.append("注意")
    lines.append("-" * 18)
    lines.append("この予測は過去実績に基づく候補生成です。")
    lines.append("的中保証は確認できません。")

    return "\n".join(lines) + "\n"


def export_latest_prediction_txt(
    input_csv: str = DEFAULT_INPUT_CSV,
    output_txt: str = DEFAULT_OUTPUT_TXT,
    max_predictions: int = 25,
) -> None:
    rows = read_rows(input_csv)
    latest = rows[-1]
    text = build_latest_text(latest, max_predictions=max_predictions)

    out_path = Path(output_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="最新のLoto7予測のみをtxt出力します。")
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="入力CSV。既定: loto7_predictions.csv")
    parser.add_argument("--output-txt", default=DEFAULT_OUTPUT_TXT, help="出力TXT。既定: latest_loto7_prediction.txt")
    parser.add_argument("--max-predictions", type=int, default=25, help="出力する予測数。既定: 25")
    args = parser.parse_args()

    export_latest_prediction_txt(
        input_csv=args.input_csv,
        output_txt=args.output_txt,
        max_predictions=args.max_predictions,
    )
    print(f"保存完了: {args.output_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
