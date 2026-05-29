#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_loto7_csv.py

NEW_LOTO7 リポジトリ内の loto7.csv を更新するためのスクリプト。
外部の Joker7822/loto7 リポジトリには依存しない。

主な動作:
    1. ローカルの loto7.csv を読み込む
    2. 参照元ページからロト7の過去結果を取得する
    3. ローカルに存在しない回を追加する
    4. 既存回に差分があれば更新する
    5. loto7.csv を抽せん日順で保存する

既定の参照元:
    https://www.ohtashp.com/topics/takarakuji/loto7/

注意:
    ohtashp.com は公式ではありません。同ページでも公式発表との確認が推奨されています。
    正確性を最優先する場合は、公式発表確認後に --manual-* で手動追加してください。
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import re
import sys
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_CSV_PATH = "loto7.csv"
DEFAULT_SOURCE_URL = "https://www.ohtashp.com/topics/takarakuji/loto7/"

HEADER = ["抽せん日", "本数字", "ボーナス数字", "回別"]


@dataclass(frozen=True)
class DrawRow:
    date: str
    main: Tuple[int, ...]
    bonus: Tuple[int, ...]
    draw_no: int

    def csv_row(self) -> Dict[str, str]:
        return {
            "抽せん日": self.date,
            "本数字": " ".join(f"{n:02d}" for n in self.main),
            "ボーナス数字": " ".join(f"{n:02d}" for n in self.bonus),
            "回別": f"第{self.draw_no}回",
        }


def parse_numbers(value: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", str(value or "")))


def normalize_date(value: str) -> str:
    parts = parse_numbers(value)
    if len(parts) < 3:
        raise ValueError(f"日付を解析できません: {value}")
    y, m, d = parts[:3]
    return date(y, m, d).isoformat()


def validate_main(nums: Sequence[int]) -> Tuple[int, ...]:
    if len(nums) != 7:
        raise ValueError(f"本数字は7個必要です: {nums}")
    if len(set(nums)) != 7:
        raise ValueError(f"本数字に重複があります: {nums}")
    if any(n < 1 or n > 37 for n in nums):
        raise ValueError(f"本数字が範囲外です: {nums}")
    return tuple(sorted(nums))


def validate_bonus(nums: Sequence[int]) -> Tuple[int, ...]:
    if len(nums) != 2:
        raise ValueError(f"ボーナス数字は2個必要です: {nums}")
    if len(set(nums)) != 2:
        raise ValueError(f"ボーナス数字に重複があります: {nums}")
    if any(n < 1 or n > 37 for n in nums):
        raise ValueError(f"ボーナス数字が範囲外です: {nums}")
    return tuple(sorted(nums))


def fetch_text(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 NEW_LOTO7 updater",
            "Accept": "text/html,text/plain,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        raw = res.read()
    for enc in ("utf-8", "utf-8-sig", "cp932", "shift_jis"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def html_to_text(src: str) -> str:
    src = re.sub(r"(?is)<script.*?>.*?</script>", " ", src)
    src = re.sub(r"(?is)<style.*?>.*?</style>", " ", src)
    src = re.sub(r"(?i)<br\s*/?>", " ", src)
    src = re.sub(r"(?i)</(tr|p|div|li|h\d|table)>", "\n", src)
    src = re.sub(r"<[^>]+>", " ", src)
    src = html.unescape(src)
    src = re.sub(r"[\u3000\t\r]+", " ", src)
    src = re.sub(r" +", " ", src)
    src = re.sub(r"\n+", "\n", src)
    return src


def parse_draws_from_ohtashp(text: str) -> List[DrawRow]:
    """
    ohtashp の一覧ページから、以下のような並びを抽出する。

    第675回 2026/5/1 05 08 16 18 24 28 31 06 23 ...
    """
    plain = html_to_text(text)

    pattern = re.compile(
        r"第\s*(\d+)\s*回\s+"
        r"(\d{4}/\d{1,2}/\d{1,2})\s+"
        r"(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+"
        r"(\d{1,2})\s+(\d{1,2})"
    )

    rows: Dict[int, DrawRow] = {}
    for m in pattern.finditer(plain):
        draw_no = int(m.group(1))
        draw_date = normalize_date(m.group(2))
        main = validate_main(tuple(int(m.group(i)) for i in range(3, 10)))
        bonus = validate_bonus((int(m.group(10)), int(m.group(11))))
        rows[draw_no] = DrawRow(date=draw_date, main=main, bonus=bonus, draw_no=draw_no)

    return sorted(rows.values(), key=lambda r: r.draw_no)


def read_local_csv(path: str) -> List[DrawRow]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []

    rows: List[DrawRow] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                draw_date = normalize_date(row.get("抽せん日", ""))
                main = validate_main(parse_numbers(row.get("本数字", "")))
                bonus = validate_bonus(parse_numbers(row.get("ボーナス数字", "")))
                draw_no_values = parse_numbers(row.get("回別", ""))
                if not draw_no_values:
                    continue
                draw_no = int(draw_no_values[0])
                rows.append(DrawRow(draw_date, main, bonus, draw_no))
            except Exception as exc:
                print(f"[WARN] 壊れた行をスキップ: {row} / {exc}", file=sys.stderr)
    return sorted(rows, key=lambda r: r.draw_no)


def write_local_csv(path: str, rows: Sequence[DrawRow]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    unique: Dict[int, DrawRow] = {r.draw_no: r for r in rows}
    ordered = sorted(unique.values(), key=lambda r: r.date)

    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        for row in ordered:
            writer.writerow(row.csv_row())


def merge_rows(local_rows: Sequence[DrawRow], source_rows: Sequence[DrawRow]) -> Tuple[List[DrawRow], int, int]:
    merged: Dict[int, DrawRow] = {r.draw_no: r for r in local_rows}
    added = 0
    updated = 0

    for src in source_rows:
        old = merged.get(src.draw_no)
        if old is None:
            merged[src.draw_no] = src
            added += 1
        elif old != src:
            merged[src.draw_no] = src
            updated += 1

    return sorted(merged.values(), key=lambda r: r.draw_no), added, updated


def build_manual_row(args: argparse.Namespace) -> Optional[DrawRow]:
    values = [args.manual_date, args.manual_main, args.manual_bonus, args.manual_draw_no]
    if not any(values):
        return None
    if not all(values):
        raise ValueError("手動追加は --manual-date, --manual-main, --manual-bonus, --manual-draw-no をすべて指定してください。")

    return DrawRow(
        date=normalize_date(args.manual_date),
        main=validate_main(parse_numbers(args.manual_main)),
        bonus=validate_bonus(parse_numbers(args.manual_bonus)),
        draw_no=int(args.manual_draw_no),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="NEW_LOTO7内のloto7.csvを更新します。")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="更新対象CSV。既定: loto7.csv")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL, help="自動取得元URL")
    parser.add_argument("--skip-fetch", action="store_true", help="外部取得せず、手動追加のみ行う")
    parser.add_argument("--manual-date", default="", help="手動追加: 抽せん日 YYYY-MM-DD または YYYY/M/D")
    parser.add_argument("--manual-main", default="", help="手動追加: 本数字7個。例: '06 08 09 18 22 24 35'")
    parser.add_argument("--manual-bonus", default="", help="手動追加: ボーナス数字2個。例: '04 20'")
    parser.add_argument("--manual-draw-no", default="", help="手動追加: 回別番号。例: 679")
    args = parser.parse_args()

    local_rows = read_local_csv(args.csv)
    source_rows: List[DrawRow] = []

    if not args.skip_fetch:
        try:
            html_text = fetch_text(args.source_url)
            source_rows = parse_draws_from_ohtashp(html_text)
            print(f"取得元から {len(source_rows)} 件を解析しました: {args.source_url}")
        except Exception as exc:
            print(f"[WARN] 自動取得に失敗しました: {exc}", file=sys.stderr)
            source_rows = []

    manual = build_manual_row(args)
    if manual is not None:
        source_rows.append(manual)
        print(f"手動追加行を受け付けました: 第{manual.draw_no}回 {manual.date}")

    if not source_rows:
        print("更新元データがありません。loto7.csvは変更しません。")
        return 0

    merged, added, updated = merge_rows(local_rows, source_rows)
    write_local_csv(args.csv, merged)

    latest = merged[-1] if merged else None
    print(f"保存完了: {args.csv}")
    print(f"追加: {added}件 / 更新: {updated}件 / 合計: {len(merged)}件")
    if latest:
        print(
            f"最新: 第{latest.draw_no}回 {latest.date} "
            f"本数字={latest.csv_row()['本数字']} ボーナス={latest.csv_row()['ボーナス数字']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
