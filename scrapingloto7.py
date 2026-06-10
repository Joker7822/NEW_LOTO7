#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scrapingloto7.py

NEW_LOTO7 リポジトリ内の loto7.csv を最新化する。

方針:
    - 静的HTMLで取得できる楽天×宝くじのロト7バックナンバーを使用
    - pandas など外部ライブラリは使わない
    - 既存の loto7.csv とマージ
    - 列順は predictor が読む形式に固定

CSV形式:
    抽せん日,本数字,ボーナス数字,回別,1等口数,1等当選金額,...,6等口数,6等当選金額,キャリーオーバー

使い方:
    python scrapingloto7.py
    python scrapingloto7.py --csv loto7.csv --months 3
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html as html_lib
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin


RAKUTEN_PAST_INDEX = "https://takarakuji.rakuten.co.jp/backnumber/loto7_past/"
FIELDNAMES = [
    "抽せん日",
    "本数字",
    "ボーナス数字",
    "回別",
    "1等口数",
    "1等当選金額",
    "2等口数",
    "2等当選金額",
    "3等口数",
    "3等当選金額",
    "4等口数",
    "4等当選金額",
    "5等口数",
    "5等当選金額",
    "6等口数",
    "6等当選金額",
    "キャリーオーバー",
]


def http_get(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            ),
            "Accept-Language": "ja,en;q=0.9",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="replace")


def strip_html(html: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_month_urls_from_past_index(past_index_html: str) -> List[str]:
    rels = set(re.findall(r"/backnumber/loto7/\d{6}/", past_index_html))
    month_keys = []
    for rel in rels:
        m = re.search(r"(\d{6})", rel)
        if m:
            month_keys.append(m.group(1))
    month_keys = sorted(set(month_keys), reverse=True)
    return [urljoin(RAKUTEN_PAST_INDEX, f"/backnumber/loto7/{key}/") for key in month_keys]


def fmt_num_list(nums: Iterable[int]) -> str:
    return " ".join(f"{n:02d}" for n in nums)


def normalize_draw_no(value: object) -> str:
    nums = re.findall(r"\d+", str(value or ""))
    if not nums:
        return ""
    return f"第{int(nums[0])}回"


def draw_no_int(value: object) -> Optional[int]:
    nums = re.findall(r"\d+", str(value or ""))
    return int(nums[0]) if nums else None


def normalize_date(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass

    return text




def normalize_money(value: object) -> str:
    """金額表記を CSV 保存向けに正規化する。例: 6,861,400円 / 該当なし。"""
    text = str(value or "").strip()
    if not text or text == "該当なし":
        return text
    m = re.search(r"([0-9,]+)\s*円", text)
    return f"{m.group(1)}円" if m else text


def normalize_unit_count(value: object) -> str:
    """口数表記を CSV 保存向けに正規化する。例: 7口 / 該当なし。"""
    text = str(value or "").strip()
    if not text or text == "該当なし":
        return text
    m = re.search(r"([0-9,]+)\s*口", text)
    return f"{m.group(1)}口" if m else text


def parse_prize_rank(seg: str, rank: int) -> tuple[str, str]:
    """月別ページの1抽せんブロックから、指定等級の口数・当選金額を抽出する。"""
    # 例: "2等 7口 6,861,400円" / "1等 該当なし 該当なし"
    m = re.search(
        rf"{rank}等\s+(該当なし|[0-9,]+\s*口)\s+(該当なし|[0-9,]+\s*円)",
        seg,
    )
    if not m:
        return "", ""
    return normalize_unit_count(m.group(1)), normalize_money(m.group(2))


def parse_carryover(seg: str) -> str:
    m = re.search(r"キャリーオーバー\s+([0-9,]+\s*円|該当なし)", seg)
    return normalize_money(m.group(1)) if m else ""

def parse_draws_from_month_page(month_html: str) -> List[Dict[str, str]]:
    text = strip_html(month_html)
    parts = re.split(r"回号\s*第", text)
    rows: List[Dict[str, str]] = []

    for seg in parts[1:]:
        m_draw = re.match(r"(\d{1,6})回\b", seg)
        if not m_draw:
            continue
        draw_no = int(m_draw.group(1))

        m_date = re.search(r"抽せん日\s*(\d{4}/\d{2}/\d{2})", seg)
        if not m_date:
            continue
        draw_date = normalize_date(m_date.group(1))

        m_main = re.search(r"本数字\s*([0-9 ]+?)\s*ボーナス数字", seg)
        if not m_main:
            continue
        main_nums = [int(x) for x in m_main.group(1).split() if x.isdigit()]

        m_bonus = re.search(r"ボーナス数字\s*[\(\（](\d+)[\)\）]\s*[\(\（](\d+)[\)\）]", seg)
        if not m_bonus:
            continue
        bonus_nums = [int(m_bonus.group(1)), int(m_bonus.group(2))]

        if len(main_nums) != 7 or len(bonus_nums) != 2:
            continue
        if len(set(main_nums)) != 7:
            continue
        if any(n < 1 or n > 37 for n in main_nums + bonus_nums):
            continue

        row = {
            "抽せん日": draw_date,
            "本数字": fmt_num_list(main_nums),
            "ボーナス数字": fmt_num_list(bonus_nums),
            "回別": f"第{draw_no}回",
        }
        for rank in range(1, 7):
            count, amount = parse_prize_rank(seg, rank)
            row[f"{rank}等口数"] = count
            row[f"{rank}等当選金額"] = amount
        row["キャリーオーバー"] = parse_carryover(seg)
        rows.append(row)

    return rows


def fetch_latest_draws(months: int = 2) -> List[Dict[str, str]]:
    index_html = http_get(RAKUTEN_PAST_INDEX)
    month_urls = parse_month_urls_from_past_index(index_html)
    if not month_urls:
        raise RuntimeError("月別ページURLの抽出に失敗しました。")

    rows: List[Dict[str, str]] = []
    for url in month_urls[: max(1, months)]:
        html = http_get(url)
        rows.extend(parse_draws_from_month_page(html))

    if not rows:
        raise RuntimeError("当せん番号の抽出に失敗しました（0件）。")

    return rows


def read_existing_csv(csv_path: str) -> List[Dict[str, str]]:
    path = Path(csv_path)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            if not row:
                continue
            normalized = {key: str(row.get(key, "")).strip() for key in FIELDNAMES}
            normalized["抽せん日"] = normalize_date(normalized.get("抽せん日", ""))
            normalized["回別"] = normalize_draw_no(normalized.get("回別", ""))
            if normalized["抽せん日"] and normalized["本数字"]:
                rows.append(normalized)
        return rows


def row_key(row: Dict[str, str]) -> str:
    draw_no = normalize_draw_no(row.get("回別", ""))
    if draw_no:
        return f"draw:{draw_no}"
    return f"date:{normalize_date(row.get('抽せん日', ''))}"


def sort_key(row: Dict[str, str]) -> tuple:
    date_text = normalize_date(row.get("抽せん日", ""))
    draw = draw_no_int(row.get("回別", ""))
    return (date_text, draw if draw is not None else 10**9)


def merge_rows(existing: List[Dict[str, str]], latest: List[Dict[str, str]]) -> List[Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {}

    for row in existing:
        merged[row_key(row)] = {key: row.get(key, "") for key in FIELDNAMES}

    for row in latest:
        merged[row_key(row)] = {key: row.get(key, "") for key in FIELDNAMES}

    # 同日重複が残るケースも日付で最終排除
    by_date: Dict[str, Dict[str, str]] = {}
    for row in sorted(merged.values(), key=sort_key):
        by_date[normalize_date(row.get("抽せん日", ""))] = row

    return sorted(by_date.values(), key=sort_key)


def write_csv(csv_path: str, rows: List[Dict[str, str]]) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def update_loto7_csv(csv_path: str = "loto7.csv", months: int = 2) -> List[Dict[str, str]]:
    existing = read_existing_csv(csv_path)
    latest = fetch_latest_draws(months=months)
    merged = merge_rows(existing, latest)
    write_csv(csv_path, merged)

    existing_keys = {row_key(row) for row in existing}
    added_or_updated = [row for row in latest if row_key(row) not in existing_keys]

    print(f"[OK] {csv_path}: existing={len(existing)} latest_fetch={len(latest)} merged={len(merged)}")
    if added_or_updated:
        print(f"[OK] {len(added_or_updated)}件の新規候補を取得しました。")
        for row in sorted(added_or_updated, key=sort_key, reverse=True):
            print(f"  {row['回別']} {row['抽せん日']} 本: {row['本数字']} B: {row['ボーナス数字']} 1等: {row.get('1等口数', '')} {row.get('1等当選金額', '')} CO: {row.get('キャリーオーバー', '')}")
    else:
        print("[OK] 追加対象はありません。既に最新の可能性があります。")

    if merged:
        last = merged[-1]
        print(f"[LATEST] {last['回別']} {last['抽せん日']} 本: {last['本数字']} B: {last['ボーナス数字']} 1等: {last.get('1等口数', '')} {last.get('1等当選金額', '')} CO: {last.get('キャリーオーバー', '')}")

    return merged


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="loto7.csv を楽天バックナンバーから更新します。")
    parser.add_argument("--csv", default="loto7.csv", help="更新するCSVパス")
    parser.add_argument("--months", type=int, default=3, help="取得する直近月数")
    args = parser.parse_args(argv)

    try:
        update_loto7_csv(args.csv, months=max(1, args.months))
    except Exception as exc:
        print(f"[ERROR] scraping failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
