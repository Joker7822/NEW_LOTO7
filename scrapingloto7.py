# -*- coding: utf-8 -*-
"""
scrapingloto7.py

loto7.csv を最新化します。

従来のみずほページは抽せん一覧が JavaScript で描画され、CI(ヘッドレス)環境だと
部分的にしか描画されず「要素数不足」になりやすいので、ここでは
静的HTMLで当せん番号が取得できる「楽天×宝くじ」のページから取得します。

- 直近2か月分（ページが存在する分）を取得して既存CSVにマージ
- CSVは抽せん日順にソートして上書き（重複は回別/抽せん日で除去）

使い方:
  python scrapingloto7.py
  python scrapingloto7.py --csv loto7.csv --months 3
"""

from __future__ import annotations

import argparse
import datetime as dt
import html as html_lib
import re
import sys
import urllib.request
from urllib.parse import urljoin

import pandas as pd


RAKUTEN_PAST_INDEX = "https://takarakuji.rakuten.co.jp/backnumber/loto7_past/"


def _http_get(url: str, timeout: int = 30) -> str:
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
        # ほぼUTF-8だが、念のためreplace
        return raw.decode("utf-8", errors="replace")


def _strip_html(html: str) -> str:
    s = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    s = re.sub(r"(?is)<style.*?>.*?</style>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = html_lib.unescape(s)
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_month_urls_from_past_index(past_index_html: str) -> list[str]:
    """
    /backnumber/loto7_past/ から月別ページ (/backnumber/loto7/YYYYMM/) を抽出。
    """
    # 例: /backnumber/loto7/202512/
    rels = set(re.findall(r"/backnumber/loto7/\d{6}/", past_index_html))
    month_keys = sorted({re.search(r"(\d{6})", r).group(1) for r in rels}, reverse=True)
    return [urljoin(RAKUTEN_PAST_INDEX, f"/backnumber/loto7/{k}/") for k in month_keys]


def _parse_draws_from_month_page(month_html: str) -> list[dict]:
    """
    月別ページから (回別, 抽せん日, 本数字7, ボーナス2) を抽出。
    """
    text = _strip_html(month_html)

    # draw segment split
    parts = re.split(r"回号\s*第", text)
    out: list[dict] = []

    for seg in parts[1:]:
        # seg: "0656回 抽せん日 2025/12/12 本数字 1 4 ... ボーナス数字 (14) (25) ..."
        m_draw = re.match(r"(\d{1,6})回\b", seg)
        if not m_draw:
            continue
        draw = int(m_draw.group(1))

        m_date = re.search(r"抽せん日\s*(\d{4}/\d{2}/\d{2})", seg)
        if not m_date:
            continue
        date_str = m_date.group(1)

        m_main = re.search(r"本数字\s*([0-9 ]+?)\s*ボーナス数字", seg)
        if not m_main:
            continue
        main_nums = [int(x) for x in m_main.group(1).split() if x.isdigit()]

        m_bonus = re.search(r"ボーナス数字\s*[\(\（](\d+)[\)\）]\s*[\(\（](\d+)[\)\）]", seg)
        if not m_bonus:
            continue
        bonus_nums = [int(m_bonus.group(1)), int(m_bonus.group(2))]

        if len(main_nums) != 7 or len(bonus_nums) != 2:
            # ページのノイズに当たったらスキップ（致命エラーにはしない）
            continue

        out.append(
            {
                "回別": draw,
                "抽せん日": date_str,
                "本数字": " ".join(str(n) for n in main_nums),
                "ボーナス数字": " ".join(str(n) for n in bonus_nums),
            }
        )

    return out


def fetch_latest_draws(months: int = 2) -> pd.DataFrame:
    """
    直近 months か月ぶん（ページが存在する範囲）を取得して DataFrame 化。
    """
    idx_html = _http_get(RAKUTEN_PAST_INDEX)
    month_urls = _parse_month_urls_from_past_index(idx_html)
    if not month_urls:
        raise RuntimeError("月別ページURLの抽出に失敗しました。")

    rows: list[dict] = []
    for url in month_urls[: max(1, months)]:
        html = _http_get(url)
        rows.extend(_parse_draws_from_month_page(html))

    if not rows:
        raise RuntimeError("当せん番号の抽出に失敗しました（0件）。")

    df = pd.DataFrame(rows)
    # 正規化（datetimeにしておく）
    df["_date"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    df = df.dropna(subset=["_date"]).copy()
    df["抽せん日"] = df["_date"].dt.strftime("%Y-%m-%d")
    df = df.drop(columns=["_date"])
    return df


def update_loto7_csv(csv_path: str = "loto7.csv", months: int = 2) -> pd.DataFrame:
    """
    csv_path を更新し、更新後の DataFrame を返します。
    """
    try:
        existing = pd.read_csv(csv_path)
    except FileNotFoundError:
        existing = pd.DataFrame(columns=["回別", "抽せん日", "本数字", "ボーナス数字"])

    latest = fetch_latest_draws(months=months)

    # 既存側も日付を正規化（重複判定のため）
    if "抽せん日" in existing.columns:
        existing["_date"] = pd.to_datetime(existing["抽せん日"], errors="coerce")
        existing["抽せん日"] = existing["_date"].dt.strftime("%Y-%m-%d")
        existing = existing.drop(columns=["_date"])
    else:
        existing["抽せん日"] = pd.NaT

    # 列を揃えてマージ（既存の追加列があれば残す）
    all_cols = list(dict.fromkeys(list(existing.columns) + list(latest.columns)))
    existing2 = existing.reindex(columns=all_cols)
    latest2 = latest.reindex(columns=all_cols)

    merged = pd.concat([existing2, latest2], ignore_index=True)

    # ソート & 重複除去
    merged["_date"] = pd.to_datetime(merged["抽せん日"], errors="coerce")
    merged = merged.dropna(subset=["_date"]).copy()

    # まず回別で重複排除（無い場合は抽せん日で）
    if "回別" in merged.columns:
        merged = merged.sort_values(["回別", "_date"]).drop_duplicates(subset=["回別"], keep="last")
    merged = merged.sort_values("_date").drop_duplicates(subset=["抽せん日"], keep="last")

    merged["抽せん日"] = merged["_date"].dt.strftime("%Y-%m-%d")
    merged = merged.drop(columns=["_date"])

    merged.to_csv(csv_path, index=False, encoding="utf-8")

    # 追加分の表示（回別or抽せん日で差分）
    existed_dates = set(existing2.get("抽せん日", pd.Series(dtype=str)).dropna().astype(str).tolist())
    new_rows = merged[~merged["抽せん日"].astype(str).isin(existed_dates)].copy()

    if len(new_rows) == 0:
        print(f"[OK] {csv_path}: 追加はありません（すでに最新っぽい）")
    else:
        print(f"[OK] {csv_path}: {len(new_rows)}件 追加/更新しました")
        # 直近順に表示
        show = new_rows.sort_values("抽せん日", ascending=False)
        for _, r in show.iterrows():
            print(f'  第{int(r["回別"]):04d}回 {r["抽せん日"]}  本: {r["本数字"]}  B: {r["ボーナス数字"]}')

    return merged


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="loto7.csv", help="出力/更新するCSVパス (default: loto7.csv)")
    ap.add_argument("--months", type=int, default=2, help="取得する直近月数 (default: 2)")
    args = ap.parse_args(argv)

    update_loto7_csv(args.csv, months=max(1, args.months))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
