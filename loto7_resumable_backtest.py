#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Timeout-safe wrapper for NEW_LOTO7 chunked backtest.

Runs the existing chunked backtest with small chunks and pushes generated progress
files immediately. This makes GitHub Actions timeout resumable because completed
CSV/resume files are persisted before the 6-hour limit.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROGRESS_FILES = [
    "loto7_backtest_detail.csv",
    "loto7_backtest_summary.csv",
    "loto7_backtest_report.txt",
    "loto7_backtest_resume.json",
    "loto7_memorybank.csv",
    "loto7_memorybank_4plus.csv",
    "loto7_memorybank_5plus.csv",
    "loto7_memorybank_6hit.csv",
]


def run(cmd: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def git_checkpoint(message: str) -> None:
    run(["git", "config", "user.name", "github-actions"])
    run(["git", "config", "user.email", "github-actions@github.com"])
    for p in PROGRESS_FILES:
        if Path(p).exists():
            run(["git", "add", p])
    if run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        print("[checkpoint] no changes")
        return
    c = run(["git", "commit", "-m", message])
    print(c.stdout)
    if c.returncode != 0:
        return
    for i in range(3):
        run(["git", "pull", "--rebase", "origin", "main"])
        p = run(["git", "push", "origin", "HEAD:main"])
        print(p.stdout)
        if p.returncode == 0:
            print("[checkpoint] pushed")
            return
    raise RuntimeError("checkpoint push failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--min-train", default="1")
    parser.add_argument("--tickets", default="5")
    parser.add_argument("--pool-size", default="24")
    parser.add_argument("--chunk-size", default="1")
    parser.add_argument("--max-chunks", default="1")
    args = parser.parse_args()

    chunk_count = max(1, int(args.chunk_size) * max(1, int(args.max_chunks)))
    for step in range(chunk_count):
        cmd = [
            sys.executable,
            "loto7_chunked_backtest.py",
            "--csv", args.csv,
            "--min-train", str(args.min_train),
            "--tickets", str(args.tickets),
            "--pool-size", str(args.pool_size),
            "--chunk-size", "1",
            "--max-chunks", "1",
        ]
        print(f"[run] step={step + 1}/{chunk_count}")
        result = run(cmd)
        print(result.stdout)
        if result.returncode != 0:
            return result.returncode
        git_checkpoint(f"checkpoint NEW_LOTO7 backtest step {step + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
