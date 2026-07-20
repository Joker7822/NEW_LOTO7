#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mirror legacy NEW_LOTO7 outputs into the canonical v2 layout.

The migration is intentionally non-destructive. Legacy paths remain in place so
existing workflows and resume files continue to work while consumers migrate to
``outputs/{production,evidence,state,diagnostics}``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from loto7.paths import BINDINGS, OUTPUT_LAYOUT_VERSION  # noqa: E402


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _copy_file(source: Path, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and source.stat().st_size == target.stat().st_size:
        try:
            if source.read_bytes() == target.read_bytes():
                return 0
        except OSError:
            pass
    shutil.copy2(source, target)
    return 1


def _copy_tree(source: Path, target: Path, *, exclude: set[str] | None = None) -> int:
    copied = 0
    excluded = exclude or set()
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if relative.parts and relative.parts[0] in excluded:
            continue
        copied += _copy_file(path, target / relative)
    return copied


def migrate(root: Path, *, verify_only: bool = False) -> Dict[str, object]:
    actions: List[Dict[str, object]] = []
    copied_files = 0
    missing_sources: List[str] = []

    for item in BINDINGS:
        source = root / item.legacy
        target = root / item.canonical
        exists = source.exists()
        if not exists:
            missing_sources.append(item.legacy)
        copied = 0
        if exists and not verify_only:
            if source.is_dir():
                exclude = {"sealed"} if item.key == "generation4_diagnostics" else set()
                copied = _copy_tree(source, target, exclude=exclude)
            else:
                copied = _copy_file(source, target)
        copied_files += copied
        actions.append(
            {
                "key": item.key,
                "category": item.category,
                "legacy": item.legacy,
                "canonical": item.canonical,
                "source_exists": exists,
                "resumable": item.resumable,
                "copied_files": copied,
            }
        )

    payload: Dict[str, object] = {
        "created_at": now_iso(),
        "kind": "loto7_output_layout_migration",
        "output_layout_version": OUTPUT_LAYOUT_VERSION,
        "mode": "verify_only" if verify_only else "copy_non_destructive",
        "legacy_paths_retained": True,
        "resume_compatibility": "preserved",
        "copied_files": copied_files,
        "missing_source_count": len(missing_sources),
        "missing_sources": missing_sources,
        "actions": actions,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Mirror legacy outputs into the canonical v2 layout.")
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--manifest", default="outputs/layout_manifest.json")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    payload = migrate(root, verify_only=args.verify_only)
    manifest = root / args.manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
