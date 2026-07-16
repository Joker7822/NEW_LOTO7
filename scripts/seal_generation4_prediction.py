#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create an immutable SHA-256 manifest for a Generation 4 prediction."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(payload: Dict[str, object]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def prediction_draw_no(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream), None)
    if not row:
        raise SystemExit("prediction CSV is empty")
    return int(float(str(row.get("prediction_draw_no") or "0")))


def read_index(path: Path) -> Dict[str, object]:
    if not path.exists() or path.stat().st_size <= 0:
        return {"kind": "loto7_generation4_sealed_index", "entries": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("sealed index must be a JSON object")
    if not isinstance(payload.get("entries"), list):
        payload["entries"] = []
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Seal Generation 4 prediction files with SHA-256.")
    parser.add_argument("--prediction", default="outputs/evolution_best_prediction.csv")
    parser.add_argument("--summary", default="outputs/generation4/latest_generation4_summary.json")
    parser.add_argument("--shadow", default="outputs/generation4/latest_shadow_predictions.json")
    parser.add_argument("--dataset", default="loto7.csv")
    parser.add_argument("--full-model", default="loto7_best_model.json")
    parser.add_argument("--recent-model", default="outputs/recent_era/recent_era_best_model.json")
    parser.add_argument("--super-model", default="outputs/super_recent/super_recent_best_model.json")
    parser.add_argument("--script", default="scripts/build_generation4_prediction.py")
    parser.add_argument("--output-dir", default="outputs/generation4/sealed")
    parser.add_argument("--latest", default="outputs/generation4/latest_sealed_manifest.json")
    parser.add_argument("--index", default="outputs/generation4/sealed_index.json")
    args = parser.parse_args()

    required = [args.prediction, args.summary, args.shadow, args.dataset, args.full_model, args.recent_model, args.script]
    files: List[Path] = []
    for raw in required:
        path = Path(raw)
        if not path.exists() or path.stat().st_size <= 0:
            raise SystemExit(f"required seal input missing: {path}")
        files.append(path)
    super_path = Path(args.super_model)
    if super_path.exists() and super_path.stat().st_size > 0:
        files.append(super_path)

    draw_no = prediction_draw_no(Path(args.prediction))
    file_hashes = {str(path): sha256_file(path) for path in files}
    body: Dict[str, object] = {
        "kind": "loto7_generation4_sealed_prediction",
        "prediction_draw_no": draw_no,
        "created_at": now_iso(),
        "git_sha": os.environ.get("GITHUB_SHA", ""),
        "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        "files": file_hashes,
        "hash_algorithm": "sha256",
    }
    digest = canonical_digest(body)
    payload = dict(body)
    payload["manifest_sha256"] = digest

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"prediction_{draw_no}_{digest[:16]}.json"
    digest_path = manifest_path.with_suffix(".sha256")
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("manifest_sha256") != digest:
            raise SystemExit(f"sealed manifest collision: {manifest_path}")
    else:
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        digest_path.write_text(f"{digest}  {manifest_path.name}\n", encoding="utf-8")

    latest_path = Path(args.latest)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    index_path = Path(args.index)
    index_payload = read_index(index_path)
    entries = index_payload.get("entries", [])
    assert isinstance(entries, list)
    if not any(isinstance(item, dict) and item.get("manifest_sha256") == digest for item in entries):
        entries.append({
            "prediction_draw_no": draw_no,
            "manifest_sha256": digest,
            "manifest_path": str(manifest_path),
            "created_at": payload["created_at"],
            "git_sha": payload["git_sha"],
        })
    index_payload["entries"] = sorted(
        entries,
        key=lambda item: (int(item.get("prediction_draw_no", 0)), str(item.get("created_at", ""))),
    )
    index_payload["latest_manifest_sha256"] = digest
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"manifest": str(manifest_path), "manifest_sha256": digest,
                      "prediction_draw_no": draw_no}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
