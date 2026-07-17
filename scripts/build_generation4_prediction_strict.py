#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-closed entrypoint for Generation 4 production prediction adoption."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.strict_adoption_gates import (  # noqa: E402
    null_league_adoption_gate,
    read_json,
    recalibrated_conformal_number_pool,
    write_json,
)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main(argv: Optional[List[str]] = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--null-league-summary",
        default="outputs/generation4/null_strategy_league_summary.json",
    )
    parser.add_argument("--require-null-league-summary", action="store_true")
    parser.add_argument(
        "--strict-gate-summary",
        default="outputs/generation4/strict_adoption_gate.json",
    )
    parser.add_argument("--conformal-required-hits", type=int, default=4)
    strict, delegated = parser.parse_known_args(raw_args)

    null_path = Path(strict.null_league_summary)
    null_payload = read_json(str(null_path)) if null_path.exists() and null_path.stat().st_size > 0 else None
    null_gate = null_league_adoption_gate(
        null_payload,
        require_available=bool(strict.require_null_league_summary),
    )
    gate_payload = {
        "created_at": now_iso(),
        "kind": "loto7_generation4_strict_adoption_gate",
        "null_league_summary": str(null_path),
        "null_league_gate": null_gate,
        "conformal_recalibration": {
            "enabled": True,
            "required_main_hits": int(strict.conformal_required_hits),
            "method": "rolling_prior_top_k_minimum_hit_coverage_v1",
        },
        "adoption_allowed": bool(null_gate.get("adoption_allowed")),
    }
    write_json(strict.strict_gate_summary, gate_payload)
    if not gate_payload["adoption_allowed"]:
        print(json.dumps(gate_payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    import scripts.generation4_core as generation4_core  # noqa: E402

    required_hits = int(strict.conformal_required_hits)

    def patched_conformal_number_pool(draws, **kwargs):
        return recalibrated_conformal_number_pool(
            draws,
            required_hits=required_hits,
            **kwargs,
        )

    generation4_core.conformal_number_pool = patched_conformal_number_pool

    # Import only after patching so the builder binds the recalibrated function.
    import scripts.build_generation4_prediction as builder  # noqa: E402

    delegated.extend(["--null-league-summary", strict.null_league_summary])
    return int(builder.main(delegated))


if __name__ == "__main__":
    raise SystemExit(main())
