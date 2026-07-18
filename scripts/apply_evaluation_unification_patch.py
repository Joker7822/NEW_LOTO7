#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    text = read(path)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one exact match, found {count}: {old[:100]!r}")
    write(path, text.replace(old, new, 1))


def regex_once(path: str, pattern: str, replacement: str) -> None:
    text = read(path)
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE | re.DOTALL)
    if count != 1:
        raise RuntimeError(f"{path}: expected one regex match, found {count}: {pattern[:100]!r}")
    write(path, updated)


EVALUATION_CORE = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical LOTO7 prize loading and financial evaluation utilities.

All production, holdout and diagnostic evaluators should use these definitions.
Legacy ``roi`` aliases are intentionally defined as profit ROI:
    (total_payout - total_cost) / total_cost
The payout ratio is exposed separately as ``payout_roi``.
"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Dict, Mapping, Optional

EVALUATOR_VERSION = "loto7-evaluator-2026.07.18-v1"
RANK_ORDER = ["1等", "2等", "3等", "4等", "5等", "6等", "外れ"]
PRIZE_RANKS = ["1等", "2等", "3等", "4等", "5等", "6等"]
HIGH_GRADE_RANKS = ["1等", "2等", "3等", "4等"]


def file_sha256(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def draw_no_int(text: object) -> Optional[int]:
    match = re.search(r"\d+", str(text or ""))
    return int(match.group(0)) if match else None


def parse_money_yen(text: object) -> int:
    raw = str(text or "").strip()
    if not raw or raw == "該当なし":
        return 0
    match = re.search(r"([0-9,]+)", raw)
    return int(match.group(1).replace(",", "")) if match else 0


def load_prize_rows(csv_path: str | Path) -> Dict[int, Dict[str, str]]:
    output: Dict[int, Dict[str, str]] = {}
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            draw_no = draw_no_int(row.get("回別"))
            if draw_no is not None:
                output[draw_no] = {key: str(value or "").strip() for key, value in row.items()}
    return output


def has_any_prize_amount(row: Mapping[str, object]) -> bool:
    return any(str(row.get(f"{rank}当選金額", "")).strip() for rank in PRIZE_RANKS)


def prize_amount_for_rank(row: Mapping[str, object], rank: str) -> int:
    if rank == "外れ":
        return 0
    return parse_money_yen(row.get(f"{rank}当選金額", ""))


def profit_roi(total_cost: int, total_payout: int) -> float:
    return (total_payout - total_cost) / total_cost if total_cost > 0 else 0.0


def payout_roi(total_cost: int, total_payout: int) -> float:
    return total_payout / total_cost if total_cost > 0 else 0.0


def financial_metrics(
    *,
    total_cost: int,
    total_payout: int,
    total_tickets: int = 0,
    winning_tickets: int = 0,
    target_draws: int = 0,
    winning_draws: int = 0,
) -> Dict[str, object]:
    profit = int(total_payout) - int(total_cost)
    profit_ratio = profit_roi(int(total_cost), int(total_payout))
    payout_ratio = payout_roi(int(total_cost), int(total_payout))
    ticket_hit = winning_tickets / total_tickets if total_tickets > 0 else 0.0
    draw_hit = winning_draws / target_draws if target_draws > 0 else 0.0
    return {
        "total_cost": int(total_cost),
        "total_payout": int(total_payout),
        "profit": profit,
        "roi": round(profit_ratio, 6),
        "roi_percent": round(profit_ratio * 100.0, 3),
        "profit_roi": round(profit_ratio, 6),
        "profit_roi_percent": round(profit_ratio * 100.0, 3),
        "payout_roi": round(payout_ratio, 6),
        "payout_roi_percent": round(payout_ratio * 100.0, 3),
        "ticket_hit_rate": round(ticket_hit, 6),
        "ticket_hit_rate_percent": round(ticket_hit * 100.0, 3),
        "draw_hit_rate": round(draw_hit, 6),
        "draw_hit_rate_percent": round(draw_hit * 100.0, 3),
        "evaluator_version": EVALUATOR_VERSION,
    }


def finalize_stats(stats: Mapping[str, object]) -> Dict[str, object]:
    ranks_value = stats.get("rank_counts", {})
    ranks = ranks_value if isinstance(ranks_value, Mapping) else {}
    rank_counts = {rank: int(ranks.get(rank, 0) or 0) for rank in RANK_ORDER}
    winning_tickets = sum(rank_counts[rank] for rank in PRIZE_RANKS)
    high_grade = sum(rank_counts[rank] for rank in HIGH_GRADE_RANKS)
    output = dict(stats)
    output.update(
        financial_metrics(
            total_cost=int(stats.get("total_cost", 0) or 0),
            total_payout=int(stats.get("total_payout", 0) or 0),
            total_tickets=int(stats.get("total_tickets", 0) or 0),
            winning_tickets=winning_tickets,
            target_draws=int(stats.get("draw_count", 0) or 0),
            winning_draws=int(stats.get("draw_hit_count", 0) or 0),
        )
    )
    output["grade_hit_count"] = winning_tickets
    output["high_grade_hit_count"] = high_grade
    output["rank_counts"] = rank_counts
    return output
'''

VERIFY_SCRIPT = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify that holdout and role-backtest best-model evaluations are identical."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scripts.evaluation_core import EVALUATOR_VERSION, financial_metrics

Key = Tuple[int, int]


def normalize_ticket(value: object) -> str:
    return " ".join(f"{int(token):02d}" for token in str(value or "").replace(",", " ").split() if token.isdigit())


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def verify(holdout_path: Path, role_path: Path, unit_cost: int = 300) -> Dict[str, object]:
    holdout_rows = read_rows(holdout_path)
    role_rows = [row for row in read_rows(role_path) if row.get("system") == "best_model"]
    holdout: Dict[Key, Dict[str, str]] = {
        (int(row["draw_no"]), int(row["combo_index"])): row for row in holdout_rows
    }
    role: Dict[Key, Dict[str, str]] = {
        (int(row["target_draw_no"]), int(row["ticket_index"])): row for row in role_rows
    }
    missing_in_role = sorted(set(holdout) - set(role))
    missing_in_holdout = sorted(set(role) - set(holdout))
    mismatches: List[Dict[str, object]] = []
    for key in sorted(set(holdout) & set(role)):
        left = holdout[key]
        right = role[key]
        checks = {
            "ticket": (normalize_ticket(left.get("ticket")), normalize_ticket(right.get("numbers"))),
            "main_match": (str(left.get("main_match", "")), str(right.get("main_match", ""))),
            "bonus_match": (str(left.get("bonus_match", "")), str(right.get("bonus_match", ""))),
            "rank": (str(left.get("rank", "")), str(right.get("rank", ""))),
            "payout": (str(left.get("prize_amount", "0")), str(right.get("payout", "0"))),
        }
        differences = {name: values for name, values in checks.items() if values[0] != values[1]}
        if differences:
            mismatches.append({"draw_no": key[0], "ticket_index": key[1], "differences": differences})

    def aggregate(rows: List[Dict[str, str]], *, holdout_format: bool) -> Dict[str, object]:
        total_payout = sum(int(row.get("prize_amount" if holdout_format else "payout", 0) or 0) for row in rows)
        rank_key = "rank"
        winners = sum(1 for row in rows if row.get(rank_key) not in {"", "外れ"})
        draw_numbers = {int(row.get("draw_no" if holdout_format else "target_draw_no", 0) or 0) for row in rows}
        winning_draw_numbers = {
            int(row.get("draw_no" if holdout_format else "target_draw_no", 0) or 0)
            for row in rows if row.get(rank_key) not in {"", "外れ"}
        }
        return financial_metrics(
            total_cost=len(rows) * unit_cost,
            total_payout=total_payout,
            total_tickets=len(rows),
            winning_tickets=winners,
            target_draws=len(draw_numbers),
            winning_draws=len(winning_draw_numbers),
        )

    holdout_metrics = aggregate(holdout_rows, holdout_format=True)
    role_metrics = aggregate(role_rows, holdout_format=False)
    metrics_equal = all(
        holdout_metrics.get(key) == role_metrics.get(key)
        for key in ("total_cost", "total_payout", "profit", "profit_roi_percent", "payout_roi_percent", "ticket_hit_rate_percent", "draw_hit_rate_percent")
    )
    payload: Dict[str, object] = {
        "kind": "loto7_evaluator_consistency",
        "evaluator_version": EVALUATOR_VERSION,
        "holdout_rows": len(holdout_rows),
        "role_best_model_rows": len(role_rows),
        "missing_in_role_count": len(missing_in_role),
        "missing_in_holdout_count": len(missing_in_holdout),
        "mismatch_count": len(mismatches),
        "mismatch_samples": mismatches[:20],
        "holdout_metrics": holdout_metrics,
        "role_best_model_metrics": role_metrics,
        "metrics_equal": metrics_equal,
        "passed": not missing_in_role and not missing_in_holdout and not mismatches and metrics_equal,
    }
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", default="outputs/holdout/holdout_result.csv")
    parser.add_argument("--role", default="outputs/role_ensemble/role_ensemble_backtest.csv")
    parser.add_argument("--output", default="outputs/role_ensemble/evaluator_consistency.json")
    parser.add_argument("--unit-cost", type=int, default=300)
    args = parser.parse_args(argv)
    payload = verify(Path(args.holdout), Path(args.role), args.unit_cost)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
'''

TEST_FILE = r'''from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.evaluation_core import EVALUATOR_VERSION, financial_metrics
from scripts.verify_evaluator_consistency import verify


class EvaluationUnificationTests(unittest.TestCase):
    def test_financial_metrics_use_profit_roi_and_separate_payout_roi(self) -> None:
        metrics = financial_metrics(
            total_cost=1000,
            total_payout=1250,
            total_tickets=10,
            winning_tickets=2,
            target_draws=2,
            winning_draws=1,
        )
        self.assertEqual(metrics["profit"], 250)
        self.assertEqual(metrics["roi_percent"], 25.0)
        self.assertEqual(metrics["profit_roi_percent"], 25.0)
        self.assertEqual(metrics["payout_roi_percent"], 125.0)
        self.assertEqual(metrics["ticket_hit_rate_percent"], 20.0)
        self.assertEqual(metrics["draw_hit_rate_percent"], 50.0)
        self.assertEqual(metrics["evaluator_version"], EVALUATOR_VERSION)

    def test_holdout_and_role_best_rows_must_match_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            holdout = root / "holdout.csv"
            role = root / "role.csv"
            with holdout.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["draw_no", "combo_index", "ticket", "main_match", "bonus_match", "rank", "prize_amount"])
                writer.writeheader()
                writer.writerow({"draw_no": 2, "combo_index": 1, "ticket": "01 02 03 04 05 06 07", "main_match": 4, "bonus_match": 0, "rank": "5等", "prize_amount": 1500})
            with role.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["system", "target_draw_no", "ticket_index", "numbers", "main_match", "bonus_match", "rank", "payout"])
                writer.writeheader()
                writer.writerow({"system": "best_model", "target_draw_no": 2, "ticket_index": 1, "numbers": "01 02 03 04 05 06 07", "main_match": 4, "bonus_match": 0, "rank": "5等", "payout": 1500})
            payload = verify(holdout, role, 300)
            self.assertTrue(payload["passed"])
            self.assertTrue(payload["metrics_equal"])

    def test_stale_role_resume_output_is_deleted(self) -> None:
        from scripts.backtest_role_ensemble import load_resume_details

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.json"
            model.write_text("{}", encoding="utf-8")
            output = root / "detail.csv"
            output.write_text("stale\n", encoding="utf-8")
            state = root / "state.json"
            state.write_text(json.dumps({"csv": "loto7.csv", "best_model": str(model), "genome_id": "g1", "purchase_count": 5, "unit_cost": 300, "holdout_start_draw": 2, "min_train_draws": 1, "target_draws_total": 1}), encoding="utf-8")
            args = SimpleNamespace(
                resume=True,
                state=str(state),
                output=str(output),
                csv="loto7.csv",
                best_model=str(model),
                purchase_count=5,
                unit_cost=300,
                holdout_start_draw=2,
                min_train_draws=1,
                max_targets=0,
            )
            rows, completed, missing, removed = load_resume_details(args, "g1", [2])
            self.assertEqual(rows, [])
            self.assertEqual(completed, set())
            self.assertFalse(output.exists())

    def test_shadow_evaluate_only_does_not_append_latest_prediction(self) -> None:
        from scripts.update_generation4_shadow_history import main as update_shadow

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draws = root / "loto7.csv"
            headers = ["抽せん日", "本数字", "ボーナス数字", "回別"] + [f"{rank}当選金額" for rank in ["1等", "2等", "3等", "4等", "5等", "6等"]]
            with draws.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=headers)
                writer.writeheader()
                writer.writerow({"抽せん日": "2026-01-01", "本数字": "01 02 03 04 05 06 07", "ボーナス数字": "08 09", "回別": "第1回", "1等当選金額": "1000円"})
                writer.writerow({"抽せん日": "2026-01-08", "本数字": "01 02 03 04 05 06 07", "ボーナス数字": "08 09", "回別": "第2回", "1等当選金額": "1000円"})
                writer.writerow({"抽せん日": "2026-01-15", "本数字": "10 11 12 13 14 15 16", "ボーナス数字": "17 18", "回別": "第3回", "1等当選金額": "1000円"})
            history = root / "shadow.csv"
            fields = ["prediction_draw_no", "prediction_date", "strategy", "tickets_json", "status", "actual_main", "actual_bonus", "max_main_match", "total_main_matches", "winning_tickets", "total_payout", "utility", "created_at"]
            with history.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"prediction_draw_no": "2", "strategy": "generation4", "tickets_json": json.dumps(["01 02 03 04 05 06 07"]), "status": "pending"})
            latest = root / "latest.json"
            latest.write_text(json.dumps({"prediction_draw_no": 3, "strategies": {"generation4": ["10 11 12 13 14 15 16"]}}), encoding="utf-8")
            summary = root / "summary.json"
            report = root / "report.txt"
            result = update_shadow([
                "--csv", str(draws), "--latest-shadow", str(latest), "--history", str(history),
                "--summary", str(summary), "--report", str(report), "--evaluate-only",
            ])
            self.assertEqual(result, 0)
            with history.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["prediction_draw_no"], "2")
            self.assertEqual(rows[0]["status"], "evaluated")
            self.assertEqual(rows[0]["total_payout"], "1000")


if __name__ == "__main__":
    unittest.main()
'''

write("scripts/evaluation_core.py", EVALUATION_CORE)
write("scripts/verify_evaluator_consistency.py", VERIFY_SCRIPT)
write("tests/test_evaluation_unification.py", TEST_FILE)

# holdout_evaluator.py: canonical prize loader/metrics and state fingerprint.
replace_once(
    "holdout_evaluator.py",
    '''from loto7_evolution_trainer import (\n    Draw,\n    evaluate_ticket,\n    generate_tickets,\n    load_best_model,\n    load_draws,\n)\n\nRANK_ORDER = ["1等", "2等", "3等", "4等", "5等", "6等", "外れ"]\nPRIZE_RANKS = ["1等", "2等", "3等", "4等", "5等", "6等"]\n''',
    '''from loto7_evolution_trainer import (\n    Draw,\n    evaluate_ticket,\n    generate_tickets,\n    load_best_model,\n    load_draws,\n)\nfrom scripts.evaluation_core import (\n    EVALUATOR_VERSION,\n    PRIZE_RANKS,\n    RANK_ORDER,\n    draw_no_int,\n    file_sha256,\n    has_any_prize_amount,\n    load_prize_rows,\n    payout_roi as canonical_payout_roi,\n    prize_amount_for_rank,\n    profit_roi as canonical_profit_roi,\n)\n''',
)
regex_once(
    "holdout_evaluator.py",
    r'''def draw_no_int\(text: object\) -> Optional\[int\]:.*?def fmt_ticket\(ticket: Sequence\[int\]\) -> str:''',
    '''def fmt_ticket(ticket: Sequence[int]) -> str:''',
)
replace_once(
    "holdout_evaluator.py",
    '''def roi_from_profit(total_cost: int, total_payout: int) -> float:\n    """収支ベースROI: (払戻額 - 購入額) / 購入額。"""\n    if total_cost <= 0:\n        return 0.0\n    return (total_payout - total_cost) / total_cost\n\n\ndef roi_from_payout(total_cost: int, total_payout: int) -> float:\n    """従来の回収率: 払戻額 / 購入額。互換確認用に別名で保持する。"""\n    if total_cost <= 0:\n        return 0.0\n    return total_payout / total_cost\n''',
    '''def roi_from_profit(total_cost: int, total_payout: int) -> float:\n    """Canonical profit ROI: (payout - cost) / cost."""\n    return canonical_profit_roi(total_cost, total_payout)\n\n\ndef roi_from_payout(total_cost: int, total_payout: int) -> float:\n    """Canonical payout ratio: payout / cost."""\n    return canonical_payout_roi(total_cost, total_payout)\n''',
)
replace_once(
    "holdout_evaluator.py",
    '''        "min_train_draws": args.min_train_draws,\n        "output": args.output,\n''',
    '''        "min_train_draws": args.min_train_draws,\n        "evaluator_version": EVALUATOR_VERSION,\n        "model_sha256": file_sha256(args.best_model),\n        "output": args.output,\n''',
)
replace_once(
    "holdout_evaluator.py",
    '''        "model_score": genome.score,\n        "holdout_start_draw": args.holdout_start_draw,\n''',
    '''        "model_score": genome.score,\n        "model_sha256": file_sha256(args.best_model),\n        "evaluator_version": EVALUATOR_VERSION,\n        "holdout_start_draw": args.holdout_start_draw,\n''',
)

# Role backtest: canonical metrics and stale-resume invalidation.
replace_once(
    "scripts/backtest_role_ensemble.py",
    '''from merge_evolution_shards import (  # noqa: E402\n    PRIZE_RANKS,\n    RANK_ORDER,\n    fmt_ticket,\n    load_model,\n    load_prize_rows,\n    make_role_ensemble_prediction_rows,\n    prize_amount_for_rank,\n    select_target_indices,\n)\n''',
    '''from merge_evolution_shards import (  # noqa: E402\n    fmt_ticket,\n    load_model,\n    make_role_ensemble_prediction_rows,\n    select_target_indices,\n)\nfrom scripts.evaluation_core import (  # noqa: E402\n    EVALUATOR_VERSION,\n    PRIZE_RANKS,\n    RANK_ORDER,\n    file_sha256,\n    finalize_stats as canonical_finalize_stats,\n    load_prize_rows,\n    prize_amount_for_rank,\n)\n''',
)
regex_once(
    "scripts/backtest_role_ensemble.py",
    r'''def finalize_stats\(stats: Dict\[str, object\]\) -> Dict\[str, object\]:.*?\n    return out\n''',
    '''def finalize_stats(stats: Dict[str, object]) -> Dict[str, object]:\n    return canonical_finalize_stats(stats)\n''',
)
replace_once(
    "scripts/backtest_role_ensemble.py",
    '''        and int(state.get("min_train_draws", -1)) == int(args.min_train_draws)\n        and int(state.get("target_draws_total", -1)) == len(target_draw_nos)\n''',
    '''        and int(state.get("min_train_draws", -1)) == int(args.min_train_draws)\n        and state.get("evaluator_version") == EVALUATOR_VERSION\n        and state.get("model_sha256") == file_sha256(args.best_model)\n        and int(state.get("target_draws_total", -1)) == len(target_draw_nos)\n''',
)
replace_once(
    "scripts/backtest_role_ensemble.py",
    '''    if not state_path.exists() or state_path.stat().st_size <= 0:\n        return [], set(), [], 0\n''',
    '''    if not state_path.exists() or state_path.stat().st_size <= 0:\n        Path(args.output).unlink(missing_ok=True)\n        return [], set(), [], 0\n''',
)
replace_once(
    "scripts/backtest_role_ensemble.py",
    '''    if not state_matches(state, args, genome_id, target_draw_nos):\n        print("[INFO] role ensemble state does not match current settings; starting fresh")\n        return [], set(), [], 0\n''',
    '''    if not state_matches(state, args, genome_id, target_draw_nos):\n        print("[INFO] role ensemble state/model/evaluator fingerprint changed; deleting stale detail CSV")\n        Path(args.output).unlink(missing_ok=True)\n        return [], set(), [], 0\n''',
)
replace_once(
    "scripts/backtest_role_ensemble.py",
    '''        "genome_id": genome_id,\n        "purchase_count": args.purchase_count,\n''',
    '''        "genome_id": genome_id,\n        "model_sha256": file_sha256(args.best_model),\n        "evaluator_version": EVALUATOR_VERSION,\n        "purchase_count": args.purchase_count,\n''',
)
# The same fragment exists in save_state; patch the next occurrence.
replace_once(
    "scripts/backtest_role_ensemble.py",
    '''        "genome_id": genome_id,\n        "purchase_count": args.purchase_count,\n''',
    '''        "genome_id": genome_id,\n        "model_sha256": file_sha256(args.best_model),\n        "evaluator_version": EVALUATOR_VERSION,\n        "purchase_count": args.purchase_count,\n''',
)
replace_once(
    "scripts/backtest_role_ensemble.py",
    '''    lines.append(f"[Role Ensemble]") if False else None\n''',
    '''    lines.append(f"[Role Ensemble]") if False else None\n''',
) if False else None
replace_once(
    "scripts/backtest_role_ensemble.py",
    '''    lines.append(f"roi_percent: {role.get('roi_percent')}")\n    lines.append(f"profit: {role.get('profit')}")\n''',
    '''    lines.append(f"profit_roi_percent: {role.get('profit_roi_percent')}")\n    lines.append(f"payout_roi_percent: {role.get('payout_roi_percent')}")\n    lines.append(f"profit: {role.get('profit')}")\n''',
)
replace_once(
    "scripts/backtest_role_ensemble.py",
    '''    lines.append(f"roi_percent: {best.get('roi_percent')}")\n    lines.append(f"profit: {best.get('profit')}")\n''',
    '''    lines.append(f"profit_roi_percent: {best.get('profit_roi_percent')}")\n    lines.append(f"payout_roi_percent: {best.get('payout_roi_percent')}")\n    lines.append(f"profit: {best.get('profit')}")\n''',
)
replace_once(
    "scripts/backtest_role_ensemble.py",
    '''    role_roi = float(role.get("roi_percent", 0.0))\n    best_roi = float(best.get("roi_percent", 0.0))\n''',
    '''    role_roi = float(role.get("profit_roi_percent", role.get("roi_percent", 0.0)))\n    best_roi = float(best.get("profit_roi_percent", best.get("roi_percent", 0.0)))\n''',
)

# Live history evaluator uses the same canonical prize and financial metrics.
replace_once(
    "scripts/check_prediction_history_results.py",
    '''from loto7_evolution_trainer import Draw, evaluate_ticket, load_draws, parse_nums\nfrom holdout_evaluator import load_prize_rows, prize_amount_for_rank\n''',
    '''from loto7_evolution_trainer import Draw, evaluate_ticket, load_draws, parse_nums\nfrom scripts.evaluation_core import financial_metrics, load_prize_rows, prize_amount_for_rank\n''',
)
replace_once(
    "scripts/check_prediction_history_results.py",
    '''    total_profit = total_payout - total_cost\n    hit_rate = (winning_rows / evaluated_rows * 100.0) if evaluated_rows else 0.0\n    payout_roi = (total_payout / total_cost * 100.0) if total_cost else 0.0\n    profit_roi = (total_profit / total_cost * 100.0) if total_cost else 0.0\n''',
    '''    metrics = financial_metrics(\n        total_cost=total_cost,\n        total_payout=total_payout,\n        total_tickets=evaluated_rows,\n        winning_tickets=winning_rows,\n    )\n    total_profit = int(metrics["profit"])\n    hit_rate = float(metrics["ticket_hit_rate_percent"])\n    payout_roi = float(metrics["payout_roi_percent"])\n    profit_roi = float(metrics["profit_roi_percent"])\n''',
)

# Shadow updater: canonical prize evaluator and evaluate-only mode.
replace_once(
    "scripts/update_generation4_shadow_history.py",
    '''from loto7_evolution_trainer import Draw, evaluate_ticket, load_draws  # noqa: E402\nfrom merge_evolution_shards import load_prize_rows, prize_amount_for_rank  # noqa: E402\nfrom scripts.generation4_core import bounded_strategy_utility, eprocess_from_history  # noqa: E402\n''',
    '''from loto7_evolution_trainer import Draw, evaluate_ticket, load_draws  # noqa: E402\nfrom scripts.evaluation_core import load_prize_rows, prize_amount_for_rank  # noqa: E402\nfrom scripts.generation4_core import bounded_strategy_utility, eprocess_from_history  # noqa: E402\n''',
)
replace_once(
    "scripts/update_generation4_shadow_history.py",
    '''def main() -> int:\n''',
    '''def main(argv: Optional[List[str]] = None) -> int:\n''',
)
replace_once(
    "scripts/update_generation4_shadow_history.py",
    '''    parser.add_argument("--min-evaluated-draws", type=int, default=30)\n    args = parser.parse_args()\n''',
    '''    parser.add_argument("--min-evaluated-draws", type=int, default=30)\n    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate existing rows without appending latest-shadow predictions")\n    args = parser.parse_args(argv)\n''',
)
replace_once(
    "scripts/update_generation4_shadow_history.py",
    '''    latest_path = Path(args.latest_shadow)\n    if latest_path.exists() and latest_path.stat().st_size > 0:\n''',
    '''    latest_path = Path(args.latest_shadow)\n    if not args.evaluate_only and latest_path.exists() and latest_path.stat().st_size > 0:\n''',
)
replace_once(
    "scripts/update_generation4_shadow_history.py",
    '''        "pending_rows": sum(1 for row in rows if row.get("status") != "evaluated"),\n        "eprocess": eprocess,\n''',
    '''        "pending_rows": sum(1 for row in rows if row.get("status") != "evaluated"),\n        "update_mode": "evaluate_only" if args.evaluate_only else "evaluate_and_append_latest",\n        "eprocess": eprocess,\n''',
)

# Evolution workflow: enforce exact holdout/role equivalence after a clean role rebuild.
replace_once(
    ".github/workflows/loto7_evolution.yml",
    '''            --progress-every 10 \\\n            --max-runtime-minutes 320 \\\n            --safe-exit-minutes 30\n      - name: Commit role ensemble backtest outputs\n''',
    '''            --progress-every 10 \\\n            --max-runtime-minutes 320 \\\n            --safe-exit-minutes 30\n      - name: Verify unified evaluator consistency\n        run: |\n          set -euo pipefail\n          python scripts/verify_evaluator_consistency.py \\\n            --holdout outputs/holdout/holdout_result.csv \\\n            --role outputs/role_ensemble/role_ensemble_backtest.csv \\\n            --output outputs/role_ensemble/evaluator_consistency.json \\\n            --unit-cost 300\n      - name: Commit role ensemble backtest outputs\n''',
)

# Generation 4 workflow: evaluate existing live evidence before adoption and retain it on rejection.
replace_once(
    ".github/workflows/loto7_generation4_run.yml",
    '''            --unit-cost 300 \\\n            --seed "${GITHUB_RUN_ID}"\n\n      - name: Build Generation 4 prediction through strict adoption gates\n''',
    '''            --unit-cost 300 \\\n            --seed "${GITHUB_RUN_ID}"\n\n      - name: Refresh existing production results before adoption decision\n        run: |\n          set -euo pipefail\n          python scripts/check_prediction_history_results.py \\\n            --history outputs/evolution_prediction_history.csv \\\n            --csv loto7.csv \\\n            --output outputs/evolution_prediction_history_result.txt\n\n          python scripts/update_generation4_shadow_history.py \\\n            --csv loto7.csv \\\n            --latest-shadow outputs/generation4/latest_shadow_predictions.json \\\n            --history outputs/generation4/shadow_history.csv \\\n            --summary outputs/generation4/champion_challenger_summary.json \\\n            --report outputs/generation4/champion_challenger_report.txt \\\n            --challenger generation4 \\\n            --champion beam_baseline \\\n            --promotion-threshold 20 \\\n            --min-evaluated-draws 30 \\\n            --evaluate-only\n\n      - name: Build Generation 4 prediction through strict adoption gates\n''',
)
replace_once(
    ".github/workflows/loto7_generation4_run.yml",
    '''          test -s outputs/generation4/strict_adoption_gate.json\n\n          if [ "$ADOPTION_ALLOWED" = "true" ]; then\n''',
    '''          test -s outputs/generation4/strict_adoption_gate.json\n          test -s outputs/evolution_prediction_history_result.txt\n          test -s outputs/generation4/shadow_history.csv\n          test -s outputs/generation4/champion_challenger_summary.json\n          test -s outputs/generation4/champion_challenger_report.txt\n\n          if [ "$ADOPTION_ALLOWED" = "true" ]; then\n''',
)
replace_once(
    ".github/workflows/loto7_generation4_run.yml",
    '''          cp -f outputs/generation4/null_strategy_league_report.txt "$SNAPSHOT/outputs/generation4/"\n\n          if [ "$ADOPTION_ALLOWED" = "true" ]; then\n''',
    '''          cp -f outputs/generation4/null_strategy_league_report.txt "$SNAPSHOT/outputs/generation4/"\n          cp -f outputs/evolution_prediction_history_result.txt "$SNAPSHOT/outputs/"\n          cp -f outputs/generation4/shadow_history.csv "$SNAPSHOT/outputs/generation4/"\n          cp -f outputs/generation4/champion_challenger_summary.json "$SNAPSHOT/outputs/generation4/"\n          cp -f outputs/generation4/champion_challenger_report.txt "$SNAPSHOT/outputs/generation4/"\n\n          if [ "$ADOPTION_ALLOWED" = "true" ]; then\n''',
)
replace_once(
    ".github/workflows/loto7_generation4_run.yml",
    '''            cp -f "$SNAPSHOT/outputs/generation4/null_strategy_league_report.txt" outputs/generation4/\n\n            git add -f \\\n''',
    '''            cp -f "$SNAPSHOT/outputs/generation4/null_strategy_league_report.txt" outputs/generation4/\n            cp -f "$SNAPSHOT/outputs/evolution_prediction_history_result.txt" outputs/evolution_prediction_history_result.txt\n            cp -f "$SNAPSHOT/outputs/generation4/shadow_history.csv" outputs/generation4/shadow_history.csv\n            cp -f "$SNAPSHOT/outputs/generation4/champion_challenger_summary.json" outputs/generation4/champion_challenger_summary.json\n            cp -f "$SNAPSHOT/outputs/generation4/champion_challenger_report.txt" outputs/generation4/champion_challenger_report.txt\n\n            git add -f \\\n''',
)
replace_once(
    ".github/workflows/loto7_generation4_run.yml",
    '''              outputs/generation4/null_strategy_league_summary.json \\\n              outputs/generation4/null_strategy_league_report.txt\n\n            if [ "$ADOPTION_ALLOWED" = "true" ]; then\n''',
    '''              outputs/generation4/null_strategy_league_summary.json \\\n              outputs/generation4/null_strategy_league_report.txt \\\n              outputs/evolution_prediction_history_result.txt \\\n              outputs/generation4/shadow_history.csv \\\n              outputs/generation4/champion_challenger_summary.json \\\n              outputs/generation4/champion_challenger_report.txt\n\n            if [ "$ADOPTION_ALLOWED" = "true" ]; then\n''',
)
replace_once(
    ".github/workflows/loto7_generation4_run.yml",
    '''            else\n              COMMIT_MESSAGE="Record rejected LOTO7 Generation 4 adoption [skip ci]"\n            fi\n''',
    '''            else\n              COMMIT_MESSAGE="Record rejected LOTO7 Generation 4 adoption and refresh live results [skip ci]"\n            fi\n''',
)

# Validation workflow watches and runs the new regression suite.
replace_once(
    ".github/workflows/loto7_validation_tests.yml",
    '''      - "scripts/update_generation4_shadow_history.py"\n      - "scripts/seal_generation4_prediction.py"\n''',
    '''      - "scripts/evaluation_core.py"\n      - "scripts/verify_evaluator_consistency.py"\n      - "scripts/update_generation4_shadow_history.py"\n      - "scripts/check_prediction_history_results.py"\n      - "holdout_evaluator.py"\n      - "scripts/backtest_role_ensemble.py"\n      - "scripts/seal_generation4_prediction.py"\n''',
)
replace_once(
    ".github/workflows/loto7_validation_tests.yml",
    '''      - "tests/test_strict_adoption_gates.py"\n''',
    '''      - "tests/test_strict_adoption_gates.py"\n      - "tests/test_evaluation_unification.py"\n''',
)
replace_once(
    ".github/workflows/loto7_validation_tests.yml",
    '''            scripts/null_strategy_league.py \\\n            scripts/update_generation4_shadow_history.py \\\n''',
    '''            scripts/null_strategy_league.py \\\n            scripts/evaluation_core.py \\\n            scripts/verify_evaluator_consistency.py \\\n            scripts/update_generation4_shadow_history.py \\\n            scripts/check_prediction_history_results.py \\\n            holdout_evaluator.py \\\n            scripts/backtest_role_ensemble.py \\\n''',
)
replace_once(
    ".github/workflows/loto7_validation_tests.yml",
    '''            tests/test_generation4_pipeline.py \\\n            tests/test_strict_adoption_gates.py\n''',
    '''            tests/test_generation4_pipeline.py \\\n            tests/test_strict_adoption_gates.py \\\n            tests/test_evaluation_unification.py\n''',
)
replace_once(
    ".github/workflows/loto7_validation_tests.yml",
    '''            tests.test_generation4_pipeline \\\n            tests.test_strict_adoption_gates -v\n''',
    '''            tests.test_generation4_pipeline \\\n            tests.test_strict_adoption_gates \\\n            tests.test_evaluation_unification -v\n''',
)

print("Applied evaluator unification and live-result refresh patch.")
