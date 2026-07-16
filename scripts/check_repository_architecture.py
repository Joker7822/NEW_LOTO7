#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate NEW_LOTO7 repository architecture invariants.

The guard intentionally focuses on ownership and workflow safety.  It does not
move legacy root modules automatically because those modules are imported by
long-running training and validation jobs.  Structural migrations must retain
compatibility wrappers and regression coverage.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/repository_layout.json"
DEFAULT_JSON = ROOT / "docs/architecture/repository_architecture_guard.json"
DEFAULT_MARKDOWN = ROOT / "docs/architecture/repository_architecture_guard.md"
WORKFLOW_DIR = ROOT / ".github/workflows"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def workflow_files() -> List[Path]:
    return sorted([*WORKFLOW_DIR.glob("*.yml"), *WORKFLOW_DIR.glob("*.yaml")])


def workflow_name(text: str, fallback: str) -> str:
    match = re.search(r"^name:\s*(.+?)\s*$", text, re.MULTILINE)
    return match.group(1).strip().strip("\"'") if match else fallback


def production_write_markers(path: str) -> List[str]:
    if path.endswith("evolution_best_prediction.csv"):
        return [f"--prediction {path}"]
    if path.endswith("latest_prediction_report.txt"):
        return [f"--prediction-report {path}"]
    if path.endswith("evolution_prediction_history.csv"):
        return ["scripts/update_prediction_history.py", f"--history {path}"]
    if path.endswith("evolution_prediction_history_result.txt"):
        return ["scripts/check_prediction_history_results.py", f"--output {path}"]
    return [path]


def detect_production_writers(
    workflows: Mapping[str, str], production_outputs: Iterable[str]
) -> Dict[str, List[str]]:
    writers: Dict[str, List[str]] = defaultdict(list)
    for workflow, text in workflows.items():
        for output in production_outputs:
            markers = production_write_markers(output)
            if all(marker in text for marker in markers):
                writers[output].append(workflow)
    return {key: sorted(value) for key, value in sorted(writers.items())}


def render_markdown(payload: Mapping[str, object]) -> str:
    lines = [
        "# Repository Architecture Guard",
        "",
        f"Generated: `{payload.get('created_at')}`",
        "",
        f"Status: **{payload.get('status')}**",
        "",
        "## Summary",
        "",
        f"- Workflows: **{payload.get('workflow_count')}**",
        f"- Root Python files: **{payload.get('root_python_count')}**",
        f"- Tracked output files: **{payload.get('tracked_output_count')}**",
        f"- Errors: **{len(payload.get('errors', []))}**",
        f"- Warnings: **{len(payload.get('warnings', []))}**",
        "",
        "## Production output writers",
        "",
    ]
    writers = payload.get("production_writers", {})
    if isinstance(writers, dict):
        for output, owners in writers.items():
            lines.append(f"- `{output}`: {', '.join(f'`{owner}`' for owner in owners) or 'none'}")
    lines.extend(["", "## Errors", ""])
    errors = payload.get("errors", [])
    lines.extend(f"- {item}" for item in errors) if errors else lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    warnings = payload.get("warnings", [])
    lines.extend(f"- {item}" for item in warnings) if warnings else lines.append("- None")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- Generation 4 Production is the only workflow that may build committed production predictions.",
            "- Evolution workflows produce models, candidates, state, and diagnostics only.",
            "- Sealed manifests are immutable evidence and are not treated as disposable diagnostics.",
            "- Root Python implementations remain a compatibility layer until package migration tests exist.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--json", default=str(DEFAULT_JSON.relative_to(ROOT)))
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    canonical = str(config["production_workflow"])
    canonical_name = str(config["production_workflow_name"])
    forbidden = [str(item) for item in config.get("forbidden_workflows", [])]
    outputs = [str(item) for item in config.get("production_outputs", [])]
    required_sources = [str(item) for item in config.get("workflow_run_sources", [])]

    files = workflow_files()
    workflows = {rel(path): read_text(path) for path in files}
    names: Dict[str, List[str]] = defaultdict(list)
    errors: List[str] = []
    warnings: List[str] = []

    for path, text in workflows.items():
        names[workflow_name(text, Path(path).stem)].append(path)
        if re.search(r"^\s*queue\s*:", text, re.MULTILINE):
            errors.append(f"Non-standard concurrency.queue is present in {path}")

    for name, paths in sorted(names.items()):
        if len(paths) > 1:
            errors.append(f"Duplicate workflow name {name!r}: {', '.join(sorted(paths))}")

    for path in forbidden:
        if (ROOT / path).exists():
            errors.append(f"Forbidden or one-time workflow still exists: {path}")

    canonical_text = workflows.get(canonical)
    if canonical_text is None:
        errors.append(f"Canonical production workflow is missing: {canonical}")
    else:
        actual_name = workflow_name(canonical_text, Path(canonical).stem)
        if actual_name != canonical_name:
            errors.append(
                f"Canonical workflow name mismatch: expected {canonical_name!r}, got {actual_name!r}"
            )
        if "workflow_run:" not in canonical_text:
            errors.append("Canonical production workflow has no workflow_run trigger")
        for source in required_sources:
            if f"- {source}" not in canonical_text:
                errors.append(f"Canonical workflow is missing upstream trigger: {source}")
        if "cancel-in-progress: true" not in canonical_text:
            warnings.append("Canonical workflow should use latest-state-wins concurrency")

    writers = detect_production_writers(workflows, outputs)
    for output in outputs:
        owners = writers.get(output, [])
        if owners != [canonical]:
            errors.append(
                f"Production output {output} must have exactly one builder ({canonical}); detected {owners}"
            )

    temp_outputs = [
        "outputs/generation4/current_run_snapshot.json",
        "outputs/generation4/current_run_snapshot.txt",
        "outputs/generation4/dispatch_requested.txt",
    ]
    for path in temp_outputs:
        if (ROOT / path).exists():
            errors.append(f"Temporary execution-control output is tracked: {path}")

    root_python = sorted(path.name for path in ROOT.glob("*.py"))
    if len(root_python) > 8:
        warnings.append(
            f"Root still contains {len(root_python)} Python modules; retain as compatibility layer until Phase 2 migration"
        )

    tracked_outputs = sorted(path for path in (ROOT / "outputs").rglob("*") if path.is_file())
    if len(tracked_outputs) > 80:
        warnings.append(
            f"outputs/ contains {len(tracked_outputs)} tracked files; reproducible diagnostics should move to Actions artifacts"
        )

    payload: Dict[str, object] = {
        "created_at": now_iso(),
        "status": "pass" if not errors else "fail",
        "config": rel(config_path),
        "canonical_production_workflow": canonical,
        "workflow_count": len(files),
        "workflow_names": {key: sorted(value) for key, value in sorted(names.items())},
        "production_writers": writers,
        "root_python_count": len(root_python),
        "root_python_files": root_python,
        "tracked_output_count": len(tracked_outputs),
        "errors": errors,
        "warnings": warnings,
    }

    json_path = ROOT / args.json
    markdown_path = ROOT / args.markdown
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if args.report_only or not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
