#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply the safe Phase-1 NEW_LOTO7 repository restructuring.

This migration deliberately avoids moving root Python modules.  Those modules
form an import compatibility layer for existing training and Actions jobs.
Phase 1 establishes one production prediction owner, removes obsolete workflow
entry points, documents output ownership, and adds architecture enforcement.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"
CANONICAL = WORKFLOWS / "loto7_generation4_run.yml"
LEGACY_DUAL = WORKFLOWS / "loto7_dual_prediction.yml"
EVOLUTION = WORKFLOWS / "loto7_evolution.yml"
MODEL_SELF = WORKFLOWS / "loto7_model_self_evolution.yml"
AUDIT_WORKFLOW = WORKFLOWS / "repository_structure_audit.yml"
VALIDATION_WORKFLOW = WORKFLOWS / "loto7_validation_tests.yml"
REPORT_JSON = ROOT / "docs/architecture/repository_restructure_report.json"
REPORT_MD = ROOT / "docs/architecture/repository_restructure_report.md"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_if_changed(path: Path, content: str, changed: List[str]) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else None
    if old == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    changed.append(path.relative_to(ROOT).as_posix())


def remove_nonstandard_queue(text: str) -> str:
    return re.sub(r"^\s*queue\s*:\s*.*\n", "", text, flags=re.MULTILINE)


def restructure_canonical(changed: List[str]) -> None:
    text = CANONICAL.read_text(encoding="utf-8")
    header_pattern = re.compile(r"\Aname:.*?\n\non:\n.*?\n\npermissions:", re.DOTALL)
    header = '''name: LOTO7 Generation 4 Production

on:
  workflow_dispatch:
    inputs:
      null_simulations:
        description: "Number of null-strategy simulations"
        required: false
        default: "180"
  workflow_run:
    workflows:
      - LOTO7 Evolution Trainer
      - LOTO7 Model Self Evolution
      - LOTO7 Nested Walk Forward Validation
    types:
      - completed
  push:
    branches:
      - main
    paths:
      - ".github/workflows/loto7_generation4_run.yml"
      - "config/repository_layout.json"
      - "scripts/generation4_core.py"
      - "scripts/build_generation4_prediction.py"
      - "scripts/null_strategy_league.py"
      - "scripts/update_generation4_shadow_history.py"
      - "scripts/seal_generation4_prediction.py"
      - "scripts/finalize_generation4_report.py"
      - "tests/test_generation4_pipeline.py"

permissions:'''
    if not header_pattern.search(text):
        raise SystemExit("cannot locate canonical workflow header")
    text = header_pattern.sub(header, text, count=1)
    text = re.sub(
        r"concurrency:\n.*?\n\njobs:",
        "concurrency:\n  group: loto7-generation4-production-main\n  cancel-in-progress: true\n\njobs:",
        text,
        count=1,
        flags=re.DOTALL,
    )
    job_marker = "  generation4-full-run:\n"
    condition = (
        "  generation4-full-run:\n"
        "    if: ${{ github.event_name != 'workflow_run' || "
        "(github.event.workflow_run.conclusion == 'success' && "
        "github.event.workflow_run.head_branch == 'main') }}\n"
    )
    if condition not in text:
        if job_marker not in text:
            raise SystemExit("cannot locate canonical production job")
        text = text.replace(job_marker, condition, 1)
    text = remove_nonstandard_queue(text)
    write_if_changed(CANONICAL, text, changed)


def remove_legacy_dual(changed: List[str]) -> None:
    if LEGACY_DUAL.exists():
        LEGACY_DUAL.unlink()
        changed.append(LEGACY_DUAL.relative_to(ROOT).as_posix())


def remove_legacy_evolution_prediction_writer(changed: List[str]) -> None:
    text = EVOLUTION.read_text(encoding="utf-8")
    marker = "\n  final-prediction:\n"
    if marker in text:
        start = text.index(marker)
        end_marker = "\n  ml-stack:\n"
        end = text.find(end_marker, start)
        if end < 0:
            raise SystemExit("cannot find ml-stack boundary after final-prediction")
        text = text[:start] + text[end:]
    text = remove_nonstandard_queue(text)
    write_if_changed(EVOLUTION, text, changed)


def remove_model_self_prediction_writer(changed: List[str]) -> None:
    text = MODEL_SELF.read_text(encoding="utf-8")
    start_marker = "\n      - name: Build latest prediction from adopted model\n"
    if start_marker in text:
        start = text.index(start_marker)
        end_marker = "\n      - name: Commit model self-evolution outputs\n"
        end = text.find(end_marker, start)
        if end < 0:
            raise SystemExit("cannot find model self-evolution commit boundary")
        text = text[:start] + text[end:]
    filtered: List[str] = []
    forbidden_line_tokens = (
        "scripts/build_latest_prediction_from_best_model.py",
        "outputs/evolution_best_prediction.csv",
        "outputs/holdout/latest_prediction_report.txt",
    )
    for line in text.splitlines():
        if any(token in line for token in forbidden_line_tokens):
            continue
        filtered.append(line)
    text = "\n".join(filtered) + "\n"
    text = remove_nonstandard_queue(text)
    write_if_changed(MODEL_SELF, text, changed)


def clean_all_workflow_concurrency(changed: List[str]) -> None:
    for path in sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")]):
        text = path.read_text(encoding="utf-8")
        updated = remove_nonstandard_queue(text)
        write_if_changed(path, updated, changed)


def patch_audit_workflow(changed: List[str]) -> None:
    text = AUDIT_WORKFLOW.read_text(encoding="utf-8")
    additions = [
        '      - "scripts/check_repository_architecture.py"',
        '      - "scripts/restructure_repository.py"',
        '      - "config/repository_layout.json"',
    ]
    anchor = '      - "scripts/audit_repository_structure.py"'
    if anchor in text:
        for item in reversed(additions):
            if item not in text:
                text = text.replace(anchor, anchor + "\n" + item, 1)
    compile_anchor = "          python -m py_compile scripts/audit_repository_structure.py"
    replacement = (
        "          python -m py_compile scripts/audit_repository_structure.py \\\n"
        "            scripts/check_repository_architecture.py scripts/restructure_repository.py\n"
        "          python scripts/check_repository_architecture.py \\\n"
        "            --json docs/architecture/repository_architecture_guard.json \\\n"
        "            --markdown docs/architecture/repository_architecture_guard.md\n"
    )
    if "repository_architecture_guard.json" not in text and compile_anchor in text:
        text = text.replace(compile_anchor + "\n", replacement, 1)
    add_anchor = "          git add docs/architecture/repository_structure_audit.json \\\n                   docs/architecture/repository_structure_audit.md"
    add_replacement = (
        "          git add docs/architecture/repository_structure_audit.json \\\n"
        "                   docs/architecture/repository_structure_audit.md \\\n"
        "                   docs/architecture/repository_architecture_guard.json \\\n"
        "                   docs/architecture/repository_architecture_guard.md"
    )
    if add_anchor in text and "git add docs/architecture/repository_architecture_guard.json" not in text:
        text = text.replace(add_anchor, add_replacement, 1)
    artifact_anchor = "            docs/architecture/repository_structure_audit.md"
    if "            docs/architecture/repository_architecture_guard.json" not in text:
        text = text.replace(
            artifact_anchor,
            artifact_anchor
            + "\n            docs/architecture/repository_architecture_guard.json"
            + "\n            docs/architecture/repository_architecture_guard.md",
            1,
        )
    text = remove_nonstandard_queue(text)
    write_if_changed(AUDIT_WORKFLOW, text, changed)


def patch_validation_workflow(changed: List[str]) -> None:
    text = VALIDATION_WORKFLOW.read_text(encoding="utf-8")
    path_anchor = '      - "scripts/seal_generation4_prediction.py"'
    for item in (
        '      - "scripts/check_repository_architecture.py"',
        '      - "config/repository_layout.json"',
        '      - ".github/workflows/loto7_generation4_run.yml"',
    ):
        if item not in text and path_anchor in text:
            text = text.replace(path_anchor, path_anchor + "\n" + item, 1)
    compile_anchor = "            scripts/seal_generation4_prediction.py \\\n"
    if "            scripts/check_repository_architecture.py \\\n" not in text and compile_anchor in text:
        text = text.replace(
            compile_anchor,
            compile_anchor + "            scripts/check_repository_architecture.py \\\n",
            1,
        )
    test_anchor = "          python -m unittest \\\n"
    if "python scripts/check_repository_architecture.py" not in text and test_anchor in text:
        text = text.replace(
            test_anchor,
            "          python scripts/check_repository_architecture.py\n" + test_anchor,
            1,
        )
    text = remove_nonstandard_queue(text)
    write_if_changed(VALIDATION_WORKFLOW, text, changed)


def write_architecture_docs(changed: List[str]) -> None:
    layout = '''# NEW_LOTO7 Repository Layout

## Current stable layout

```text
.github/workflows/   GitHub Actions orchestration
config/              Repository and runtime policy
scripts/             CLI, validation, reporting and maintenance tools
tests/               Focused regression and leakage tests
docs/architecture/   Architecture, audits and migration decisions
outputs/              Versioned production evidence, state and diagnostics
root *.py             Compatibility layer for established training imports
```

## Production path

`LOTO7 Generation 4 Production` is the only workflow allowed to build and
commit the production prediction CSV, cumulative history and latest report.
Evolution workflows own model/state generation only. Nested validation owns
candidate promotion evidence only.

## Phase-2 package migration

Root Python implementations will move gradually to:

```text
src/loto7/data/
src/loto7/models/
src/loto7/validation/
src/loto7/portfolio/
src/loto7/reporting/
```

Every move must keep a root compatibility wrapper until all workflow imports,
unit tests and historical resume files are verified.
'''
    ownership = '''# Workflow Ownership

| Responsibility | Owner workflow | Committed output class |
|---|---|---|
| Dataset refresh, long evolution, holdout, role backtest | `LOTO7 Evolution Trainer` | model/state/diagnostics |
| Full-model standalone evolution | `LOTO7 Model Self Evolution` | model/state/diagnostics |
| Recent and Super candidate generation | `LOTO7 Recent Era Self Evolution` | guarded candidates/diagnostics |
| Sealed nested validation and promotion | `LOTO7 Nested Walk Forward Validation` | validation evidence/models |
| Production prediction, live history, e-process and seal | `LOTO7 Generation 4 Production` | production/evidence |
| Report aggregation | `LOTO7 TXT Reports` | derived reports |
| Architecture verification | `Repository Structure Audit` | architecture reports |

## Latest-state concurrency

Production prediction uses one stable concurrency group with
`cancel-in-progress: true`. A newer model state supersedes an older queued or
running prediction. GitHub Actions does not use a custom `queue:` key.
'''
    retention = '''# Output Retention Policy

## Production — retain in Git

- `outputs/evolution_best_prediction.csv`
- `outputs/evolution_prediction_history.csv`
- `outputs/evolution_prediction_history_result.txt`
- `outputs/holdout/latest_prediction_report.txt`

## Immutable evidence — retain in Git

- `outputs/generation4/latest_sealed_manifest.json`
- `outputs/generation4/sealed_index.json`
- `outputs/generation4/sealed/*`

## State — retain while resumable

Model evolution, Recent/Super and validation state remain versioned until a
checkpoint compaction workflow is introduced.

## Reproducible diagnostics — migrate to Actions artifacts

Large training frames, memory banks, backtest detail CSVs and binary model
experiments should be uploaded as Actions artifacts. Compact summaries and the
latest accepted model may remain in Git. No existing diagnostic is deleted in
Phase 1 because several workflows still consume these paths.
'''
    outputs_readme = '''# outputs/

This directory contains four different classes of generated material:

1. **production** — the current five-ticket prediction and live history;
2. **evidence** — SHA-256 sealed manifests and immutable prediction records;
3. **state** — resumable evolution and nested-validation checkpoints;
4. **diagnostics** — reproducible backtests, reports and experimental outputs.

The canonical ownership and retention rules are defined in
`config/repository_layout.json` and `docs/architecture/OUTPUT_RETENTION.md`.
Only `LOTO7 Generation 4 Production` may build committed production outputs.
'''
    write_if_changed(ROOT / "docs/architecture/REPOSITORY_LAYOUT.md", layout, changed)
    write_if_changed(ROOT / "docs/architecture/WORKFLOW_OWNERSHIP.md", ownership, changed)
    write_if_changed(ROOT / "docs/architecture/OUTPUT_RETENTION.md", retention, changed)
    write_if_changed(ROOT / "outputs/README.md", outputs_readme, changed)


def patch_readme_and_gitignore(changed: List[str]) -> None:
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    marker = "## Repository architecture"
    if marker not in text:
        text = text.rstrip() + '''

## Repository architecture

The canonical workflow and output ownership are documented in
[`docs/architecture/REPOSITORY_LAYOUT.md`](docs/architecture/REPOSITORY_LAYOUT.md).
Production predictions are generated only by **LOTO7 Generation 4 Production**
and are accompanied by SHA-256 sealed evidence.
'''
    write_if_changed(readme, text.rstrip() + "\n", changed)

    ignore = ROOT / ".gitignore"
    ignore_text = ignore.read_text(encoding="utf-8") if ignore.exists() else ""
    block = '''
# Repository architecture: transient and reproducible local files
__pycache__/
.pytest_cache/
.mypy_cache/
*.pyc
*.tmp
outputs/**/folds/
outputs/**/tmp/
outputs/**/.tmp/
'''
    if "# Repository architecture: transient" not in ignore_text:
        ignore_text = ignore_text.rstrip() + "\n" + block
    write_if_changed(ignore, ignore_text.rstrip() + "\n", changed)


def write_report(changed: List[str]) -> None:
    payload: Dict[str, object] = {
        "created_at": now_iso(),
        "phase": "phase_1_safe_restructure",
        "changed_paths": sorted(set(changed)),
        "production_workflow": ".github/workflows/loto7_generation4_run.yml",
        "removed_workflow": ".github/workflows/loto7_dual_prediction.yml",
        "root_python_moved": false,
        "tracked_outputs_deleted": false,
        "notes": [
            "Generation 4 Production is the sole production prediction builder.",
            "Legacy root modules remain as a compatibility layer.",
            "Large diagnostics are retained until artifact consumers are migrated.",
        ],
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Repository Restructure Report",
        "",
        f"Generated: `{payload['created_at']}`",
        "",
        "## Applied",
        "",
        "- One canonical Generation 4 production workflow",
        "- Legacy Dual workflow removed",
        "- Legacy Evolution and Model Self Evolution production writers removed",
        "- Non-standard `concurrency.queue` removed",
        "- Architecture guard and ownership policy added",
        "- Root modules preserved for compatibility",
        "- Existing model/state/diagnostic outputs preserved",
        "",
        "## Changed paths",
        "",
    ]
    lines.extend(f"- `{path}`" for path in sorted(set(changed)))
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path in (REPORT_JSON, REPORT_MD):
        relative = path.relative_to(ROOT).as_posix()
        if relative not in changed:
            changed.append(relative)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        raise SystemExit("Use --apply to perform the idempotent Phase-1 restructure.")

    changed: List[str] = []
    restructure_canonical(changed)
    remove_legacy_dual(changed)
    remove_legacy_evolution_prediction_writer(changed)
    remove_model_self_prediction_writer(changed)
    clean_all_workflow_concurrency(changed)
    patch_audit_workflow(changed)
    patch_validation_workflow(changed)
    write_architecture_docs(changed)
    patch_readme_and_gitignore(changed)
    write_report(changed)
    print(json.dumps({"changed_paths": sorted(set(changed))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
