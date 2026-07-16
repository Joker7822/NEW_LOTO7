#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit the NEW_LOTO7 repository structure without third-party packages.

The audit is intentionally read-only. It inventories tracked files, workflow
entry points, Python references, generated artifacts, duplicate names and root
clutter, then writes machine-readable JSON and a concise Markdown report.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set

EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
GENERATED_PREFIXES = (
    "outputs/",
    "artifacts/",
    "reports/",
)
SOURCE_PREFIXES = (
    "scripts/",
    "tests/",
    ".github/workflows/",
    "docs/",
)
PYTHON_COMMAND_RE = re.compile(r"(?:python|python3)\s+(?:-m\s+)?([A-Za-z0-9_./-]+\.py)")
PATH_RE = re.compile(r"(?<![A-Za-z0-9_])((?:scripts|tests|outputs|docs)/[A-Za-z0-9_./-]+)")
IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_.]*)", re.MULTILINE)
WORKFLOW_NAME_RE = re.compile(r"^name:\s*(.+?)\s*$", re.MULTILINE)
WORKFLOW_REF_RE = re.compile(r"^\s*-\s+([^#\n]+?)\s*$", re.MULTILINE)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_files(root: Path) -> Iterable[Path]:
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
        for filename in sorted(files):
            path = Path(current) / filename
            if path.is_file():
                yield path


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            return ""
    except OSError:
        return ""


def top_level(path: str) -> str:
    return path.split("/", 1)[0] if "/" in path else "<root>"


def classify(path: str) -> str:
    if path.startswith(".github/workflows/"):
        return "workflow"
    if path.startswith("tests/"):
        return "test"
    if path.startswith("scripts/"):
        return "script"
    if path.startswith("docs/"):
        return "documentation"
    if path.startswith("outputs/"):
        return "generated_output"
    if path.endswith(".py"):
        return "root_python"
    if path.endswith((".json", ".csv", ".txt")):
        return "root_data_or_report"
    return "other"


def module_to_candidates(module: str) -> List[str]:
    parts = module.split(".")
    return ["/".join(parts) + ".py", "/".join(parts) + "/__init__.py"]


def audit(root: Path) -> Dict[str, object]:
    paths = [relative(path, root) for path in iter_files(root)]
    path_set = set(paths)
    sizes = {path: (root / path).stat().st_size for path in paths}
    categories = Counter(classify(path) for path in paths)
    top_levels = Counter(top_level(path) for path in paths)
    suffixes = Counter(Path(path).suffix.lower() or "<none>" for path in paths)

    root_files = sorted(path for path in paths if "/" not in path)
    root_python = sorted(path for path in root_files if path.endswith(".py"))
    workflows = sorted(path for path in paths if path.startswith(".github/workflows/") and path.endswith((".yml", ".yaml")))
    scripts = sorted(path for path in paths if path.endswith(".py") and not path.startswith("tests/"))
    tests = sorted(path for path in paths if path.startswith("tests/") and path.endswith(".py"))
    outputs = sorted(path for path in paths if path.startswith("outputs/"))

    workflow_details: List[Dict[str, object]] = []
    workflow_referenced_paths: Set[str] = set()
    workflow_names: Dict[str, List[str]] = defaultdict(list)
    workflow_triggers: Dict[str, List[str]] = defaultdict(list)
    for workflow in workflows:
        text = read_text(root / workflow)
        name_match = WORKFLOW_NAME_RE.search(text)
        name = name_match.group(1).strip().strip('"\'') if name_match else Path(workflow).stem
        workflow_names[name].append(workflow)
        referenced = sorted({match.group(1) for match in PYTHON_COMMAND_RE.finditer(text) if match.group(1) in path_set})
        referenced.extend(sorted({p for p in PATH_RE.findall(text) if p in path_set and p not in referenced}))
        workflow_referenced_paths.update(referenced)
        trigger_tokens = []
        for token in ("workflow_dispatch", "workflow_run", "push", "schedule", "pull_request"):
            if re.search(rf"^\s*{re.escape(token)}\s*:", text, re.MULTILINE):
                trigger_tokens.append(token)
                workflow_triggers[token].append(workflow)
        invalid_queue = bool(re.search(r"^\s*queue\s*:", text, re.MULTILINE))
        workflow_details.append(
            {
                "path": workflow,
                "name": name,
                "triggers": trigger_tokens,
                "referenced_paths": referenced,
                "contains_nonstandard_concurrency_queue": invalid_queue,
                "line_count": len(text.splitlines()),
            }
        )

    imported_paths: Set[str] = set()
    file_references: Dict[str, Set[str]] = defaultdict(set)
    for source in scripts + tests:
        text = read_text(root / source)
        for module in IMPORT_RE.findall(text):
            for candidate in module_to_candidates(module):
                if candidate in path_set:
                    imported_paths.add(candidate)
                    file_references[candidate].add(source)
        for candidate in PATH_RE.findall(text):
            if candidate in path_set:
                file_references[candidate].add(source)

    referenced_python = workflow_referenced_paths | imported_paths
    likely_orphan_python = sorted(
        path
        for path in scripts
        if path not in referenced_python
        and path not in {"setup.py"}
        and not path.endswith("/__init__.py")
        and Path(path).name not in {"conftest.py"}
    )

    stem_groups: Dict[str, List[str]] = defaultdict(list)
    for path in paths:
        stem_groups[Path(path).name.lower()].append(path)
    duplicate_filenames = {
        name: sorted(group)
        for name, group in sorted(stem_groups.items())
        if len(group) > 1 and name not in {"__init__.py", "readme.md"}
    }

    model_files = sorted(
        path for path in paths
        if path.endswith(".json") and any(token in Path(path).name.lower() for token in ("model", "genome", "state"))
    )
    history_files = sorted(
        path for path in paths
        if any(token in Path(path).name.lower() for token in ("history", "report", "summary"))
        and path.endswith((".csv", ".json", ".txt"))
    )
    large_files = [
        {"path": path, "bytes": sizes[path]}
        for path in sorted(paths, key=lambda item: sizes[item], reverse=True)[:30]
    ]
    duplicate_workflow_names = {name: files for name, files in workflow_names.items() if len(files) > 1}
    invalid_queue_workflows = [item["path"] for item in workflow_details if item["contains_nonstandard_concurrency_queue"]]

    recommendations: List[Dict[str, object]] = []
    if len(root_python) > 8:
        recommendations.append({
            "priority": "P0",
            "issue": "root_python_clutter",
            "detail": f"Root contains {len(root_python)} Python files.",
            "action": "Keep compatibility wrappers at root and move implementations into src/loto7 or scripts by responsibility.",
        })
    if len(outputs) > 40:
        recommendations.append({
            "priority": "P0",
            "issue": "tracked_generated_outputs",
            "detail": f"Repository tracks {len(outputs)} files under outputs/.",
            "action": "Separate immutable prediction evidence from reproducible intermediate outputs; retain only latest, sealed, and compact history files.",
        })
    if invalid_queue_workflows:
        recommendations.append({
            "priority": "P0",
            "issue": "invalid_workflow_concurrency_key",
            "detail": ", ".join(invalid_queue_workflows),
            "action": "Remove non-standard concurrency.queue keys and use documented group/cancel-in-progress only.",
        })
    if duplicate_workflow_names:
        recommendations.append({
            "priority": "P0",
            "issue": "duplicate_workflow_names",
            "detail": json.dumps(duplicate_workflow_names, ensure_ascii=False),
            "action": "Give each workflow a unique name and one production owner for each output file.",
        })
    if likely_orphan_python:
        recommendations.append({
            "priority": "P1",
            "issue": "possibly_orphaned_python",
            "detail": f"{len(likely_orphan_python)} Python files have no detected workflow/import reference.",
            "action": "Review before archiving; static detection can miss dynamic calls.",
        })
    recommendations.extend([
        {
            "priority": "P1",
            "issue": "generation_ownership",
            "detail": "Multiple generations and workflows can write prediction outputs.",
            "action": "Make Generation 4 the sole writer of production prediction outputs; legacy workflows write only candidate or diagnostic artifacts.",
        },
        {
            "priority": "P1",
            "issue": "package_boundaries",
            "detail": "Training, evaluation, prediction, workflow helpers and reporting are mixed.",
            "action": "Adopt src/loto7/{data,models,validation,portfolio,reporting} and keep scripts as thin CLI entry points.",
        },
        {
            "priority": "P2",
            "issue": "output_retention",
            "detail": "State, reports, model candidates and sealed evidence share outputs/.",
            "action": "Split outputs into production/, validation/, state/, diagnostics/, sealed/ and define retention rules.",
        },
    ])

    return {
        "created_at": now_iso(),
        "repository_root": str(root),
        "file_count": len(paths),
        "directory_file_counts": dict(sorted(top_levels.items())),
        "category_counts": dict(sorted(categories.items())),
        "suffix_counts": dict(sorted(suffixes.items())),
        "root_files": root_files,
        "root_python_files": root_python,
        "workflow_count": len(workflows),
        "workflows": workflow_details,
        "workflow_trigger_counts": {key: len(value) for key, value in sorted(workflow_triggers.items())},
        "duplicate_workflow_names": duplicate_workflow_names,
        "nonstandard_concurrency_queue_workflows": invalid_queue_workflows,
        "python_file_count": len(scripts),
        "test_file_count": len(tests),
        "workflow_referenced_python": sorted(path for path in workflow_referenced_paths if path.endswith(".py")),
        "possibly_orphaned_python": likely_orphan_python,
        "tracked_output_count": len(outputs),
        "tracked_outputs": outputs,
        "model_or_state_files": model_files,
        "history_report_summary_files": history_files,
        "duplicate_filenames": duplicate_filenames,
        "largest_files": large_files,
        "recommendations": recommendations,
    }


def render_markdown(payload: Mapping[str, object]) -> str:
    lines = [
        "# NEW_LOTO7 Repository Structure Audit",
        "",
        f"Generated: `{payload.get('created_at')}`",
        "",
        "## Summary",
        "",
        f"- Tracked files: **{payload.get('file_count')}**",
        f"- Workflows: **{payload.get('workflow_count')}**",
        f"- Python implementation/CLI files: **{payload.get('python_file_count')}**",
        f"- Test files: **{payload.get('test_file_count')}**",
        f"- Tracked files under `outputs/`: **{payload.get('tracked_output_count')}**",
        f"- Root Python files: **{len(payload.get('root_python_files', []))}**",
        "",
        "## Directory distribution",
        "",
        "| Location | Files |",
        "|---|---:|",
    ]
    for name, count in dict(payload.get("directory_file_counts", {})).items():
        lines.append(f"| `{name}` | {count} |")

    lines.extend(["", "## Workflows", "", "| Workflow | Triggers | Lines | Notes |", "|---|---|---:|---|"])
    for item in payload.get("workflows", []):
        if not isinstance(item, dict):
            continue
        notes = []
        if item.get("contains_nonstandard_concurrency_queue"):
            notes.append("non-standard `queue` key")
        lines.append(
            f"| `{item.get('name')}`<br>`{item.get('path')}` | "
            f"{', '.join(item.get('triggers', [])) or '-'} | {item.get('line_count')} | "
            f"{'; '.join(notes) or '-'} |"
        )

    lines.extend(["", "## Highest-priority findings", ""])
    for item in payload.get("recommendations", []):
        if not isinstance(item, dict):
            continue
        lines.extend([
            f"### {item.get('priority')} — {item.get('issue')}",
            "",
            str(item.get("detail", "")),
            "",
            f"**Recommended action:** {item.get('action', '')}",
            "",
        ])

    orphans = payload.get("possibly_orphaned_python", [])
    lines.extend(["## Possibly unreferenced Python files", ""])
    if orphans:
        lines.extend(f"- `{path}`" for path in orphans)
    else:
        lines.append("- None detected")

    lines.extend(["", "## Largest tracked files", "", "| File | Bytes |", "|---|---:|"])
    for item in payload.get("largest_files", [])[:20]:
        if isinstance(item, dict):
            lines.append(f"| `{item.get('path')}` | {item.get('bytes')} |")

    lines.extend([
        "",
        "> Static-reference detection is conservative. A file listed as possibly unreferenced must be reviewed before deletion.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit NEW_LOTO7 repository structure.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json", default="docs/architecture/repository_structure_audit.json")
    parser.add_argument("--markdown", default="docs/architecture/repository_structure_audit.md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    payload = audit(root)
    json_path = root / args.json
    markdown_path = root / args.markdown
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "file_count": payload["file_count"],
        "workflow_count": payload["workflow_count"],
        "tracked_output_count": payload["tracked_output_count"],
        "possibly_orphaned_python_count": len(payload["possibly_orphaned_python"]),
        "json": args.json,
        "markdown": args.markdown,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
