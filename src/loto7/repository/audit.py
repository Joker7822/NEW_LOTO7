"""Fail-closed repository architecture audit for NEW_LOTO7."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Mapping

from loto7.repository.layout import RepositoryLayout, load_repository_layout

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_DIR = ROOT / ".github/workflows"
DEFAULT_JSON = ROOT / "docs/architecture/repository_architecture_guard.json"
DEFAULT_MARKDOWN = ROOT / "docs/architecture/repository_architecture_guard.md"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


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


def detect_git_writers(
    workflows: Mapping[str, str], outputs: Iterable[str]
) -> dict[str, List[str]]:
    writers: dict[str, List[str]] = defaultdict(list)
    for workflow, text in workflows.items():
        if "git push" not in text:
            continue
        for output in outputs:
            markers = production_write_markers(output)
            if all(marker in text for marker in markers):
                writers[output].append(workflow)
    return {key: sorted(value) for key, value in sorted(writers.items())}


def all_repository_paths() -> List[str]:
    excluded = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
    result: List[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in excluded for part in relative.parts):
            continue
        result.append(relative.as_posix())
    return sorted(result)


def is_thin_wrapper(path: Path, *, max_lines: int = 80) -> bool:
    text = path.read_text(encoding="utf-8")
    lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    delegates = (
        "from loto7." in text
        or "import loto7." in text
        or "_impl.main" in text
        or "runpy.run_module" in text
    )
    return len(lines) <= max_lines and delegates


def ownership_errors(
    layout: RepositoryLayout,
    workflows: Mapping[str, str],
    writers: Mapping[str, List[str]],
) -> List[str]:
    errors: List[str] = []
    for owner in layout.workflow_ownership:
        text = workflows.get(owner.workflow_path)
        if text is None:
            errors.append(
                f"Workflow owner is missing for {owner.responsibility}: {owner.workflow_path}"
            )
            continue
        actual = workflow_name(text, Path(owner.workflow_path).stem)
        if actual != owner.workflow_name:
            errors.append(
                f"Workflow name mismatch for {owner.responsibility}: "
                f"expected {owner.workflow_name!r}, got {actual!r}"
            )
        for output in owner.outputs:
            actual_writers = writers.get(output, [])
            if actual_writers != [owner.workflow_path]:
                errors.append(
                    f"{output} must be written only by {owner.workflow_path}; "
                    f"detected {actual_writers}"
                )
    return errors


def audit(layout: RepositoryLayout) -> dict[str, object]:
    paths = all_repository_paths()
    files = workflow_files()
    workflows = {rel(path): path.read_text(encoding="utf-8") for path in files}
    names: dict[str, List[str]] = defaultdict(list)
    errors: List[str] = []
    warnings: List[str] = []

    errors.extend(layout.validate_top_level(paths))

    for path, text in workflows.items():
        names[workflow_name(text, Path(path).stem)].append(path)
        if re.search(r"^\s*queue\s*:", text, re.MULTILINE):
            errors.append(f"Non-standard concurrency.queue is present in {path}")

    for name, owners in sorted(names.items()):
        if len(owners) > 1:
            errors.append(f"Duplicate workflow name {name!r}: {', '.join(sorted(owners))}")

    forbidden = [str(item) for item in layout.raw.get("forbidden_workflows", [])]
    for path in forbidden:
        if (ROOT / path).exists():
            errors.append(f"Forbidden or one-time workflow still exists: {path}")

    production_outputs = [str(item) for item in layout.raw.get("production_outputs", [])]
    writers = detect_git_writers(workflows, production_outputs)
    errors.extend(ownership_errors(layout, workflows, writers))

    for module in layout.required_package_modules:
        if not (ROOT / module).is_file():
            errors.append(f"Required package module is missing: {module}")

    root_python = sorted(path.name for path in ROOT.glob("*.py"))
    unexpected_root_python = sorted(set(root_python) - set(layout.allowed_root_python))
    if unexpected_root_python:
        errors.append(
            "Root Python files are not registered compatibility entrypoints: "
            + ", ".join(unexpected_root_python)
        )

    wrappers = [
        str(item)
        for item in layout.raw.get("compatibility_wrappers", [])
        if str(item).endswith(".py")
    ]
    for value in wrappers:
        path = ROOT / value
        if not path.is_file():
            errors.append(f"Registered compatibility wrapper is missing: {value}")
        elif not is_thin_wrapper(path):
            errors.append(f"Compatibility entrypoint is not a thin wrapper: {value}")

    temp_outputs = [
        "outputs/generation4/current_run_snapshot.json",
        "outputs/generation4/current_run_snapshot.txt",
        "outputs/generation4/dispatch_requested.txt",
    ]
    for path in temp_outputs:
        if (ROOT / path).exists():
            errors.append(f"Temporary execution-control output is tracked: {path}")

    output_files = sorted(
        rel(path) for path in (ROOT / "outputs").rglob("*") if path.is_file()
    )
    unclassified_outputs = [
        path for path in output_files if layout.classify_output(path) == "unclassified"
    ]
    if unclassified_outputs:
        warnings.append(
            f"{len(unclassified_outputs)} output files are not classified by repository policy"
        )

    legacy_files = [
        path
        for path in output_files
        if any(path.startswith(prefix) for prefix in layout.legacy_output_roots)
    ]
    if legacy_files:
        warnings.append(
            f"{len(legacy_files)} files remain in legacy output roots during compatibility migration"
        )

    payload: dict[str, object] = {
        "created_at": now_iso(),
        "status": "pass" if not errors else "fail",
        "schema_version": layout.schema_version,
        "workflow_count": len(files),
        "workflow_names": {key: sorted(value) for key, value in sorted(names.items())},
        "workflow_ownership": [
            {
                "responsibility": item.responsibility,
                "workflow_path": item.workflow_path,
                "workflow_name": item.workflow_name,
                "outputs": list(item.outputs),
            }
            for item in layout.workflow_ownership
        ],
        "production_writers": writers,
        "root_python_count": len(root_python),
        "root_python_files": root_python,
        "tracked_output_count": len(output_files),
        "unclassified_output_count": len(unclassified_outputs),
        "legacy_output_count": len(legacy_files),
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
    }
    return payload


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
        f"- Policy schema: **{payload.get('schema_version')}**",
        f"- Workflows: **{payload.get('workflow_count')}**",
        f"- Root Python compatibility files: **{payload.get('root_python_count')}**",
        f"- Tracked outputs: **{payload.get('tracked_output_count')}**",
        f"- Unclassified outputs: **{payload.get('unclassified_output_count')}**",
        f"- Errors: **{len(payload.get('errors', []))}**",
        f"- Warnings: **{len(payload.get('warnings', []))}**",
        "",
        "## Workflow ownership",
        "",
    ]
    for item in payload.get("workflow_ownership", []):
        if not isinstance(item, Mapping):
            continue
        lines.append(
            f"- **{item.get('responsibility')}**: "
            f"`{item.get('workflow_name')}` (`{item.get('workflow_path')}`)"
        )
    lines.extend(["", "## Errors", ""])
    errors = payload.get("errors", [])
    lines.extend(f"- {item}" for item in errors) if errors else lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    warnings = payload.get("warnings", [])
    lines.extend(f"- {item}" for item in warnings) if warnings else lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate NEW_LOTO7 repository architecture.")
    parser.add_argument("--config", default="config/repository_layout.json")
    parser.add_argument("--json", default=str(DEFAULT_JSON.relative_to(ROOT)))
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)

    layout = load_repository_layout(args.config, root=ROOT)
    payload = audit(layout)
    json_path = ROOT / args.json
    markdown_path = ROOT / args.markdown
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if args.report_only or payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
