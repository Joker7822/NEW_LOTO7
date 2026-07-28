"""Repository inventory reporting aligned with architecture policy v4."""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from loto7.repository.layout import RepositoryLayout, load_repository_layout

ROOT = Path(__file__).resolve().parents[3]
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
}
WORKFLOW_NAME_RE = re.compile(r"^name:\s*(.+?)\s*$", re.MULTILINE)
PYTHON_COMMAND_RE = re.compile(
    r"(?:python|python3)\s+(?:-m\s+)?([A-Za-z0-9_./-]+\.py)"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_files(root: Path) -> Iterable[Path]:
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(directory for directory in dirs if directory not in EXCLUDED_DIRS)
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
        except (UnicodeDecodeError, OSError):
            return ""
    except OSError:
        return ""


def category(path: str, layout: RepositoryLayout) -> str:
    if path.startswith(".github/workflows/"):
        return "workflow"
    if path.startswith("src/loto7/") and path.endswith(".py"):
        return "package"
    if path.startswith("tests/") and path.endswith(".py"):
        return "test"
    if path.startswith("scripts/") and path.endswith(".py"):
        return "cli_or_compatibility"
    if path.startswith("docs/"):
        return "documentation"
    if path.startswith("config/"):
        return "configuration"
    if path.startswith("outputs/"):
        return f"output:{layout.classify_output(path)}"
    if "/" not in path and path.endswith(".py"):
        return "root_compatibility"
    return "other"


def audit(root: Path, layout: RepositoryLayout) -> dict[str, object]:
    paths = [relative(path, root) for path in iter_files(root)]
    path_set = set(paths)
    sizes = {path: (root / path).stat().st_size for path in paths}
    categories = Counter(category(path, layout) for path in paths)
    top_levels = Counter(path.split("/", 1)[0] if "/" in path else "<root>" for path in paths)

    workflows = sorted(
        path
        for path in paths
        if path.startswith(".github/workflows/") and path.endswith((".yml", ".yaml"))
    )
    workflow_details: list[dict[str, object]] = []
    workflow_names: dict[str, list[str]] = defaultdict(list)
    referenced_python: set[str] = set()
    for workflow in workflows:
        text = read_text(root / workflow)
        match = WORKFLOW_NAME_RE.search(text)
        name = match.group(1).strip().strip("\"'") if match else Path(workflow).stem
        workflow_names[name].append(workflow)
        referenced = sorted(
            {
                item.group(1)
                for item in PYTHON_COMMAND_RE.finditer(text)
                if item.group(1) in path_set
            }
        )
        referenced_python.update(referenced)
        triggers = [
            token
            for token in (
                "workflow_dispatch",
                "workflow_run",
                "push",
                "schedule",
                "pull_request",
            )
            if re.search(rf"^\s*{re.escape(token)}\s*:", text, re.MULTILINE)
        ]
        workflow_details.append(
            {
                "path": workflow,
                "name": name,
                "triggers": triggers,
                "referenced_python": referenced,
                "line_count": len(text.splitlines()),
            }
        )

    package_files = sorted(
        path for path in paths if path.startswith("src/loto7/") and path.endswith(".py")
    )
    script_files = sorted(
        path for path in paths if path.startswith("scripts/") and path.endswith(".py")
    )
    root_python = sorted(path for path in paths if "/" not in path and path.endswith(".py"))
    outputs = sorted(path for path in paths if path.startswith("outputs/"))
    output_counts = Counter(layout.classify_output(path) for path in outputs)

    registry_payload = json.loads(
        (root / "config/workflow_registry.json").read_text(encoding="utf-8")
    )
    registry_items = registry_payload.get("workflows", [])
    registered_paths = {
        str(item.get("path"))
        for item in registry_items
        if isinstance(item, Mapping)
    }

    recommendations: list[dict[str, object]] = []
    unregistered = sorted(set(workflows) - registered_paths)
    if unregistered:
        recommendations.append(
            {
                "priority": "P0",
                "issue": "unregistered_workflows",
                "detail": ", ".join(unregistered),
                "action": "Register every Workflow in config/workflow_registry.json.",
            }
        )

    unexpected_root = sorted(set(root_python) - set(layout.allowed_root_python))
    if unexpected_root:
        recommendations.append(
            {
                "priority": "P0",
                "issue": "unexpected_root_python",
                "detail": ", ".join(unexpected_root),
                "action": "Move implementations into src/loto7 and keep only tested wrappers.",
            }
        )

    legacy_outputs = [
        path
        for path in outputs
        if any(path.startswith(prefix) for prefix in layout.legacy_output_roots)
    ]
    if legacy_outputs:
        recommendations.append(
            {
                "priority": "P1",
                "issue": "legacy_output_aliases",
                "detail": f"{len(legacy_outputs)} files remain under legacy output roots.",
                "action": "Convert active workflows to canonical paths before removing aliases.",
            }
        )

    implementation_roots = [
        path
        for path in root_python
        if path not in set(layout.raw.get("compatibility_wrappers", []))
    ]
    if implementation_roots:
        recommendations.append(
            {
                "priority": "P1",
                "issue": "remaining_root_implementations",
                "detail": f"{len(implementation_roots)} allowlisted root implementations remain.",
                "action": "Migrate one responsibility at a time with import and Resume tests.",
            }
        )

    return {
        "created_at": now_iso(),
        "policy_schema_version": layout.schema_version,
        "file_count": len(paths),
        "directory_file_counts": dict(sorted(top_levels.items())),
        "category_counts": dict(sorted(categories.items())),
        "root_files": sorted(path for path in paths if "/" not in path),
        "root_python_files": root_python,
        "package_python_count": len(package_files),
        "script_python_count": len(script_files),
        "workflow_count": len(workflows),
        "registered_workflow_count": len(registered_paths),
        "workflows": workflow_details,
        "duplicate_workflow_names": {
            name: values
            for name, values in sorted(workflow_names.items())
            if len(values) > 1
        },
        "workflow_referenced_python": sorted(referenced_python),
        "tracked_output_count": len(outputs),
        "output_class_counts": dict(sorted(output_counts.items())),
        "largest_files": [
            {"path": path, "bytes": sizes[path]}
            for path in sorted(paths, key=lambda item: sizes[item], reverse=True)[:30]
        ],
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
        f"- Policy schema: **{payload.get('policy_schema_version')}**",
        f"- Tracked files: **{payload.get('file_count')}**",
        f"- Workflows: **{payload.get('workflow_count')}**",
        f"- Registered Workflows: **{payload.get('registered_workflow_count')}**",
        f"- Package Python files: **{payload.get('package_python_count')}**",
        f"- Script/compatibility Python files: **{payload.get('script_python_count')}**",
        f"- Root compatibility Python files: **{len(payload.get('root_python_files', []))}**",
        f"- Tracked outputs: **{payload.get('tracked_output_count')}**",
        "",
        "## Output classes",
        "",
        "| Class | Files |",
        "|---|---:|",
    ]
    for name, count in dict(payload.get("output_class_counts", {})).items():
        lines.append(f"| `{name}` | {count} |")

    lines.extend(["", "## Workflow inventory", "", "| Workflow | Triggers | Lines |", "|---|---|---:|"])
    for item in payload.get("workflows", []):
        if not isinstance(item, Mapping):
            continue
        lines.append(
            f"| `{item.get('name')}`<br>`{item.get('path')}` | "
            f"{', '.join(item.get('triggers', [])) or '-'} | {item.get('line_count')} |"
        )

    lines.extend(["", "## Migration recommendations", ""])
    recommendations = payload.get("recommendations", [])
    if not recommendations:
        lines.append("- None")
    for item in recommendations:
        if not isinstance(item, Mapping):
            continue
        lines.extend(
            [
                f"### {item.get('priority')} — {item.get('issue')}",
                "",
                str(item.get("detail", "")),
                "",
                f"**Action:** {item.get('action', '')}",
                "",
            ]
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory NEW_LOTO7 repository structure.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="config/repository_layout.json")
    parser.add_argument("--json", default="docs/architecture/repository_structure_audit.json")
    parser.add_argument("--markdown", default="docs/architecture/repository_structure_audit.md")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    layout = load_repository_layout(args.config, root=root)
    payload = audit(root, layout)
    json_path = root / args.json
    markdown_path = root / args.markdown
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
