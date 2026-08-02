#!/usr/bin/env python3
"""Build an actionable backlog for migrating legacy and unclassified outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from loto7.repository.layout import RepositoryLayout, load_repository_layout


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def output_files(root: Path) -> list[Path]:
    outputs = root / "outputs"
    if not outputs.exists():
        return []
    return sorted(path for path in outputs.rglob("*") if path.is_file())


def build_backlog(
    root: Path,
    layout: RepositoryLayout,
    *,
    top_limit: int = 50,
) -> dict[str, object]:
    files = output_files(root)
    records = [
        {
            "path": relative(path, root),
            "bytes": path.stat().st_size,
            "class": layout.classify_output(relative(path, root)),
            "suffix": path.suffix.lower() or "<none>",
        }
        for path in files
    ]

    class_counts = Counter(str(item["class"]) for item in records)
    unclassified = [item for item in records if item["class"] == "unclassified"]

    legacy_roots: dict[str, dict[str, object]] = {}
    all_legacy: list[dict[str, object]] = []
    for prefix in layout.legacy_output_roots:
        matching = [item for item in records if str(item["path"]).startswith(prefix)]
        all_legacy.extend(matching)
        suffix_counts = Counter(str(item["suffix"]) for item in matching)
        legacy_roots[prefix] = {
            "file_count": len(matching),
            "total_bytes": sum(int(item["bytes"]) for item in matching),
            "suffix_counts": dict(sorted(suffix_counts.items())),
            "largest_files": sorted(
                matching,
                key=lambda item: (-int(item["bytes"]), str(item["path"])),
            )[:10],
        }

    unique_legacy = {str(item["path"]): item for item in all_legacy}
    largest_legacy = sorted(
        unique_legacy.values(),
        key=lambda item: (-int(item["bytes"]), str(item["path"])),
    )[: max(1, top_limit)]

    return {
        "kind": "loto7_output_migration_backlog",
        "schema_version": 1,
        "created_at": now_iso(),
        "policy_schema_version": layout.schema_version,
        "total_output_files": len(records),
        "total_output_bytes": sum(int(item["bytes"]) for item in records),
        "class_counts": dict(sorted(class_counts.items())),
        "legacy_output_count": len(unique_legacy),
        "legacy_output_bytes": sum(
            int(item["bytes"]) for item in unique_legacy.values()
        ),
        "legacy_roots": legacy_roots,
        "unclassified_output_count": len(unclassified),
        "unclassified_outputs": unclassified,
        "largest_legacy_outputs": largest_legacy,
        "migration_order": [
            "Preserve production outputs and sealed evidence.",
            "Move resumable state with read-fallback compatibility before deleting legacy state.",
            "Move diagnostic summaries before large reproducible detail files.",
            "Delete a legacy file only after all workflows and resume readers use the canonical path.",
        ],
    }


def render_markdown(payload: Mapping[str, object]) -> str:
    lines = [
        "# Output Migration Backlog",
        "",
        f"Generated: `{payload.get('created_at')}`",
        "",
        "## Summary",
        "",
        f"- Total tracked outputs: **{payload.get('total_output_files')}**",
        f"- Legacy outputs: **{payload.get('legacy_output_count')}**",
        f"- Unclassified outputs: **{payload.get('unclassified_output_count')}**",
        "",
        "## Legacy roots",
        "",
        "| Root | Files | Bytes |",
        "|---|---:|---:|",
    ]
    roots = payload.get("legacy_roots", {})
    if isinstance(roots, Mapping):
        for prefix, raw in roots.items():
            item = raw if isinstance(raw, Mapping) else {}
            lines.append(
                f"| `{prefix}` | {item.get('file_count', 0)} | {item.get('total_bytes', 0)} |"
            )

    lines.extend(["", "## Unclassified outputs", ""])
    unclassified = payload.get("unclassified_outputs", [])
    if isinstance(unclassified, list) and unclassified:
        for item in unclassified:
            if isinstance(item, Mapping):
                lines.append(f"- `{item.get('path')}` ({item.get('bytes', 0)} bytes)")
    else:
        lines.append("- None")

    lines.extend(["", "## Largest legacy outputs", ""])
    largest = payload.get("largest_legacy_outputs", [])
    if isinstance(largest, list) and largest:
        for item in largest[:25]:
            if isinstance(item, Mapping):
                lines.append(f"- `{item.get('path')}` ({item.get('bytes', 0)} bytes)")
    else:
        lines.append("- None")

    lines.extend(["", "## Migration order", ""])
    order = payload.get("migration_order", [])
    if isinstance(order, list):
        lines.extend(f"1. {item}" for item in order)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="config/repository_layout.json")
    parser.add_argument(
        "--json", default="docs/architecture/output_migration_backlog.json"
    )
    parser.add_argument(
        "--markdown", default="docs/architecture/output_migration_backlog.md"
    )
    parser.add_argument("--top-limit", type=int, default=50)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    layout = load_repository_layout(args.config, root=root)
    payload = build_backlog(root, layout, top_limit=max(1, args.top_limit))

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
