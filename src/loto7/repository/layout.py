"""Machine-readable repository layout policy for NEW_LOTO7."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

DEFAULT_POLICY_PATH = Path("config/repository_layout.json")


@dataclass(frozen=True)
class WorkflowOwnership:
    responsibility: str
    workflow_path: str
    workflow_name: str
    outputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class RepositoryLayout:
    schema_version: int
    allowed_top_level_directories: tuple[str, ...]
    allowed_root_files: tuple[str, ...]
    allowed_root_python: tuple[str, ...]
    required_package_modules: tuple[str, ...]
    canonical_output_roots: tuple[str, ...]
    legacy_output_roots: tuple[str, ...]
    workflow_ownership: tuple[WorkflowOwnership, ...]
    raw: Mapping[str, object]

    def owner_for(self, responsibility: str) -> WorkflowOwnership:
        for owner in self.workflow_ownership:
            if owner.responsibility == responsibility:
                return owner
        raise KeyError(f"unknown workflow responsibility: {responsibility}")

    def classify_output(self, path: str) -> str:
        normalized = path.replace("\\", "/")
        classes = self.raw.get("output_classes", {})
        if isinstance(classes, Mapping):
            for category, prefixes in classes.items():
                if not isinstance(prefixes, Sequence) or isinstance(prefixes, (str, bytes)):
                    continue
                for prefix in prefixes:
                    value = str(prefix)
                    if normalized == value.rstrip("/") or normalized.startswith(value):
                        return str(category)
        return "unclassified"

    def validate_top_level(self, paths: Iterable[str]) -> list[str]:
        errors: list[str] = []
        allowed_dirs = set(self.allowed_top_level_directories)
        allowed_files = set(self.allowed_root_files)
        for value in paths:
            normalized = value.replace("\\", "/").lstrip("./")
            if not normalized:
                continue
            if "/" in normalized:
                top = normalized.split("/", 1)[0]
                if top not in allowed_dirs:
                    errors.append(f"Unexpected top-level directory: {top}")
            elif normalized not in allowed_files:
                errors.append(f"Unexpected root file: {normalized}")
        return sorted(set(errors))


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def load_repository_layout(
    path: str | Path = DEFAULT_POLICY_PATH,
    *,
    root: str | Path = ".",
) -> RepositoryLayout:
    policy_path = Path(root) / Path(path)
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    package_policy = payload.get("package_policy", {})
    if not isinstance(package_policy, Mapping):
        package_policy = {}
    output_layout = payload.get("output_layout", {})
    if not isinstance(output_layout, Mapping):
        output_layout = {}

    owners: list[WorkflowOwnership] = []
    raw_owners = payload.get("workflow_ownership", [])
    if isinstance(raw_owners, list):
        for item in raw_owners:
            if not isinstance(item, Mapping):
                continue
            owners.append(
                WorkflowOwnership(
                    responsibility=str(item.get("responsibility", "")),
                    workflow_path=str(item.get("workflow_path", "")),
                    workflow_name=str(item.get("workflow_name", "")),
                    outputs=_strings(item.get("outputs")),
                )
            )

    return RepositoryLayout(
        schema_version=int(payload.get("schema_version", 0)),
        allowed_top_level_directories=_strings(payload.get("allowed_top_level_directories")),
        allowed_root_files=_strings(payload.get("allowed_root_files")),
        allowed_root_python=_strings(payload.get("allowed_root_python")),
        required_package_modules=_strings(package_policy.get("required_modules")),
        canonical_output_roots=_strings(output_layout.get("canonical_roots")),
        legacy_output_roots=_strings(output_layout.get("legacy_roots")),
        workflow_ownership=tuple(owners),
        raw=payload,
    )
