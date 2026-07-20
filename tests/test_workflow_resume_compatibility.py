#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from loto7.paths import BINDINGS, resumable_bindings


class WorkflowResumeCompatibilityTests(unittest.TestCase):
    def test_required_package_files_exist(self) -> None:
        required = [
            "pyproject.toml",
            "src/loto7/__init__.py",
            "src/loto7/paths.py",
            "src/loto7/evaluation/core.py",
            "src/loto7/evaluation/hit_metrics.py",
            "src/loto7/evaluation/robust.py",
            "src/loto7/validation/hit_rate_gate.py",
        ]
        for relative in required:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_legacy_wrappers_delegate_to_package(self) -> None:
        wrappers = {
            "scripts/evaluation_core.py": "loto7.evaluation.core",
            "scripts/robust_model_metrics.py": "loto7.evaluation.robust",
            "scripts/check_hit_rate_promotion.py": "loto7.validation.hit_rate_gate",
        }
        for relative, marker in wrappers.items():
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn(marker, text)

    def test_all_output_categories_and_resume_aliases_exist(self) -> None:
        self.assertEqual(
            {item.category for item in BINDINGS},
            {"production", "evidence", "state", "diagnostics"},
        )
        resumable = list(resumable_bindings())
        self.assertGreaterEqual(len(resumable), 4)
        for item in resumable:
            self.assertTrue(item.legacy.startswith("outputs/"))
            self.assertTrue(item.canonical.startswith("outputs/"))

    def test_all_workflow_python_references_exist(self) -> None:
        workflow_dir = ROOT / ".github/workflows"
        for workflow in sorted(workflow_dir.glob("*.yml")):
            text = workflow.read_text(encoding="utf-8")
            for token in text.replace("\\", " ").split():
                candidate = token.strip("'\"|()[]{}:,; ")
                if candidate.endswith(".py") and "/" in candidate:
                    self.assertTrue((ROOT / candidate).is_file(), f"{workflow.name}: {candidate}")


if __name__ == "__main__":
    unittest.main()
