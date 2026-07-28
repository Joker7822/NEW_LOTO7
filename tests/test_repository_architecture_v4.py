from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from loto7.repository.audit import is_thin_wrapper
from loto7.repository.layout import load_repository_layout


class RepositoryArchitectureV4Tests(unittest.TestCase):
    def test_policy_has_required_owners_and_generation5(self) -> None:
        layout = load_repository_layout()
        self.assertGreaterEqual(layout.schema_version, 4)
        responsibilities = {item.responsibility for item in layout.workflow_ownership}
        self.assertIn("production_prediction_publication", responsibilities)
        self.assertIn("automatic_model_promotion", responsibilities)
        self.assertIn("canonical_output_sync", responsibilities)
        generation5 = layout.owner_for("automatic_model_promotion")
        self.assertEqual(
            generation5.workflow_name,
            "LOTO7 Generation 5 Precision Evolution",
        )

    def test_canonical_output_roots_are_classified(self) -> None:
        layout = load_repository_layout()
        samples = {
            "outputs/production/latest_prediction.csv": "production",
            "outputs/evidence/generation4/sealed_index.json": "evidence",
            "outputs/state/full/state.json": "state",
            "outputs/diagnostics/generation5/promotion_report.txt": "diagnostics",
        }
        for path, expected in samples.items():
            self.assertEqual(layout.classify_output(path), expected)

    def test_unexpected_top_level_directory_fails(self) -> None:
        layout = load_repository_layout()
        errors = layout.validate_top_level(
            ["README.md", "src/loto7/paths.py", "random_dir/file.txt"]
        )
        self.assertEqual(errors, ["Unexpected top-level directory: random_dir"])

    def test_registered_architecture_script_is_thin_wrapper(self) -> None:
        self.assertTrue(
            is_thin_wrapper(Path("scripts/check_repository_architecture.py"))
        )

    def test_policy_json_is_machine_readable(self) -> None:
        payload = json.loads(
            Path("config/repository_layout.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["schema_version"], 4)
        self.assertIn("workflow_ownership", payload)
        self.assertIn("compatibility_wrappers", payload)
        self.assertEqual(payload["output_layout"]["version"], 3)

    def test_workflow_registry_covers_every_workflow(self) -> None:
        registry = json.loads(
            Path("config/workflow_registry.json").read_text(encoding="utf-8")
        )
        items = registry["workflows"]
        registered_paths = {item["path"] for item in items}
        actual_paths = {
            path.as_posix()
            for path in Path(".github/workflows").glob("*.y*ml")
        }
        self.assertEqual(registered_paths, actual_paths)
        self.assertEqual(len(items), len({item["name"] for item in items}))

        for item in items:
            text = Path(item["path"]).read_text(encoding="utf-8")
            match = re.search(r"^name:\s*(.+?)\s*$", text, re.MULTILINE)
            self.assertIsNotNone(match, item["path"])
            actual_name = match.group(1).strip().strip("\"'")
            self.assertEqual(item["name"], actual_name)


if __name__ == "__main__":
    unittest.main()
