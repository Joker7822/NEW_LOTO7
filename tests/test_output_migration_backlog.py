from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from loto7.repository.layout import load_repository_layout
from scripts.build_output_migration_backlog import build_backlog, render_markdown


class OutputMigrationBacklogTests(unittest.TestCase):
    def test_backlog_lists_legacy_roots_and_unclassified_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir(parents=True)
            (root / "outputs/production").mkdir(parents=True)
            (root / "outputs/holdout").mkdir(parents=True)
            (root / "outputs/misc").mkdir(parents=True)
            (root / "outputs/production/latest.csv").write_text("a\n", encoding="utf-8")
            (root / "outputs/holdout/detail.csv").write_text("abcdef\n", encoding="utf-8")
            (root / "outputs/misc/unknown.bin").write_bytes(b"1234")

            policy = {
                "schema_version": 4,
                "allowed_top_level_directories": ["config", "outputs"],
                "allowed_root_files": [],
                "allowed_root_suffixes": [],
                "allowed_root_python": [],
                "package_policy": {"required_modules": []},
                "workflow_ownership": [],
                "output_layout": {
                    "canonical_roots": ["outputs/production/"],
                    "legacy_roots": ["outputs/holdout/"],
                },
                "output_classes": {
                    "production": ["outputs/production/"],
                    "diagnostics": ["outputs/holdout/"],
                },
            }
            (root / "config/repository_layout.json").write_text(
                json.dumps(policy),
                encoding="utf-8",
            )

            layout = load_repository_layout(root=root)
            payload = build_backlog(root, layout, top_limit=5)

            self.assertEqual(payload["total_output_files"], 3)
            self.assertEqual(payload["legacy_output_count"], 1)
            self.assertEqual(payload["unclassified_output_count"], 1)
            self.assertEqual(
                payload["unclassified_outputs"][0]["path"],
                "outputs/misc/unknown.bin",
            )
            self.assertEqual(
                payload["legacy_roots"]["outputs/holdout/"]["file_count"],
                1,
            )
            markdown = render_markdown(payload)
            self.assertIn("outputs/holdout/", markdown)
            self.assertIn("outputs/misc/unknown.bin", markdown)


if __name__ == "__main__":
    unittest.main()
