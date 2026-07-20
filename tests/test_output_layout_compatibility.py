#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.migrate_output_layout import migrate


class OutputLayoutCompatibilityTests(unittest.TestCase):
    def test_resume_state_is_copied_without_removing_legacy_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy = root / "outputs/model_self_evolution/state.json"
            legacy.parent.mkdir(parents=True)
            legacy.write_text('{"generation": 7}\n', encoding="utf-8")
            production = root / "outputs/evolution_best_prediction.csv"
            production.parent.mkdir(parents=True, exist_ok=True)
            production.write_text("header\nvalue\n", encoding="utf-8")

            payload = migrate(root)

            self.assertEqual(payload["resume_compatibility"], "preserved")
            self.assertTrue(legacy.exists())
            self.assertTrue((root / "outputs/state/full/state.json").exists())
            self.assertTrue((root / "outputs/production/latest_prediction.csv").exists())


if __name__ == "__main__":
    unittest.main()
