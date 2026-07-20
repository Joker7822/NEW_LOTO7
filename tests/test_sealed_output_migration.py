#!/usr/bin/env python3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.migrate_output_layout import migrate


class SealedOutputMigrationTests(unittest.TestCase):
    def test_sealed_json_and_hash_are_copied(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy = root / "outputs/generation4/sealed"
            legacy.mkdir(parents=True)
            (legacy / "prediction.json").write_text("{}\n", encoding="utf-8")
            (legacy / "prediction.sha256").write_text("abc\n", encoding="utf-8")
            migrate(root)
            canonical = root / "outputs/evidence/generation4/sealed"
            self.assertTrue((canonical / "prediction.json").exists())
            self.assertTrue((canonical / "prediction.sha256").exists())


if __name__ == "__main__":
    unittest.main()
