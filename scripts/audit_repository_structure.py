#!/usr/bin/env python3
"""Compatibility CLI for the canonical repository inventory report."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from loto7.repository.inventory import main

if __name__ == "__main__":
    raise SystemExit(main())
