#!/usr/bin/env python3
"""Compatibility CLI and import surface for the canonical architecture audit."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from loto7.repository.audit import *  # noqa: F401,F403
from loto7.repository.audit import main

if __name__ == "__main__":
    raise SystemExit(main())
