#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility CLI for :mod:`loto7.evaluation.robust`."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from loto7.evaluation.robust import *  # noqa: F401,F403,E402
from loto7.evaluation.robust import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
