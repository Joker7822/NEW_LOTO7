#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for :mod:`loto7.evaluation.core`."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from loto7.evaluation.core import *  # noqa: F401,F403,E402
