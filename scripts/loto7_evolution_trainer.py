#!/usr/bin/env python3
"""Compatibility shim for direct execution of scripts under ``scripts/``.

When Python executes ``python scripts/<name>.py``, ``sys.path[0]`` is the
``scripts`` directory rather than the repository root. Several hardened
Generation 5 modules import the repository-root ``loto7_evolution_trainer.py``.
This shim makes that root module resolvable without requiring callers to set
PYTHONPATH manually.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_MODULE = ROOT / "loto7_evolution_trainer.py"
ROOT_ALIAS = "_new_loto7_root_evolution_trainer"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

module = sys.modules.get(ROOT_ALIAS)
if module is None:
    spec = importlib.util.spec_from_file_location(ROOT_ALIAS, ROOT_MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load root evolution trainer: {ROOT_MODULE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[ROOT_ALIAS] = module
    spec.loader.exec_module(module)

for name, value in vars(module).items():
    if name in {"__name__", "__loader__", "__package__", "__spec__"}:
        continue
    globals()[name] = value
