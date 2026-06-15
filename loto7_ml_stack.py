#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hotfix wrapper for loto7_ml_stack.py.

2026-06-16:
- Fixes the label threshold reversal without changing the original ML stack logic.
- label_4plus means main_match >= 4.
- label_5plus means main_match >= 5.

The original implementation is loaded from the pinned Git blob SHA below, patched in
memory, then executed. The blob SHA is immutable, so the runtime patch is stable.
"""
from __future__ import annotations

import base64
import json
import urllib.request

ORIGINAL_BLOB_SHA = "663fb6fb41a7565ed1709c871b4aa7e2de3ab9b6"
BLOB_API_URL = f"https://api.github.com/repos/Joker7822/NEW_LOTO7/git/blobs/{ORIGINAL_BLOB_SHA}"

OLD_LABELS = """        label_4plus=1 if main_match >= 5 else 0,
        label_5plus=1 if main_match >= 4 else 0,"""
NEW_LABELS = """        # Fixed 2026-06-16: label_4plus is 4+ main matches, label_5plus is 5+ main matches.
        label_4plus=1 if main_match >= 4 else 0,
        label_5plus=1 if main_match >= 5 else 0,"""

OLD_STATUS = """    status = {
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "imports": {"""
NEW_STATUS = """    status = {
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "label_definitions": {
            "label_4plus": "1 when main_match >= 4",
            "label_5plus": "1 when main_match >= 5",
        },
        "imports": {"""


def _load_original_source() -> str:
    with urllib.request.urlopen(BLOB_API_URL, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("encoding") != "base64":
        raise RuntimeError(f"unexpected blob encoding: {payload.get('encoding')}")
    return base64.b64decode(payload["content"]).decode("utf-8")


def _patched_source() -> str:
    source = _load_original_source()
    if OLD_LABELS not in source:
        raise RuntimeError("expected label block was not found in original loto7_ml_stack.py")
    source = source.replace(OLD_LABELS, NEW_LABELS, 1)
    if OLD_STATUS in source:
        source = source.replace(OLD_STATUS, NEW_STATUS, 1)
    return source


exec(compile(_patched_source(), "loto7_ml_stack_original_hotfixed.py", "exec"), globals())
