#!/usr/bin/env python3
"""Compatibility entry point for Generation 5 Selection-Null finalist screening."""

from scripts._generation5_selection_evolver_impl import *  # noqa: F401,F403


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
