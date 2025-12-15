#!/usr/bin/env python3
"""
run_tests.py - Run the test suite using pytest

This script is a tiny wrapper so you can run the test suite with:

    python run_tests.py [pytest args...]

It behaves like `python -m pytest` and passes any provided arguments
directly through to pytest. If `pytest` is not importable it falls
back to invoking `python -m pytest`.
"""
from __future__ import annotations
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    try:
        import pytest

        # Pass through any args the user supplied (or none, which is fine)
        return pytest.main(argv)
    except Exception:
        # If pytest isn't installed as a package, try calling it as a module
        return subprocess.call([sys.executable, "-m", "pytest"] + argv)


if __name__ == "__main__":
    raise SystemExit(main())
