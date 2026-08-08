#!/usr/bin/env python3
"""Run every test suite and report a single aggregate result.

Usage:
    python tests/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

TESTS = [
    "test_metrics.py",
    "test_asism_signals.py",
    "test_pipeline_contracts.py",
    "test_provenance_contracts.py",
    "test_roundtrip_contracts.py",
]


def main() -> int:
    here = Path(__file__).resolve().parent
    repo = here.parent

    total_passed, total_failed, failed_suites = 0, 0, []

    for name in TESTS:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
        result = subprocess.run(
            [sys.executable, str(here / name)], cwd=repo, capture_output=True, text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        print(result.stdout, flush=True)
        if result.stderr.strip():
            print(result.stderr, flush=True)

        summary = [line for line in result.stdout.splitlines() if "passed," in line]
        if summary:
            parts = summary[-1].split()
            total_passed += int(parts[0])
            total_failed += int(parts[2])
        if result.returncode != 0:
            failed_suites.append(name)

    print(f"\n{'=' * 70}", flush=True)
    print(f"TOTAL: {total_passed} passed, {total_failed} failed across {len(TESTS)} suites", flush=True)
    if failed_suites:
        print(f"FAILED SUITES: {failed_suites}", flush=True)
    print(f"{'=' * 70}", flush=True)
    return 1 if failed_suites else 0


if __name__ == "__main__":
    raise SystemExit(main())
