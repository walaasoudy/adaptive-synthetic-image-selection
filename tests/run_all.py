#!/usr/bin/env python3
"""Run every test suite and report a single aggregate result.

Usage:
    python tests/run_all.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

# DISCOVERED, never hand-listed. This command is the documented production STOP gate
# (docs/runpod_stage2_to_5_commands.md: "STOP unless ... every test passes"), so a hardcoded list is
# a silent-coverage hazard: test_learned_asism.py — 85 tests covering the thesis's novel
# contribution — was absent from the previous hardcoded list, and this command reported a green
# "78 passed" while never running any of them.
def discover_suites(here: Path) -> list[str]:
    return sorted(path.name for path in here.glob("test_*.py"))


def main() -> int:
    here = Path(__file__).resolve().parent
    repo = here.parent
    TESTS = discover_suites(here)
    if not TESTS:
        print(f"NO TEST SUITES FOUND under {here} — refusing to report success.", flush=True)
        return 1
    print(f"Discovered {len(TESTS)} suite(s): {', '.join(TESTS)}\n", flush=True)

    total_passed, total_failed, failed_suites = 0, 0, []

    for name in TESTS:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
        path = here / name
        # Most suites carry their own `__main__` self-runner so the gate needs no test framework.
        # Suites that use pytest fixtures (monkeypatch, tmp_path) or pytest.raises cannot self-run
        # and MUST go through pytest — running them as bare scripts executes no tests at all, which
        # is how test_learned_asism.py previously contributed a silent zero to this gate.
        self_running = '__name__ == "__main__"' in path.read_text(encoding="utf-8")
        command = (
            [sys.executable, str(path)] if self_running
            else [sys.executable, "-m", "pytest", str(path), "-q"]
        )
        result = subprocess.run(
            command, cwd=repo, capture_output=True, text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        print(result.stdout, flush=True)
        if result.stderr.strip():
            print(result.stderr, flush=True)

        if not self_running and "No module named pytest" in (result.stdout + result.stderr):
            print(
                f"  {name} needs pytest, which is not installed. Install it "
                "(environment/requirements.txt) — this suite has NOT been run.",
                flush=True,
            )
            failed_suites.append(name)
            continue

        passed = re.search(r"(\d+) passed", result.stdout)
        failed = re.search(r"(\d+) failed", result.stdout)
        counted = False
        if passed:
            total_passed += int(passed.group(1)); counted = True
        if failed:
            total_failed += int(failed.group(1)); counted = True
        if not counted:
            # A suite that reports no countable result is treated as a failure, never as a silent
            # pass — an unparseable summary means we do not know that anything ran.
            print(f"  {name} produced no parseable test summary; treating as FAILED.", flush=True)
            failed_suites.append(name)
            continue
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
