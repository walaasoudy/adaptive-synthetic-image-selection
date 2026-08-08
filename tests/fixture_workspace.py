"""Repository-controlled, per-test writable workspaces with unconditional cleanup."""
from __future__ import annotations

import itertools
import shutil
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "_runtime"
_COUNTER = itertools.count()


@contextmanager
def fixture_workspace(name: str):
    ROOT.mkdir(exist_ok=True)
    path = ROOT / f"{name}-{next(_COUNTER):04d}"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=False)
