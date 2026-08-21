"""Build a Jupyter notebook containing the repository's real implementation code."""
from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "notebooks" / "thesis_full_project_code.ipynb"

SECTIONS = [
    ("Shared utilities", "scripts/utils"),
    ("Data preparation", "scripts/data"),
    ("Stage 1 — SDXL LoRA training", "scripts/train"),
    ("Stage 2 — Synthetic generation", "scripts/generate"),
    ("Stage 3 — ASISM", "scripts/asism"),
    ("Stage 4 — Classifier training", "scripts/classify"),
    ("Stage 5 — Evaluation", "scripts/eval"),
    ("End-to-end smoke pipeline", "scripts/smoke"),
]


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str, *, tags: list[str] | None = None) -> dict:
    metadata = {"tags": tags} if tags else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    markdown(
        "# Full Thesis Project — Actual Source Code\n\n"
        "This notebook contains the **real Python implementation from the repository**, grouped by "
        "pipeline stage. Each source cell begins with its original file path. The files in the repository "
        "remain the source of truth; this notebook is a portable explanation/reference copy.\n\n"
        "> Source cells are tagged `project-source`. Do not execute a source cell in isolation: many scripts "
        "use command-line arguments and upstream artifact gates. Use the launcher cells below to run them "
        "from the repository root."
    ),
    markdown(
        "## Notebook launcher\n\n"
        "This helper runs the original project scripts safely from the repository root and preserves their "
        "normal command-line behavior. Heavy stages still require CheXpert, approved manifests, and GPU resources."
    ),
    code(
        "from pathlib import Path\n"
        "import subprocess\n"
        "import sys\n\n"
        "def find_repo_root(start=Path.cwd()):\n"
        "    for candidate in (start.resolve(), *start.resolve().parents):\n"
        "        if (candidate / 'scripts').is_dir() and (candidate / 'configs').is_dir():\n"
        "            return candidate\n"
        "    raise FileNotFoundError('Open this notebook from the project checkout.')\n\n"
        "REPO = find_repo_root()\n\n"
        "def run_script(relative_path, *args, env=None):\n"
        "    command = [sys.executable, str(REPO / relative_path), *map(str, args)]\n"
        "    print('Running:', subprocess.list2cmdline(command))\n"
        "    return subprocess.run(command, cwd=REPO, env=env, check=True)\n\n"
        "print('Repository:', REPO)"
    ),
    markdown(
        "## Quick local engineering validation\n\n"
        "These switches default to `False` to prevent accidental long runs. The smoke fixture is explicitly "
        "non-scientific and does not represent thesis results."
    ),
    code(
        "RUN_TESTS = False\n"
        "RUN_LOCAL_SMOKE = False\n\n"
        "if RUN_TESTS:\n"
        "    run_script('tests/run_all.py')\n"
        "if RUN_LOCAL_SMOKE:\n"
        "    run_script('scripts/smoke/run_smoke_pipeline.py', '--phase', 'local')\n"
        "if not (RUN_TESTS or RUN_LOCAL_SMOKE):\n"
        "    print('Nothing executed. Change a switch to True when ready.')"
    ),
]

for title, relative_dir in SECTIONS:
    cells.append(markdown(f"# {title}\n"))
    source_dir = REPO / relative_dir
    for path in sorted(source_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        relative = path.relative_to(REPO).as_posix()
        source = path.read_text(encoding="utf-8")
        cells.append(markdown(f"## `{relative}`\n"))
        cells.append(code(f"# SOURCE FILE: {relative}\n{source}", tags=["project-source"]))

cells.append(markdown("# Configuration files\n\nThese YAML files are the project's auditable configuration layer."))
for path in sorted((REPO / "configs").glob("*.yaml")):
    relative = path.relative_to(REPO).as_posix()
    content = path.read_text(encoding="utf-8")
    cells.append(markdown(f"## `{relative}`\n\n```yaml\n{content}\n```"))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Created {OUTPUT} with {len(cells)} cells")
