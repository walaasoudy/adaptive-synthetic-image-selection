"""Global, cross-stage experiment registry: one parquet row per run, so every Stage 1-5 run is
queryable in a single place (outputs/experiments/experiment_registry.parquet) and nothing
overwrites a previous run's record.

Generalizes the per-artifact manifest provenance already produced by every stage (see
scripts/utils/manifest.py: make_run_id, build_checkpoint_metadata) into something queryable
*across* runs, rather than only inspectable one manifest file at a time. `parent_experiment_id`
lets a Stage 4 classifier run point at the Stage 3 selection run it consumed, which points at the
Stage 2 generation run, which points at the Stage 1 checkpoint — the full provenance chain as one
table.

Concurrency note: appends/updates are implemented as read-existing -> modify -> atomic replace
(temp file + os.replace, matching the write_json idiom in manifest.py). This is correct and
sufficient for this project's actual execution model (a single researcher, a single GPU,
sequential runs) and deliberately does not add a locking layer or a real database, which would be
over-engineering for a use case with no concurrent writers.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.utils.manifest import get_git_commit_hash, hash_dict, make_run_id

DEFAULT_REGISTRY_RELATIVE_PATH = Path("outputs") / "experiments" / "experiment_registry.parquet"

REGISTRY_COLUMNS = [
    "experiment_id",
    "parent_experiment_id",
    "timestamp",
    "stage",
    "git_sha",
    "config_hash",
    "dataset_version",
    "checkpoint_path",
    "execution_time_seconds",
    "metrics",
    "status",
]


def resolve_registry_path(project_root: str | Path | None = None) -> Path:
    """Default registry location: <PROJECT_ROOT>/outputs/experiments/experiment_registry.parquet."""
    root = Path(project_root) if project_root is not None else Path(os.environ.get("PROJECT_ROOT", "."))
    return root / DEFAULT_REGISTRY_RELATIVE_PATH


def _atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        df.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_row(registry_path: Path, row: dict[str, Any]) -> None:
    if registry_path.exists():
        existing = pd.read_parquet(registry_path)
        updated = pd.concat([existing, pd.DataFrame([row], columns=REGISTRY_COLUMNS)], ignore_index=True)
    else:
        updated = pd.DataFrame([row], columns=REGISTRY_COLUMNS)
    _atomic_write_parquet(registry_path, updated)


def _update_row(registry_path: Path, experiment_id: str, updates: dict[str, Any]) -> None:
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")
    df = pd.read_parquet(registry_path)
    mask = df["experiment_id"] == experiment_id
    if not mask.any():
        raise KeyError(f"experiment_id {experiment_id!r} not found in {registry_path}")
    for key, value in updates.items():
        df.loc[mask, key] = json.dumps(value, sort_keys=True, default=str) if isinstance(value, (dict, list)) else value
    _atomic_write_parquet(registry_path, df)


class ExperimentRun:
    """Context manager registering one experiment-registry row per run.

    Usage:
        with ExperimentRun(stage="stage2_generation", config=cfg_dict,
                            parent_experiment_id=stage1_checkpoint_run_id) as run:
            ... do the stage's work ...
            run.set_checkpoint_path(output_dir)
            run.set_metrics({"num_images": 5000})

    On normal exit, status is recorded as "completed"; on any exception, "failed" (the exception
    still propagates — this context manager never swallows errors).
    """

    def __init__(
        self,
        stage: str,
        config: dict[str, Any],
        parent_experiment_id: str | None = None,
        dataset_version: str | None = None,
        experiment_id: str | None = None,
        registry_path: str | Path | None = None,
        repo_dir: str | Path = ".",
    ) -> None:
        self.stage = stage
        self.config = config
        self.parent_experiment_id = parent_experiment_id
        self.dataset_version = dataset_version
        self.experiment_id = experiment_id or make_run_id(config, repo_dir=repo_dir)
        self.registry_path = Path(registry_path) if registry_path is not None else resolve_registry_path()
        self.repo_dir = repo_dir
        self.checkpoint_path: str | None = None
        self.metrics: dict[str, Any] = {}
        self.status = "running"
        self._start_time: float | None = None

    def set_checkpoint_path(self, path: str | Path) -> None:
        self.checkpoint_path = str(path)

    def set_metrics(self, metrics: dict[str, Any]) -> None:
        self.metrics.update(metrics)

    def __enter__(self) -> "ExperimentRun":
        self._start_time = time.monotonic()
        row = {
            "experiment_id": self.experiment_id,
            "parent_experiment_id": self.parent_experiment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": self.stage,
            "git_sha": get_git_commit_hash(self.repo_dir),
            "config_hash": hash_dict(self.config),
            "dataset_version": self.dataset_version,
            "checkpoint_path": None,
            "execution_time_seconds": None,
            "metrics": "{}",
            "status": self.status,
        }
        _append_row(self.registry_path, row)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        elapsed = time.monotonic() - self._start_time if self._start_time is not None else None
        self.status = "failed" if exc_type is not None else "completed"
        _update_row(
            self.registry_path,
            self.experiment_id,
            {
                "checkpoint_path": self.checkpoint_path,
                "execution_time_seconds": elapsed,
                "metrics": self.metrics,
                "status": self.status,
            },
        )
        return False