"""Provenance helpers: run_id composition, git/config hashing, JSON manifest read/write.

Used throughout Stage 1 (dataset splits, captions, checkpoints) so every artifact records enough
to be traced back to the exact code + config + data that produced it (docs/stage1_plan.md:
reproducibility, checkpoint metadata, handoff to Stage 2).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def get_git_commit_hash(repo_dir: str | Path = ".", short: bool = True) -> str:
    try:
        args = ["git", "-C", str(repo_dir), "rev-parse"]
        args.append("--short" if short else "HEAD")
        if short:
            args.append("HEAD")
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "nogit"


def hash_dict(data: dict[str, Any], length: int = 8) -> str:
    """Stable short hash of a (JSON-serializable) config dict, for use in run_ids and provenance."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:length]


def make_run_id(config: dict[str, Any], repo_dir: str | Path = ".") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    git_sha = get_git_commit_hash(repo_dir)
    config_hash = hash_dict(config)
    return f"{timestamp}_{git_sha}_{config_hash}"


def get_library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module_name in (
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "peft",
        "bitsandbytes",
    ):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[module_name] = "not_installed"
    return versions


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_checkpoint_metadata(
    step: int,
    epoch: int,
    config: dict[str, Any],
    split_manifest_hash: str,
    seed: int,
    repo_dir: str | Path = ".",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Metadata recorded alongside every checkpoint (docs/stage1_plan.md §10)."""
    metadata = {
        "step": step,
        "epoch": epoch,
        "wall_clock_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": get_git_commit_hash(repo_dir),
        "config": config,
        "config_hash": hash_dict(config),
        "split_manifest_hash": split_manifest_hash,
        "seed": seed,
        "library_versions": get_library_versions(),
    }
    if extra:
        metadata.update(extra)
    return metadata
