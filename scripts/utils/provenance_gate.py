"""Generic upstream-artifact provenance gate.

Generalizes the staleness-check idiom already duplicated across
scripts/data/03_preprocess_images.py (`check_manifest`) and scripts/train/train_lora_sdxl.py
(`validate_training_inputs`): compare an "expected" provenance dict (computed from the *current*
config/inputs) against what was actually recorded the last time an upstream artifact was
produced, and fail loudly on a mismatch rather than silently mixing incompatible outputs.

New Stage 2-5 scripts should call `validate_upstream_artifact()` instead of writing their own
version of this comparison, satisfying "each downstream stage must validate upstream artifacts
before execution" with one shared implementation. `03_preprocess_images.py`'s and
`train_lora_sdxl.py`'s existing bespoke checks are left as-is — they already work and carry extra
script-specific behavior (e.g. legacy-manifest adoption) that doesn't belong in a generic helper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.utils.manifest import read_json


class ProvenanceMismatch(SystemExit):
    """Raised (as a SystemExit subclass) when an upstream artifact's recorded provenance doesn't
    match what the current run expects — surfaces as a clean CLI failure, not a traceback."""


def validate_upstream_artifact(
    expected: dict[str, Any],
    manifest_path: str | Path,
    stage_name: str,
    allow_missing: bool = True,
) -> None:
    """Compare `expected` (the current run's provenance for some upstream artifact) against what
    was actually recorded at `manifest_path`.

    No-ops if they match, or if the manifest doesn't exist yet and `allow_missing=True` (first
    run — nothing to compare against). Raises `ProvenanceMismatch` otherwise.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        if allow_missing:
            return
        raise ProvenanceMismatch(
            f"[{stage_name}] Required upstream manifest is missing: {manifest_path}. "
            "Run the upstream stage first."
        )

    actual = read_json(manifest_path)
    if actual == expected:
        return

    raise ProvenanceMismatch(
        f"[{stage_name}] Upstream artifact at {manifest_path} was produced with different "
        "settings than this run expects — refusing to mix incompatible provenance.\n"
        f"Recorded: {json.dumps(actual, indent=2, sort_keys=True, default=str)}\n"
        f"Expected: {json.dumps(expected, indent=2, sort_keys=True, default=str)}\n"
        "Re-run the upstream stage with matching settings, or update this stage's config to match."
    )