"""Provenance-checked input loading shared by Learned ASISM stages."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from scripts.utils.artifact_contracts import asism_score_provenance, require_score_artifact, stage2_paths
from scripts.utils.config import load_named_config
from scripts.utils.manifest import read_json


def unsafe_candidates(image_ids, iqa: pd.DataFrame | None, similarity: pd.DataFrame | None) -> dict[str, str]:
    """image_id -> rejection reason for every candidate that fails a technical safety check.

    Fail closed: a candidate with no IQA row, a missing/false iqa_valid, no similarity row, or a
    missing/true novelty_is_near_duplicate is unsafe. Pass None to skip a check (safety flag off).
    """
    reasons: dict[str, str] = {}
    ids = [str(image_id) for image_id in image_ids]
    if iqa is not None:
        valid = dict(zip(iqa["image_id"].astype(str), iqa["iqa_valid"]))
        for image_id in ids:
            value = valid.get(image_id)
            if value is None or pd.isna(value) or not bool(value):
                reasons[image_id] = "invalid_iqa_safety_gate"
    if similarity is not None:
        duplicate = dict(zip(similarity["image_id"].astype(str), similarity["novelty_is_near_duplicate"]))
        for image_id in ids:
            value = duplicate.get(image_id)
            if image_id not in reasons and (value is None or pd.isna(value) or bool(value)):
                reasons[image_id] = "near_duplicate_safety_gate"
    return reasons


def _safety_frame(cfg, signal: str, column: str, expected: dict) -> pd.DataFrame:
    path = Path(cfg.paths.scores_dir) / f"{signal}_scores.parquet"
    if not path.is_file():
        raise SystemExit(
            f"SAFETY GATE: {path} is required even when {signal} did not survive Go/No-Go — the "
            f"{column} safety check runs on every candidate pool.\n"
            f"Run: python scripts/asism/01_compute_signals.py --signal {signal}"
        )
    require_score_artifact(path, signal, expected)
    frame = pd.read_parquet(path)
    if column not in frame.columns:
        raise SystemExit(f"SAFETY GATE: {path} has no {column!r} column; refusing an unfiltered candidate pool.")
    return frame[["image_id", column]]


def load_candidate_pool(cfg):
    stage2_cfg = load_named_config("stage2_generation.yaml", "stage2")
    # Hash the Stage 3 config exactly as 01_compute_signals.py and 02_gonogo.py recorded it: freshly
    # loaded, before callers rewrite cfg.paths to namespaced locations. Hashing the caller's
    # rewritten cfg made every learned stage (04-09) reject valid score artifacts.
    pristine = load_named_config("stage3_asism.yaml", "stage3")
    expected = asism_score_provenance(pristine, stage2_cfg, str(cfg.split_namespace))
    gonogo = read_json(Path(cfg.paths.gonogo_report))
    surviving = list(gonogo["surviving_signals"])
    if not surviving:
        raise SystemExit("No signals survived Go/No-Go; Learned ASISM cannot train.")
    merged = None
    for signal in surviving:
        path = Path(cfg.paths.scores_dir) / f"{signal}_scores.parquet"
        require_score_artifact(path, signal, expected)
        frame = pd.read_parquet(path)
        merged = frame if merged is None else merged.merge(frame, on="image_id", how="inner")
    if merged is None or merged.empty:
        raise SystemExit("No non-empty candidate images remain after merging admitted signals.")
    # Technical safety gate (invalid images, memorized near-copies of real patients) runs HERE, once,
    # so every learned stage 04-09 sees the same filtered pool — and it reads the IQA and similarity
    # artifacts directly, so a signal dropped by Go/No-Go can never switch the gate off.
    safety = cfg.learned_asism.safety
    reasons = unsafe_candidates(
        merged["image_id"],
        _safety_frame(cfg, "iqa", "iqa_valid", expected) if bool(safety.reject_invalid_iqa) else None,
        _safety_frame(cfg, "similarity", "novelty_is_near_duplicate", expected)
        if bool(safety.reject_near_duplicates) else None,
    )
    counts = {reason: sum(1 for value in reasons.values() if value == reason) for reason in sorted(set(reasons.values()))}
    print(f"Safety gate: removed {len(reasons)}/{len(merged)} candidates {counts}", flush=True)
    merged = merged[~merged["image_id"].astype(str).isin(reasons)].reset_index(drop=True)
    if merged.empty:
        raise SystemExit("SAFETY GATE: every candidate image failed the technical safety checks.")
    intended = {}
    manifest = stage2_paths(stage2_cfg, str(cfg.split_namespace))["manifest_path"]
    with open(manifest, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                intended[str(row["image_id"])] = row["intended_label_vector"]
    merged["__stratum"] = [
        "|".join(sorted(label for label, value in intended.get(str(image_id), {}).items() if int(value) == 1))
        or "__no_finding__" for image_id in merged["image_id"]
    ]
    return merged, intended, surviving
