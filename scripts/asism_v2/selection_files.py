"""Condition C (what ASISM v2 selected) and D (a random draw at C's per-class counts), as files.

C is the output of stopping.progressive_select on the whole safe pool: which images and how many are
both decided there. Nothing here sets, caps or adjusts a count.

D, one independent draw per Stage 4 seed, takes the same number of images per class at random from
the same safe pool, so C against D differs in which images only.

Three outcomes are all valid results and are recorded as such:
    subset   0 < C < every candidate: C and D are written; Stage 4 trains A, B, C, D.
    all      C is every candidate: C is B's training set and D would be too. No C or D file is
             written; Stage 4 trains A and B, and "ASISM kept everything" is the result.
    none     C is empty: C is A's training set. No file is written; Stage 4 trains A and B.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import write_new_json
from .pipeline import FittedRanking
from .preserve import sha256
from .stopping import progressive_select

C_NAME, D_TEMPLATE, TRAJECTORY_NAME = "c_selected.csv", "d_selected_seed{seed}.csv", "c_trajectory.csv"
MANIFEST_NAME = "asism_v2_learned_selection_manifest.json"
D_DRAW_SEED_BASE = 20261003          # fixed before any measurement; D never depends on a result
PROTOCOL_FOR_OUTCOME = {"subset": "asism_v2_learned", "all": "asism_v2_learned_all_or_none",
                        "none": "asism_v2_learned_all_or_none"}


def stage4_ids_sha256(ids) -> str:
    """The hash Stage 4 records for the synthetic images a run trained on."""
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def draw_matched_random(pool: pd.DataFrame, counts: dict[str, int], seed: int,
                        base: int = D_DRAW_SEED_BASE) -> list[str]:
    rng = np.random.default_rng([int(base), int(seed)])
    drawn = []
    for dx in sorted(counts):
        ids = np.array(sorted(pool.loc[pool["dx"].astype(str) == dx, "image_id"].astype(str)))
        drawn += list(rng.choice(ids, size=int(counts[dx]), replace=False))
    return drawn


def build_selection(fitted: FittedRanking, pool: pd.DataFrame, stage4_seeds: list[int],
                    n_candidates: int) -> dict:
    """`pool` is the safe pool with its four signals; `n_candidates` the pool before the safety
    filter (condition B's size)."""
    result = progressive_select(fitted, pool)
    chosen = list(result["selected"]["image_id"].astype(str))
    outcome = "none" if not chosen else "all" if len(chosen) == int(n_candidates) else "subset"
    draws = ({int(seed): draw_matched_random(pool, result["counts"], seed) for seed in stage4_seeds}
             if outcome == "subset" else {})
    return {"outcome": outcome, "c_ids": chosen, "d_ids": draws, "counts": dict(result["counts"]),
            "stops": result["stops"], "trajectory": result["trajectory"], "rule": result["rule"],
            "bootstrap_models": result["bootstrap_models"],
            "score_sources": {k: int(v) for k, v in result["selected"]["score_source"].value_counts().items()}}


def _write_csv(frame: pd.DataFrame, path: Path) -> str:
    with path.open("x", encoding="utf-8", newline="") as stream:      # exclusive: never replaced
        frame.to_csv(stream, index=False)
    return sha256(path)


def write_selection(out_dir: Path, selection: dict, pool: pd.DataFrame, namespace: str,
                    stage4_seeds: list[int], provenance: dict) -> dict:
    """Writes the manifest and, for a subset outcome, C, the D draws and C's trajectory."""
    out_dir = Path(out_dir)
    if (out_dir / MANIFEST_NAME).exists():
        raise ValueError(f"{out_dir / MANIFEST_NAME} exists: a selection is made once")
    out_dir.mkdir(parents=True, exist_ok=True)
    outcome = selection["outcome"]
    manifest = {
        "stage": "ham10000_asism_v2_learned_select", "namespace": namespace,
        "protocol": PROTOCOL_FOR_OUTCOME[outcome], "selection_outcome": outcome,
        "per_class": selection["counts"], "n_selected_c": len(selection["c_ids"]),
        "n_selected_d": len(selection["c_ids"]) if outcome == "subset" else 0,
        "safe_pool": int(len(pool)), "rule": selection["rule"], "stops": selection["stops"],
        "bootstrap_models": selection["bootstrap_models"], "score_sources": selection["score_sources"],
        "count_set_by": "the stopping rule alone; no count, ratio or cap is configured",
        "stage4_seeds": [int(s) for s in stage4_seeds], **provenance,
        "final_eval_heldout_read": False, "classifier_val_read": False,
    }
    if outcome == "subset":
        if "image_path" not in pool:
            raise ValueError("the pool has no image_path column: Stage 4 could not read C")
        indexed = pool.assign(image_id=pool["image_id"].astype(str)).set_index("image_id", drop=False)

        def rows(ids):
            return indexed.loc[sorted(ids), ["image_id", "image_path", "dx"]]

        files = {C_NAME: _write_csv(rows(selection["c_ids"]), out_dir / C_NAME)}
        overlap = {}
        for seed, ids in selection["d_ids"].items():
            name = D_TEMPLATE.format(seed=seed)
            files[name] = _write_csv(rows(ids), out_dir / name)
            overlap[str(seed)] = len(set(ids) & set(selection["c_ids"]))
        files[TRAJECTORY_NAME] = _write_csv(pd.DataFrame(selection["trajectory"]), out_dir / TRAJECTORY_NAME)
        manifest.update(files_sha256=files, d_draw_seed_base=D_DRAW_SEED_BASE, d_overlap_with_c=overlap,
                        c_ids_sha256=stage4_ids_sha256(selection["c_ids"]))
    else:
        manifest.update(files_sha256={}, c_ids_sha256=None,
                        conditions=("C is every candidate, so C = B and D = B; neither is built" if outcome == "all"
                                    else "C is empty, so C = A; D is not built"))
    write_new_json(out_dir / MANIFEST_NAME, manifest)
    return manifest
