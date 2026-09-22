#!/usr/bin/env python3
"""Follow-up (post-test): the Stage 3 v2 selection — conditions C2 and D2 for a Stage 4 re-run.

WHY THIS EXISTS. In ham-final-v1, condition C lost to B partly through its budget rule (rare classes
capped near 50 synthetic images) and a POOLED quality floor that removed 49% of mel. v2 changes those
two rules, and only those, as pre-registered on 2026-09-22 in configs/ham10000_v2_selection.yaml:

    5b  target_c = max(0, N - real_c), N = 300          fill each class to N images in total
    6a  quality floor p25 computed WITHIN each class     not over the pooled candidates

THE ORDER, fixed here:
    1. safety   The same gate as v1 (load_candidate_pool): invalid images and near-copies of real
                patients are gone before anything else happens.
    2. floor    Per class: candidates below their own class's p25 score are removed.
    3. top-k    Per class: the highest-scoring survivors, up to the class's target. A class with
                fewer survivors than its target takes all of them; the shortfall is reported.
    D2          Per class: C2's count, drawn uniformly from the SAFE pool (step 1 only), seeded.

THE SCORE IS AN INPUT, NOT A CHOICE MADE HERE. Which score orders the candidates depends on the
noise-floor diagnostic's last round (scripts/followup/ham10000_proxy_noise_floor.py): a retrained
ranking network if the proxy utility proved reliable, a selection without a learned utility if not.
This script takes the score file and column it is given and records both, with the file's sha256.
The file must score EVERY safe candidate: selecting only within the candidates a score happens to
cover would be a silent second filter.

WHAT IT NEVER DOES. It never writes outputs/ham10000/stage3/asism_selected.csv or anything else of
ham-final-v1; it refuses an output directory that is the v1 Stage 3 directory. It never reads
final_eval_heldout. Nothing here is a primary thesis result; v2 is reported as a post-hoc follow-up.

Usage (on the machine that holds the Stage 2 and Stage 3 artifacts):
    python scripts/followup/ham10000_v2_select.py --namespace ham-stratified-v1 \\
        --scores <path to a .parquet or .csv with image_id and the score column> --score-column <name>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS  # noqa: E402

FORBIDDEN_SPLIT = "final_eval_heldout"
V1_SELECTION_NAME = "asism_selected.csv"
C2_NAME = "c2_selected.csv"
D2_NAME = "d2_selected.csv"
MANIFEST_NAME = "v2_selection_manifest.json"


class V2SelectionError(RuntimeError):
    """The v2 selection cannot be made as pre-registered. Nothing is written."""


# ----------------------------------------------------------------------------------------------
# The pure rules
# ----------------------------------------------------------------------------------------------


def fill_targets(real_counts: dict[str, int], fill_to_total: int, labels: list[str]) -> dict[str, int]:
    """5b: how many synthetic images each class needs to reach N images in total."""
    if int(fill_to_total) <= 0:
        raise V2SelectionError(f"fill_to_total must be positive, got {fill_to_total}")
    return {label: max(0, int(fill_to_total) - int(real_counts.get(label, 0))) for label in labels}


def per_class_floor(frame: pd.DataFrame, percentile: float) -> tuple[pd.DataFrame, dict[str, float]]:
    """6a: keep each class's candidates at or above that class's own `percentile` score."""
    if not 0.0 <= float(percentile) <= 100.0:
        raise V2SelectionError(f"quality_floor_percentile must be in [0, 100], got {percentile}")
    floors = {
        str(label): float(np.percentile(group["score"].to_numpy(dtype=np.float64), float(percentile)))
        for label, group in frame.groupby("dx")
    }
    keep = frame["score"].to_numpy(dtype=np.float64) >= frame["dx"].map(floors).to_numpy(dtype=np.float64)
    return frame[keep].reset_index(drop=True), floors


def top_k_per_class(frame: pd.DataFrame, targets: dict[str, int]) -> pd.DataFrame:
    """Each class's highest scores up to its target. Ties break on image_id, so reruns agree."""
    ordered = frame.sort_values(["dx", "score", "image_id"], ascending=[True, False, True])
    parts = [ordered[ordered["dx"] == label].head(int(target)) for label, target in targets.items()]
    return pd.concat(parts, ignore_index=True) if parts else ordered.iloc[:0]


def random_per_class(frame: pd.DataFrame, counts: dict[str, int], seed: int) -> pd.DataFrame:
    """D2: `counts[label]` candidates per class, uniformly without replacement, from `frame`.

    Candidates are put in image_id order before drawing, so the draw depends only on the seed and
    the pool, never on the order the score file happened to list them in.
    """
    rng = np.random.default_rng(int(seed))
    parts = []
    for label, count in counts.items():
        pool = frame[frame["dx"] == label].sort_values("image_id").reset_index(drop=True)
        if int(count) > len(pool):
            raise V2SelectionError(f"D2 needs {count} {label} candidates but the safe pool has {len(pool)}")
        index = np.sort(rng.choice(len(pool), size=int(count), replace=False))
        parts.append(pool.iloc[index])
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[:0]


def attach_scores(pool: pd.DataFrame, scores: pd.DataFrame, score_column: str) -> tuple[pd.DataFrame, dict]:
    """Join the safe pool to the score file, refusing anything that would select on a partial score."""
    if "image_id" not in scores.columns or score_column not in scores.columns:
        raise V2SelectionError(f"the score file needs columns image_id and {score_column!r}")
    scores = scores.assign(image_id=scores["image_id"].astype(str))
    duplicated = scores["image_id"].duplicated()
    if duplicated.any():
        raise V2SelectionError(f"the score file lists {int(duplicated.sum())} image_id(s) more than once")
    safe = pool[["image_id", "dx"]].assign(image_id=pool["image_id"].astype(str))
    merged = safe.merge(scores[["image_id", score_column] + (["dx"] if "dx" in scores.columns else [])],
                        on="image_id", how="left", suffixes=("", "_scores"))
    if "dx_scores" in merged.columns:
        disagree = merged["dx_scores"].notna() & (merged["dx_scores"].astype(str) != merged["dx"].astype(str))
        if disagree.any():
            raise V2SelectionError(f"{int(disagree.sum())} candidate(s) have a different dx in the score file")
        merged = merged.drop(columns="dx_scores")
    missing = merged[score_column].isna()
    if missing.any():
        raise V2SelectionError(
            f"{int(missing.sum())} safe candidate(s) have no score in the score file (e.g. "
            f"{merged.loc[missing, 'image_id'].head(3).tolist()}). The score must cover the whole safe "
            "pool; selecting only within what it covers would be a silent second filter."
        )
    report = {
        "safe_candidates": int(len(safe)),
        "score_rows": int(len(scores)),
        "score_rows_not_in_safe_pool": int((~scores["image_id"].isin(set(safe["image_id"]))).sum()),
    }
    return merged.rename(columns={score_column: "score"}), report


def select_v2(scored: pd.DataFrame, real_counts: dict[str, int], cfg, labels: list[str]) -> dict:
    """The whole v2 rule on an already safe, already scored pool. Pure: nothing is read or written."""
    selection = cfg.selection
    if str(selection.quality_floor_scope) != "per_class":
        raise V2SelectionError(f"v2 is pre-registered with a per-class floor, got {selection.quality_floor_scope!r}")
    if str(selection.within_class_rule) != "top_k":
        raise V2SelectionError(f"v2 is pre-registered with top_k, got {selection.within_class_rule!r}")
    if str(cfg.d2.source_pool) != "safe":
        raise V2SelectionError(f"D2 is pre-registered to draw from the safe pool, got {cfg.d2.source_pool!r}")

    targets = fill_targets(real_counts, int(selection.fill_to_total), labels)
    above, floors = per_class_floor(scored, float(selection.quality_floor_percentile))
    c2 = top_k_per_class(above, targets)
    c2_counts = {label: int((c2["dx"] == label).sum()) for label in labels}
    d2 = random_per_class(scored, c2_counts, int(cfg.d2.seed))

    candidates = {label: int((scored["dx"] == label).sum()) for label in labels}
    survived = {label: int((above["dx"] == label).sum()) for label in labels}
    return {
        "c2": c2,
        "d2": d2,
        "targets": targets,
        "floors": floors,
        "selected_per_class": c2_counts,
        "floor_attrition_per_class": {
            label: {
                "candidates": candidates[label],
                "above_quality_floor": survived[label],
                "removed_fraction": round(1.0 - survived[label] / candidates[label], 4),
            }
            for label in labels
            if candidates[label]
        },
        # Reported, never fixed by moving N: a class short of its target is a finding about the
        # generator for that class, and Stage 4 has to be read knowing it.
        "classes_short_of_target": {
            label: {"selected": c2_counts[label], "target": targets[label]}
            for label in labels
            if c2_counts[label] < targets[label]
        },
        "c2_d2_overlap_per_class": {
            label: int(len(set(c2.loc[c2["dx"] == label, "image_id"]) & set(d2.loc[d2["dx"] == label, "image_id"])))
            for label in labels
        },
    }


# ----------------------------------------------------------------------------------------------
# The entry point
# ----------------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_scores(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise V2SelectionError(f"no score file at {path}")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _output(frame: pd.DataFrame, paths: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({
        "image_id": frame["image_id"].astype(str),
        "image_path": [paths[image_id] for image_id in frame["image_id"].astype(str)],
        "dx": frame["dx"].astype(str),
        "selection_score": frame["score"].astype(float),
    })


def run(namespace: str, scores_path: Path, score_column: str, out_dir: Path | None = None) -> dict:
    from scripts.asism.ham10000_05_adaptive_thresholds import real_class_counts
    from scripts.asism.ham10000_ranking import load_candidate_pool
    from scripts.utils.config import load_named_config
    from scripts.utils.manifest import get_git_commit_hash, write_json

    cfg = load_named_config("ham10000_v2_selection.yaml", "ham_v2_selection")
    stage2 = load_named_config("ham10000_stage2.yaml", "ham_stage2")
    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    splits_cfg = load_named_config("splits_ham10000.yaml", "ham_splits")

    split = str(cfg.real_train_split)
    if split == FORBIDDEN_SPLIT:
        raise V2SelectionError("the protected split cannot set the v2 targets")
    out_dir = Path(out_dir) if out_dir else Path(cfg.paths.outputs_dir) / namespace
    v1_dirs = {Path(stage3.paths.outputs_dir).resolve(), (Path(stage3.paths.outputs_dir) / namespace).resolve()}
    if out_dir.resolve() in v1_dirs:
        raise V2SelectionError(f"{out_dir} is a v1 Stage 3 directory; v2 never writes there")

    labels = list(CLASSIFIER_TARGET_LABELS)
    pool, surviving, pool_report = load_candidate_pool(namespace, stage3, Path(stage2.paths.stage2_root))
    scored, score_report = attach_scores(pool, _read_scores(scores_path), score_column)
    real_counts = real_class_counts(Path(splits_cfg.paths.splits_root) / namespace, split)
    result = select_v2(scored, real_counts, cfg, labels)

    manifest_frame = pd.read_csv(Path(stage2.paths.stage2_root) / namespace / "all_candidates.csv")
    paths = dict(zip(manifest_frame["image_id"].astype(str), manifest_frame["image_path"].astype(str)))
    out_dir.mkdir(parents=True, exist_ok=True)
    _output(result["c2"], paths).to_csv(out_dir / C2_NAME, index=False)
    _output(result["d2"], paths).to_csv(out_dir / D2_NAME, index=False)

    manifest = {
        "stage": "ham10000_v2_selection",
        "status": "post-hoc follow-up; ham-final-v1 remains the primary result",
        "namespace": namespace,
        "order": ["safety", "per_class_quality_floor", "top_k_to_fill_target"],
        "config": {
            "fill_to_total": int(cfg.selection.fill_to_total),
            "quality_floor_percentile": float(cfg.selection.quality_floor_percentile),
            "quality_floor_scope": str(cfg.selection.quality_floor_scope),
            "within_class_rule": str(cfg.selection.within_class_rule),
            "d2_seed": int(cfg.d2.seed),
            "d2_source_pool": str(cfg.d2.source_pool),
        },
        "score_source": {"path": str(scores_path), "column": score_column, "sha256": _sha256(scores_path)},
        "surviving_signals": surviving,
        "pool": pool_report,
        "score_coverage": score_report,
        "real_train_split": split,
        "real_class_counts": real_counts,
        "target_counts": result["targets"],
        "quality_floor_per_class": result["floors"],
        "quality_floor_attrition_per_class": result["floor_attrition_per_class"],
        "selected_per_class": result["selected_per_class"],
        "classes_short_of_target": result["classes_short_of_target"],
        "c2_d2_overlap_per_class": result["c2_d2_overlap_per_class"],
        "n_selected_c2": int(len(result["c2"])),
        "n_selected_d2": int(len(result["d2"])),
        "c2_path": str(out_dir / C2_NAME),
        "d2_path": str(out_dir / D2_NAME),
        "evidence": "scores, the safe pool and the real training split; final_eval_heldout never read",
        "git_commit_hash": get_git_commit_hash(),
    }
    write_json(out_dir / MANIFEST_NAME, manifest)

    print("Stage 3 v2 selection (5b fill-to-N, 6a per-class floor)", flush=True)
    print("=" * 72, flush=True)
    for label in labels:
        print(f"  {label:<6} real={real_counts[label]:>5}  target={result['targets'][label]:>4}  "
              f"selected={result['selected_per_class'][label]:>4}", flush=True)
    print("=" * 72, flush=True)
    if result["classes_short_of_target"]:
        print(f"  short of target: {result['classes_short_of_target']}", flush=True)
    print(f"  C2 {len(result['c2'])} -> {out_dir / C2_NAME}", flush=True)
    print(f"  D2 {len(result['d2'])} -> {out_dir / D2_NAME}", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--scores", required=True, help="score file (.parquet or .csv) with image_id")
    parser.add_argument("--score-column", required=True)
    parser.add_argument("--out-dir", default=None, help="default: outputs/ham10000/stage3_v2/<namespace>")
    args = parser.parse_args()
    manifest = run(args.namespace, Path(args.scores), args.score_column, args.out_dir)
    print(json.dumps({k: manifest[k] for k in ("n_selected_c2", "n_selected_d2", "selected_per_class")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
