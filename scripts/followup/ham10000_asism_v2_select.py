#!/usr/bin/env python3
"""ASISM v2 final selection: condition C (top-ranked within class) and D (random at C's counts).

Protocol: docs/ham10000_asism_v2_final_protocol.md, frozen before any E4 result was read.

  HOW MANY  e4_consequences.json (V1 q* split by V3), written by ham10000_e4_consequences.py. This
            module never sets or adjusts a count.
              COARSE  C takes the V3 per-class counts; D draws the same counts at random.
              NO      q* = 0: C = A and D is not built. Only the manifest is written.
              GO      refused: the count belongs to E4b, which is not designed.
  WHICH     the gate's equal-weight reference composite (configs/ham10000_asism_v2_selection.yaml,
            ranking) over ranking.scored_signals: similarity and explainability. IQA acts only
            through the safety filter (reject_invalid_iqa), never in the ranking; uncertainty has no
            a-priori direction and is not scored. Safety runs before ranking.

Reads the candidate manifest and the four-signal artifacts only. Never reads a split, never trains.

Usage:
    python -m scripts.followup.ham10000_asism_v2_select --namespace ham-stratified-v1 \
        --scores-dir <four_signal/gonogo_signal_set> --gonogo-report <four_signal/gonogo_report.json> \
        --candidates <stage2/.../all_candidates.csv> --e4-consequences <e4 dir>/e4_consequences.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.asism.ham10000_02_gonogo import PRIMARY_SCORE_COLUMN
from scripts.asism.ham10000_ranking import unsafe_candidates
from scripts.followup import ham10000_e4_consequences as e4c
from scripts.followup import ham10000_e4_quantity_curve as e4
from scripts.utils.config import load_named_config
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS, normalize_diagnosis
from scripts.utils.manifest import get_git_commit_hash, read_json, sha256_file, write_json

SELECTION_CONFIG = "ham10000_asism_v2_selection.yaml"
STAGE4_CONFIG = "ham10000_asism_v2_stage4.yaml"
MANIFEST_NAME = "asism_v2_selection_manifest.json"
C_NAME = "c_selected.csv"
D_TEMPLATE = "d_selected_seed{seed}.csv"
CANDIDATES_SHA256 = "bf8047b5dde96259c1268e5d8a3db6bac0d37392088df45182fde350ac7ef51a"


class SelectionError(SystemExit):
    pass


# Roles fixed before any E4 result (docs/ham10000_asism_v2_final_protocol.md §2). A scored set that
# differs is refused rather than run, so the ranking cannot drift through the config alone.
SCORED_SIGNALS = ("similarity", "explainability")
SAFETY_ONLY_SIGNALS = ("iqa",)


def ranking_columns(admitted: list[str], scored: list[str]) -> list[str]:
    """Headline columns of the scored signals, in gate order. Each must be admitted, have an
    a-priori direction, and none may be a safety-only signal."""
    scored = [str(s) for s in scored]
    if sorted(scored) != sorted(SCORED_SIGNALS):
        raise SelectionError(f"ranking.scored_signals {scored} is not the protocol's {list(SCORED_SIGNALS)}")
    for signal in scored:
        if signal not in admitted:
            raise SelectionError(f"{signal} is scored but was not admitted by the gate")
        if signal in SAFETY_ONLY_SIGNALS or PRIMARY_SCORE_COLUMN[signal][1] is not True:
            raise SelectionError(f"{signal} has no a-priori direction or is safety-only; it cannot be scored")
    return [PRIMARY_SCORE_COLUMN[s][0] for s in PRIMARY_SCORE_COLUMN if s in scored]


def composite(group: pd.DataFrame, columns: list[str]) -> pd.Series:
    """The gate's equal-weight composite for one class (ham10000_02_gonogo.check_usefulness)."""
    parts = []
    for column in columns:
        values = group[column].astype(float)
        spread = values.max() - values.min()
        parts.append((values - values.min()) / spread if spread > 0 else values * 0.0)
    score = sum(parts) / max(len(parts), 1)
    return score.fillna(score.min() if score.notna().any() else 0.0)


def rank_within_class(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """frame indexed by image_id with a dx column -> the same rows with composite and within-class
    rank (0 = best), ordered by class then rank. Ties go to the smaller image_id."""
    out = []
    for dx, group in frame.groupby("dx", sort=True):
        scored = group.assign(composite=composite(group, columns))
        scored = scored.assign(_id=scored.index.astype(str)).sort_values(
            ["composite", "_id"], ascending=[False, True]).drop(columns="_id")
        out.append(scored.assign(rank_in_class=np.arange(len(scored))))
    return pd.concat(out)


def select_c(ranked: pd.DataFrame, per_class: dict[str, int]) -> pd.DataFrame:
    picks = []
    for dx, count in sorted(per_class.items()):
        group = ranked[ranked["dx"] == dx]
        if count > len(group):
            raise SelectionError(f"class {dx}: count {count} exceeds its safe pool of {len(group)}")
        picks.append(group[group["rank_in_class"] < count])
    return pd.concat(picks)


def draw_d(safe: pd.DataFrame, per_class: dict[str, int], seed: int, base: int) -> pd.DataFrame:
    rng = np.random.default_rng([int(base), int(seed)])
    picks = []
    for dx, count in sorted(per_class.items()):
        ids = np.array(sorted(safe.index[safe["dx"] == dx].astype(str)))
        picks.append(safe.loc[rng.choice(ids, size=count, replace=False)])
    return pd.concat(picks)


def _write_manifest_csv(frame: pd.DataFrame, path: Path) -> str:
    out = frame.reset_index()[["image_id", "image_path", "dx"]].sort_values("image_id")
    out.to_csv(path, index=False)
    return sha256_file(path)


def load_pool(candidates: Path, scores_dir: Path, gonogo_report: Path, admitted: list[str],
              columns: list[str]) -> tuple[pd.DataFrame, dict]:
    candidates_sha = sha256_file(candidates)
    if candidates_sha != CANDIDATES_SHA256:
        raise SelectionError(f"{candidates}: sha256 {candidates_sha} is not the frozen pool {CANDIDATES_SHA256}")
    report = read_json(gonogo_report)
    if sorted(report.get("surviving_signals", [])) != sorted(admitted):
        raise SelectionError(f"the gate admitted {report.get('surviving_signals')}, the protocol fixes {admitted}")
    if report.get("candidates_csv_sha256") != CANDIDATES_SHA256:
        raise SelectionError("the Go/No-Go report was computed on another candidate pool")

    pool = pd.read_csv(candidates)
    pool["dx"] = [normalize_diagnosis(v) for v in pool["dx"]]
    pool = pool.set_index(pool["image_id"].astype(str)).drop(columns="image_id")
    artifacts, provenance = {}, {}
    for signal in admitted:
        path = scores_dir / f"{signal}_scores.parquet"
        sidecar = read_json(scores_dir / f"{signal}_scores.provenance.json")
        if sidecar.get("candidates_csv_sha256") != CANDIDATES_SHA256:
            raise SelectionError(f"{path.name} was scored on another candidate pool")
        artifacts[signal] = pd.read_parquet(path)
        provenance[signal] = sha256_file(path)
    for signal, frame in artifacts.items():
        if set(frame["image_id"].astype(str)) != set(pool.index):
            raise SelectionError(f"{signal}: the artifact does not cover exactly the candidate pool")
    for column in columns:
        signal = next(s for s, (c, _) in PRIMARY_SCORE_COLUMN.items() if c == column)
        pool[column] = artifacts[signal].set_index(artifacts[signal]["image_id"].astype(str))[column].reindex(pool.index)

    unsafe = unsafe_candidates(pool.index, artifacts["iqa"], artifacts["similarity"])
    safe = pool.drop(index=list(unsafe))
    evidence = {
        "candidates_csv": str(candidates), "candidates_csv_sha256": candidates_sha,
        "gonogo_report": str(gonogo_report), "gonogo_report_sha256": sha256_file(gonogo_report),
        "signal_artifact_sha256": provenance,
        "safety_removed": len(unsafe), "safety_reasons": dict(sorted(unsafe.items())),
        "safe_pool_per_class": {c: int(n) for c, n in safe["dx"].value_counts().sort_index().items()},
    }
    return safe, evidence


def run(namespace: str, candidates: Path, scores_dir: Path, gonogo_report: Path, consequences_path: Path,
        out_root: Path | None = None) -> dict:
    cfg = load_named_config(SELECTION_CONFIG, "ham_asism_v2_selection")
    seeds = [int(s) for s in load_named_config(STAGE4_CONFIG, "ham_stage4").seeds]
    admitted = [str(s) for s in cfg.admitted_signals]
    if str(cfg.ranking.method) != "equal_weight_within_class_minmax":
        raise SelectionError(f"unknown ranking method {cfg.ranking.method!r}")

    consequences = read_json(consequences_path)
    verdict = consequences["verdict"]
    out_dir = Path(out_root or cfg.paths.outputs_dir) / namespace
    out_dir.mkdir(parents=True, exist_ok=True)

    columns = ranking_columns(admitted, list(cfg.ranking.scored_signals))
    safe, evidence = load_pool(candidates, scores_dir, gonogo_report, admitted, columns)
    if evidence["safe_pool_per_class"] != consequences["v3_pool_counts"]:
        raise SelectionError(f"the safe pool {evidence['safe_pool_per_class']} differs from the pool E4 "
                             f"split over {consequences['v3_pool_counts']}")
    # Equal class counts do not make equal pools: the ids must be the ones E4 froze in its plan.
    safe_ids_sha = e4.ids_sha256(list(safe.index.astype(str)))
    if consequences.get("v3_pool_ids_sha256") != safe_ids_sha:
        raise SelectionError(f"the safe pool's ids (sha256 {safe_ids_sha}) are not the pool E4 measured "
                             f"({consequences.get('v3_pool_ids_sha256')})")
    evidence["safe_pool_ids_sha256"] = safe_ids_sha
    manifest = {
        "stage": "ham10000_asism_v2_select", "namespace": namespace, "protocol": "asism_v2",
        "e4_consequences": str(consequences_path), "e4_consequences_sha256": sha256_file(consequences_path),
        "e4_verdict": verdict, "q_star": consequences["v1"]["q_star"],
        "ranking": {"method": str(cfg.ranking.method), "columns": columns,
                    "scored_signals": [s for s in PRIMARY_SCORE_COLUMN if s in cfg.ranking.scored_signals],
                    "safety_only": [s for s in admitted if s in SAFETY_ONLY_SIGNALS],
                    "not_scored_no_direction": [s for s in admitted if PRIMARY_SCORE_COLUMN[s][1] is None],
                    "tie_break": str(cfg.ranking.tie_break)},
        **evidence, "stage4_seeds": seeds, "git_commit_hash": get_git_commit_hash(),
        "final_eval_heldout_read": False, "classifier_val_read": False,
    }

    if verdict == e4.VERDICT_GO:
        raise SelectionError("E4 = GO: the count is E4b's (V2), which is not designed. Nothing is selected.")
    if verdict == e4.VERDICT_NO:
        manifest.update(per_class={c: 0 for c in sorted(safe["dx"].unique())}, n_selected_c=0, n_selected_d=0,
                        conditions="C = A (q* = 0); D not built")
        write_json(out_dir / MANIFEST_NAME, manifest)
        return manifest

    per_class = {str(c): int(n) for c, n in consequences["count"]["per_class"].items()}
    if per_class != e4c.allocate(int(consequences["v1"]["q_star"]), evidence["safe_pool_per_class"]):
        raise SelectionError("the per-class counts in e4_consequences.json are not V3 of q*")
    ranked = rank_within_class(safe, columns)
    ranked.reset_index().to_csv(out_dir / "c_ranking.csv", index=False)
    chosen = select_c(ranked, per_class)
    files = {C_NAME: _write_manifest_csv(chosen, out_dir / C_NAME)}
    d_overlap = {}
    for seed in seeds:
        drawn = draw_d(safe, per_class, seed, int(cfg.d.draw_seed_base))
        name = D_TEMPLATE.format(seed=seed)
        files[name] = _write_manifest_csv(drawn, out_dir / name)
        d_overlap[str(seed)] = int(len(set(drawn.index) & set(chosen.index)))
    manifest.update(
        per_class=per_class, n_selected_c=int(len(chosen)), n_selected_d=int(sum(per_class.values())),
        d_draw_seed_base=int(cfg.d.draw_seed_base), d_overlap_with_c=d_overlap,
        ranking_sha256=sha256_file(out_dir / "c_ranking.csv"), files_sha256=files,
        c_ids_sha256=hashlib.sha256("\n".join(sorted(chosen.index)).encode()).hexdigest(),
    )
    write_json(out_dir / MANIFEST_NAME, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--scores-dir", required=True, type=Path)
    parser.add_argument("--gonogo-report", required=True, type=Path)
    parser.add_argument("--e4-consequences", required=True, type=Path)
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args()
    m = run(args.namespace, args.candidates, args.scores_dir, args.gonogo_report, args.e4_consequences, args.out_root)
    print(json.dumps({k: m.get(k) for k in ("e4_verdict", "q_star", "per_class", "n_selected_c")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
