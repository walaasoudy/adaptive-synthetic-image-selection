#!/usr/bin/env python3
"""CPU dry run of the whole ASISM v2 ranker path, with the classifier replaced by a known formula.

NOT A MEASUREMENT. No model is trained, no image and no split is read, and no GPU is used. It exists
to show, before any GPU run is considered, that every step runs and writes what it should:

    plan -> measure (1,000 cells, 5 seeds) -> G1 -> acceptance -> fit -> select (C, D) -> stability

What is real: the frozen configuration, the candidate signal table (3,168 images, four signals), the
200 designed subsets, and all of the code after the classifier.
What is fake: each cell's "macro AUROC" is  0.86 + 0.012 * log(1 + sum of planted image weights)
plus seed noise, where an image's planted weight is 1 + (planted class weights . its four signals).
Because the truth is planted, the report can also say how many of the images with a positive planted
weight the selection kept.

The fit uses fewer bootstrap models than the pre-registered 200 (--bootstrap, default 20) so that the
dry run takes about an hour instead of about ten. Everything is written under --out-dir, which must
be new and outside the project's outputs. Every artifact is stamped scientific_evidence: false.

Usage:
    python -m scripts.smoke.asism_v2_ranker_dry_run --candidates <all_candidates.csv> \
        --scores-dir <four_signal/gonogo_signal_set> --out-dir <a new directory>
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np

from scripts.asism_v2.features import SIGNALS
from scripts.asism_v2.prereg import load_prereg
from scripts.asism_v2.supervision import load_signal_table
from scripts.followup import ham10000_asism_v2_learned_select as learned
from scripts.followup import ham10000_asism_v2_utility as utility

NAMESPACE = "ham-stratified-v1"
PLANT_SEED, NOISE_SD, BASE, LAM = 20261003, 0.003, 0.86, 0.012


def planted_weights(frame) -> dict[str, float]:
    """image_id -> planted weight. Class weight vectors differ in size, so classes differ in how
    many of their images are worth adding."""
    rng = np.random.default_rng(PLANT_SEED)
    weights = {}
    for dx, group in frame.groupby("dx", sort=True):
        z = group[list(SIGNALS)].to_numpy(float)
        z = (z - z.mean(0)) / np.where(z.std(0) > 0, z.std(0), 1.0)
        w = 1 + z @ (rng.normal(0, 1, len(SIGNALS)) * rng.choice([0.15, 0.5, 0.9]))
        weights.update(dict(zip(group["image_id"].astype(str), w)))
    return weights


def run(candidates: Path, scores_dir: Path, out_dir: Path, bootstrap: int) -> dict:
    prereg = load_prereg()
    out_dir = Path(out_dir)
    if out_dir.exists() or Path(prereg["paths"]["outputs_dir"]).resolve() in out_dir.resolve().parents:
        raise SystemExit("--out-dir must be a new directory outside the project's outputs")
    started = time.perf_counter()
    plan_summary = utility.run_plan(NAMESPACE, candidates, scores_dir, out_dir)
    frame, _ = load_signal_table(candidates, scores_dir, prereg["safety"])
    weight = planted_weights(frame)

    def fake_classifier(train, tuning, proxy, seed, device):
        ids = [record["image_id"] for record in train]
        rng = np.random.default_rng([int(seed), zlib.crc32("".join(sorted(ids)).encode())])
        total = max(0.0, sum(weight[i] for i in ids))
        return {"macro_auroc_ovr": float(BASE + LAM * np.log1p(total) + rng.normal(0, NOISE_SD))}

    inputs = {"real": [], "tuning": [], "candidates": {i: {"image_id": i} for i in weight},
              "hashes": {"outcome_split": "dry-run", "real_train_split": "dry-run", "measurement_code": "dry-run"}}
    # The approval switch is set for this process only, and only ever with the fake classifier above.
    utility.MEASUREMENT_APPROVED = True
    try:
        utility.run_measure(NAMESPACE, "cpu", True, out_dir=out_dir, measure_fn=fake_classifier, inputs=inputs)
    finally:
        utility.MEASUREMENT_APPROVED = False
    approved_fit = learned.fit_arguments
    learned.fit_arguments = lambda prereg, seed=None: {**approved_fit(prereg, seed), "bootstrap": int(bootstrap)}
    try:
        gate = utility.run_gate(NAMESPACE, out_dir)
        args = (NAMESPACE, candidates, scores_dir, out_dir)
        accept = learned.run_accept(*args)
        fit = learned.run_fit(*args)
        selection = learned.run_select(*args)
        stability = learned.run_stability(*args)
    finally:
        learned.fit_arguments = approved_fit

    chosen = set()
    if selection["selection_outcome"] == "subset":
        import pandas as pd
        chosen = set(pd.read_csv(out_dir / "c_selected.csv")["image_id"].astype(str))
    positive = {i for i, w in weight.items() if w > 0}
    report = {
        "scientific_evidence": False,
        "what_this_is": "CPU dry run; the classifier is a planted formula, not a trained model",
        "bootstrap_models_used": int(bootstrap), "bootstrap_models_pre_registered": prereg["fit"]["bootstrap_models"],
        "minutes": round((time.perf_counter() - started) / 60, 1),
        "plan": plan_summary,
        "measure": {"fit_rows": sum(1 for _ in open(out_dir / utility.FIT_RUNS)),
                    "test_rows": sum(1 for _ in open(out_dir / utility.TEST_RUNS)),
                    "training_seeds": prereg["supervision"]["training_seeds"]},
        "g1": {k: gate[k] for k in ("passed", "threshold", "reliability_of_mean", "icc_single_run", "subsets", "repeats")},
        "g1_within_size_not_gating": {size: round(block["reliability_of_mean"], 3)
                                      for size, block in gate["within_size_not_gating"].items()},
        "acceptance": {"accepted": accept["accepted"], "a_correlation_passed": accept["a_correlation_passed"],
                       "b_beats_size_and_class_only": accept["b_beats_size_and_class_only"],
                       "models": accept["models"]},
        "fit": {k: fit[k] for k in ("best_epoch", "validation_mse", "frozen_unidentifiable", "bootstrap_models")},
        "selection": {k: selection[k] for k in ("selection_outcome", "protocol", "per_class", "n_selected_c",
                                                "n_selected_d", "score_sources")},
        "selection_files": sorted(selection["files_sha256"]),
        "planted_truth": {"images_with_positive_weight": len(positive), "selected": len(chosen),
                          "selected_with_positive_weight": len(chosen & positive),
                          "selected_with_non_positive_weight": len(chosen - positive)},
        "stability": stability,
    }
    (out_dir / "DRY_RUN_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--scores-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=20)
    args = parser.parse_args()
    report = run(args.candidates, args.scores_dir, args.out_dir, args.bootstrap)
    print(json.dumps({k: v for k, v in report.items() if k != "plan"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
