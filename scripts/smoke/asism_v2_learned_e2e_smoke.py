#!/usr/bin/env python3
"""CPU smoke of the ASISM v2 learned path from the plan to the Stage 5 comparison, on the default paths.

NOT A MEASUREMENT and not evidence. It answers one question before any GPU run: are the steps wired
to each other? Every step reads what the step before it wrote, at the path the committed configs
give, under a fixture PROJECT_ROOT:

    plan -> measure -> G1 -> accept -> fit -> select (C, D) -> stability
         -> Stage 4 (A, B, C, D x 2 seeds, real entry point) -> aggregate -> compare (classifier_val)

What is real: the code and the committed configs of every step, the candidate ids and their four
signals, the measure phase's own input builder (real split and candidate records), the Stage 4,
aggregate and compare entry points run as the commands the pod runs.
What is fake: the images (random 64 px JPEGs; the real splits are replaced by 3 images per class;
the candidate manifest is the frozen one with only its image_path column pointed at fixture images,
so its pinned hash is replaced for this process after the source file is checked against the pin),
the utility of a cell (the planted formula of asism_v2_ranker_dry_run), and the Stage 4 budget (one
step at 64 px from random weights, set through THESIS_CONFIG_OVERLAY; two seeds instead of twenty).
The real proxy trainer is called once, outside the measure phase, on a one-step budget, to show
that the records the measure phase builds are what the trainer accepts. Nothing it returns is kept.

Usage (CPU, about 20 minutes):
    python -m scripts.smoke.asism_v2_learned_e2e_smoke --candidates <all_candidates.csv> \
        --scores-dir <four_signal/gonogo_signal_set> --work-dir <a new directory outside the repo>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
NAMESPACE = "ham-stratified-v1"
PROTOCOL = "asism_v2_learned"
SMOKE_SEEDS = [42, 43]
REAL_SPLITS = ("classifier_train", "classifier_val", "asism_tuning_heldout")
IMAGES_PER_CLASS = 3
PLANTED = {"base": 0.5, "lam": 0.06, "noise_sd": 0.001}      # the dry run's strong scenario
OVERLAY = {"ham_stage4": {"model": {"pretrained_source": "random", "resolution": 64},
                          "training": {"max_steps": 1, "batch_size": 4, "eval_every_n_steps": 1},
                          "seeds": SMOKE_SEEDS}}


def _jpeg(path: Path, rng) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(path)


def build_workspace(work: Path, candidates: Path) -> Path:
    """A PROJECT_ROOT holding what the configs expect: three small real splits and the candidate
    manifest at Stage 2's path, with every image_path pointing at a fixture image of its class."""
    from omegaconf import OmegaConf

    from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS

    rng = np.random.default_rng(0)
    processed = work / "data" / "ham10000" / "processed"
    for split in REAL_SPLITS:
        rows = []
        for label in CLASSIFIER_TARGET_LABELS:
            for index in range(IMAGES_PER_CLASS):
                image_id = f"ISIC_{split}_{label}_{index}"
                _jpeg(processed / "images" / NAMESPACE / split / f"{image_id}.jpg", rng)
                rows.append({"image_id": image_id, "lesion_id": f"L_{image_id}", "dx": label})
        (processed / "splits" / NAMESPACE).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(processed / "splits" / NAMESPACE / f"{split}.csv", index=False)

    manifest = pd.read_csv(candidates)
    for dx in sorted(manifest["dx"].astype(str).unique()):
        _jpeg(work / "synthetic" / f"{dx}.jpg", rng)
    manifest["image_path"] = [str(work / "synthetic" / f"{dx}.jpg") for dx in manifest["dx"].astype(str)]
    target = work / "outputs" / "ham10000" / "stage2" / NAMESPACE / "all_candidates.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(target, index=False)
    OmegaConf.save(OmegaConf.create(OVERLAY), work / "smoke_overlay.yaml")
    return target


def _command(args: list[str], env: dict, log: list) -> None:
    printable = "python " + " ".join(args)
    print(f"\n$ {printable}", flush=True)
    started = time.perf_counter()
    done = subprocess.run([sys.executable, *args], cwd=REPO, env=env, capture_output=True, text=True)
    log.append({"command": printable, "exit_code": done.returncode, "seconds": round(time.perf_counter() - started, 1)})
    if done.returncode != 0:
        print(done.stdout[-3000:], done.stderr[-3000:], sep="\n", flush=True)
        raise SystemExit(f"SMOKE FAILED at: {printable}")
    print(done.stdout.strip()[-600:], flush=True)


def run(candidates: Path, scores_dir: Path, work: Path, bootstrap: int) -> dict:
    work = Path(work).resolve()
    if work.exists() or REPO in work.parents:
        raise SystemExit("--work-dir must be a new directory outside the repository")
    work.mkdir(parents=True)
    started = time.perf_counter()
    os.environ["PROJECT_ROOT"] = str(work)
    os.environ["THESIS_CONFIG_OVERLAY"] = str(work / "smoke_overlay.yaml")
    pool_csv = build_workspace(work, Path(candidates))

    from scripts.asism_v2 import supervision

    if supervision.sha256(Path(candidates)) != supervision.CANDIDATES_SHA256:
        raise SystemExit(f"{candidates} is not the frozen candidate pool")
    source, fixture = pd.read_csv(candidates), pd.read_csv(pool_csv)
    assert source.drop(columns="image_path").equals(fixture.drop(columns="image_path"))
    pinned = supervision.load_signal_table.__defaults__
    supervision.load_signal_table.__defaults__ = (supervision.sha256(pool_csv),)
    try:
        return _steps(work, pool_csv, scores_dir, bootstrap, started)
    finally:
        supervision.load_signal_table.__defaults__ = pinned


def _steps(work: Path, pool_csv: Path, scores_dir: Path, bootstrap: int, started: float) -> dict:
    from scripts.asism_v2.prereg import load_prereg
    from scripts.asism_v2.supervision import load_signal_table
    from scripts.followup import ham10000_asism_v2_learned_select as learned
    from scripts.followup import ham10000_asism_v2_utility as utility
    from scripts.smoke.asism_v2_ranker_dry_run import planted_weights
    from scripts.utils.config import load_named_config
    from scripts.utils.ham10000_conditions import get_protocol

    prereg = load_prereg()
    out_dir = utility.default_out_dir(prereg, NAMESPACE)
    if work not in Path(out_dir).resolve().parents:
        raise SystemExit(f"the ranker's output directory {out_dir} is not under the fixture PROJECT_ROOT")
    checks = {}

    # ---- plan, and the measure phase's own inputs -------------------------------------------------
    utility.run_plan(NAMESPACE, pool_csv, scores_dir)
    plan = utility.load_plan(out_dir, prereg)
    inputs = utility._measure_inputs(NAMESPACE, plan, prereg, pool_csv)
    checks["measure_inputs"] = {"real_train_records": len(inputs["real"]), "outcome_records": len(inputs["tuning"]),
                                "candidate_records": len(inputs["candidates"])}
    assert len(inputs["real"]) == len(inputs["tuning"]) == 7 * IMAGES_PER_CLASS
    assert set(inputs["candidates"]) == set(plan["roles"])

    # The real proxy trainer, once, on a one-step budget: the records above are what it accepts.
    first = plan["subsets"][sorted(plan["subsets"])[0]]
    tiny = SimpleNamespace(**{**utility.proxy_recipe(prereg), "max_steps": 1, "batch_size": 4,
                              "resolution": 64, "pretrained_source": "random"})
    probe = utility._default_measure(inputs["real"] + [inputs["candidates"][i] for i in first["image_ids"][:8]],
                                     inputs["tuning"], tiny, 42, "cpu")
    metric = prereg["supervision"]["metric"]
    assert np.isfinite(probe[metric]), probe
    checks["real_proxy_trainer_one_step"] = {"metric": metric, "returned_a_finite_value": True,
                                             "value_is_meaningless_and_not_kept": True}

    # ---- measure with the planted formula, on the real inputs ------------------------------------
    frame, _ = load_signal_table(pool_csv, scores_dir, prereg["safety"])
    weight = planted_weights(frame)

    def planted(train, tuning, proxy, seed, device):
        ids = [record["image_id"] for record in train if record["image_id"] in weight]
        rng = np.random.default_rng([int(seed), zlib.crc32("".join(sorted(ids)).encode())])
        total = max(0.0, sum(weight[i] for i in ids))
        return {metric: float(PLANTED["base"] + PLANTED["lam"] * np.log1p(total) + rng.normal(0, PLANTED["noise_sd"]))}

    # The approval switch is set for this process only, and only ever with the planted formula above.
    utility.MEASUREMENT_APPROVED = True
    try:
        utility.run_measure(NAMESPACE, "cpu", True, candidates=pool_csv, measure_fn=planted)
    finally:
        utility.MEASUREMENT_APPROVED = False
    try:                                                    # and it is refused again afterwards
        utility.run_measure(NAMESPACE, "cpu", True, candidates=pool_csv, measure_fn=planted)
        raise SystemExit("BUG: the measure phase ran without approval")
    except utility.UtilityError:
        checks["measure_refused_without_approval"] = True

    # ---- G1, acceptance, fit, select, stability: default paths, nothing passed between them ------
    approved_fit = learned.fit_arguments
    learned.fit_arguments = lambda prereg, seed=None: {**approved_fit(prereg, seed), "bootstrap": int(bootstrap)}
    try:
        gate = utility.run_gate(NAMESPACE)
        args = (NAMESPACE, pool_csv, scores_dir)
        accept = learned.run_accept(*args)
        if not accept["accepted"]:
            raise SystemExit("SMOKE FAILED: the planted truth was not accepted; nothing downstream can be checked")
        learned.run_fit(*args)
        selection = learned.run_select(*args)
        stability = learned.run_stability(*args)
    finally:
        learned.fit_arguments = approved_fit
    if selection["selection_outcome"] != "subset":
        raise SystemExit(f"SMOKE FAILED: outcome {selection['selection_outcome']!r}; this smoke needs a subset")
    checks["g1_passed"] = bool(gate["passed"])
    checks["selection"] = {"outcome": selection["selection_outcome"], "n_selected_c": selection["n_selected_c"],
                           "per_class": selection["per_class"], "stability_totals": [stability["total_min"], stability["total_max"]]}

    # ---- the files Stage 4 reads are the files the selector wrote --------------------------------
    stage4 = load_named_config(get_protocol(PROTOCOL).stage4_config, "ham_stage4")
    assert [int(s) for s in stage4.seeds] == SMOKE_SEEDS
    c_path = Path(str(stage4.conditions.C.synthetic_manifest))
    assert c_path == Path(out_dir) / "c_selected.csv" and c_path.is_file(), c_path
    c_counts = pd.read_csv(c_path)["dx"].astype(str).value_counts().to_dict()
    for seed in SMOKE_SEEDS:
        d_path = Path(str(stage4.conditions.D.synthetic_manifest).replace("{seed}", str(seed)))
        assert d_path.is_file(), d_path
        assert pd.read_csv(d_path)["dx"].astype(str).value_counts().to_dict() == c_counts, seed
    assert Path(str(stage4.conditions.B.synthetic_manifest)) == pool_csv
    checks["stage4_reads_the_selectors_files"] = True
    checks["d_matches_c_per_class_for_every_seed"] = True

    # ---- Stage 4, aggregate, compare: the commands the pod runs ----------------------------------
    env = dict(os.environ)
    commands = []
    for condition in get_protocol(PROTOCOL).conditions:
        for seed in SMOKE_SEEDS:
            _command(["scripts/classify/ham10000_train_conditions.py", "--protocol", PROTOCOL,
                      "--condition", condition, "--seed", str(seed), "--device", "cpu"], env, commands)
    _command(["scripts/classify/ham10000_aggregate_conditions.py", "--protocol", PROTOCOL], env, commands)
    _command(["-m", "scripts.followup.ham10000_asism_v2_compare", "--protocol", PROTOCOL,
              "--source", "classifier_val"], env, commands)

    results = Path(str(stage4.paths.results_dir)) / NAMESPACE
    trained = {}
    for condition in get_protocol(PROTOCOL).conditions:
        for seed in SMOKE_SEEDS:
            manifest = json.loads((results / condition / f"seed{seed}" / "run_manifest.json").read_text(encoding="utf-8"))
            assert manifest["final_eval_heldout_read"] is False and manifest["protocol"] == PROTOCOL
            trained[f"{condition}/seed{seed}"] = manifest["data"]["synthetic_images"]
    n_c = int(selection["n_selected_c"])
    assert all(trained[f"A/seed{s}"] == 0 and trained[f"B/seed{s}"] == len(frame)
               and trained[f"C/seed{s}"] == n_c and trained[f"D/seed{s}"] == n_c for s in SMOKE_SEEDS), trained
    checks["synthetic_images_per_run"] = trained
    comparison = json.loads((results / "_classifier_val_comparison" / "asism_v2_comparison_classifier_val.json")
                            .read_text(encoding="utf-8"))
    assert comparison["confirmatory_family"] == ["C_vs_D:balanced_accuracy"]
    assert set(comparison["added_metrics"]) <= set(comparison["per_condition"]["C"])
    assert comparison["split"] == "classifier_val"
    checks["comparison"] = {"confirmatory_family": comparison["confirmatory_family"],
                            "added_metrics": comparison["added_metrics"], "label": comparison["label"]}

    report = {
        "scientific_evidence": False,
        "what_this_is": "CPU wiring smoke: fixture images, a planted utility formula, a one-step Stage 4 budget",
        "status": "PASS", "minutes": round((time.perf_counter() - started) / 60, 1),
        "project_root": str(work), "planted": PLANTED, "bootstrap_models_used": int(bootstrap),
        "stage4_overlay": OVERLAY["ham_stage4"], "checks": checks, "commands": commands,
        "not_exercised": ["the real proxy recipe (224 px, 300 steps, ImageNet weights) on a GPU",
                          "the real Stage 4 recipe (512 px, 3,000 steps, 20 seeds)",
                          "Stage 5 on final_eval_heldout, which no step here can read"],
    }
    (work / "SMOKE_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--scores-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=20)
    args = parser.parse_args()
    report = run(args.candidates, args.scores_dir, args.work_dir, args.bootstrap)
    print(json.dumps({k: report[k] for k in ("status", "minutes", "checks")}, indent=2))
    print(f"\nSMOKE {report['status']} -> {Path(report['project_root']) / 'SMOKE_REPORT.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
