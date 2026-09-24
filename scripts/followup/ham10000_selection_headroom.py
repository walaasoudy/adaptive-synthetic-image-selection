"""Is there enough variation between 616-image selections for selection to mean anything?

Two questions, one grid, one cheap proxy recipe (224 px / 300 steps, the v1 architecture).

  Positive control    A (real only) against B (real + every candidate). V1 measured this effect at
                      Stage 4 and it is large. If the cheap proxy cannot see it, the proxy is blind
                      and nothing measured with it means anything.

  Selection headroom  fifteen random 616-image draws. The spread between them is the ceiling on
                      what ANY selection rule can achieve. C, the selection v1 actually made, is
                      measured in the same grid so its position in that spread is visible.

Everything produced here is DIAGNOSTIC evidence about the measuring instrument and about how much
room a selection rule could have. None of it is a Stage 5 result, none of it is confirmatory, and
nothing in ASISM, the ranking network or the selection is changed on the strength of it. The
headroom figure of 0.0165 is a pre-declared EXPLORATORY reference, not a hypothesis threshold and
not a claim about the smallest utility variation that matters scientifically: it is half of a gap
observed in v1 after the fact, and a number seen before it was chosen cannot carry a confirmatory
verdict. C is measured to be located, never to tune a threshold or to re-pick a selection.

The budget is OPTIMIZER STEPS, never epochs. That is the frozen fairness protocol of
docs/stages2_to_5_plan.md:454, and the reason is that at fixed epochs a larger dataset silently
receives more gradient updates, which conflates "more data" with "more training" - the exact
confound condition B exists to rule out. A and B therefore see different numbers of epochs, and the
ratio between them is identical here (2.93) to the ratio at the Stage 4 recipe that produced v1's
headline. The asymmetry is not removed; it is recorded per run and reported.

Phases:
  --phase plan      laptop, seconds. Freezes the arms, the draws and both decision rules.
  --phase measure   GPU, about an hour. 268 runs. Resumable.
  --phase analyze   laptop, seconds. Refuses an incomplete grid, then applies the frozen rules.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.followup.ham10000_proxy_noise_floor import (
    FORBIDDEN_SPLIT, TUNING_SPLIT, V1_ARCHITECTURE, V1_PROXY,
    _append_jsonl, _gpu_name, _read_jsonl, _sha256, check_proxy_config,
    freeze_measure_inputs, variance_components, variant_proxy,
)

PROXY_VARIANT = "v1"
LABEL_METRIC = "balanced_accuracy"
METRICS = ("balanced_accuracy", "macro_auroc_ovr", "macro_f1", "accuracy")

# ---- the positive control ------------------------------------------------------------------------
ARM_REAL_ONLY = "A_real_only"
ARM_ALL_SYNTHETIC = "B_all_synthetic"
CONTROL_SEEDS = tuple(range(42, 42 + 54))          # 42..95, 54 per arm, 108 runs
# Sized for an effect attenuated four-fold from v1's Stage 4 value, because the cheap proxy was
# measured to compress between-subset spread about three-fold on macro AUROC. The anchor is a
# reference, not a prediction: the measured effect is whatever it is.
CONTROL_ANCHOR_STAGE4 = 0.1127                     # B - A, balanced accuracy, 3 seeds, classifier_val
CONTROL_ALPHA = 0.05

# ---- selection headroom --------------------------------------------------------------------------
SELECTION_SIZE = 616                               # the size v1's selection actually used
N_RANDOM_SUBSETS = 15
HEADROOM_SEEDS = tuple(range(42, 52))              # 42..51, 10 per subset
DRAW_SEED = 20260924                               # frozen before any run; draws are never re-drawn
ARM_SELECTED = "C_asism_selected"

# An EXPLORATORY reference, declared before the experiment so it cannot be moved afterwards, and
# explicitly not a confirmatory threshold.
#   Reasoning: for ASISM to beat random by B - C, that gap has to lie within about two standard
#   deviations of the random-616 distribution, i.e. selection would have to land in its top 2.5%.
#   Reading it the other way, a sigma_s far below this leaves a selection rule very little to find.
#   Provenance: B - C = 0.033 is a gap OBSERVED in v1 on a test set, and it was known before this
#   reference was written. A number seen in advance can orient an exploratory diagnostic; it cannot
#   support a confirmatory claim, and it is not asserted to be the smallest utility variation that
#   would matter. Whether the reference is met is reported as an orientation, never as a verdict.
ANCHOR_B_MINUS_C = 0.033                           # the smaller of the two available v1 gaps
SIGMA_S_REFERENCE = ANCHOR_B_MINUS_C / 2           # 0.0165
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0


class HeadroomError(RuntimeError):
    """Raised instead of continuing on a grid or an input that cannot carry the conclusion."""


# ===================================================================================================
# Statistics
# ===================================================================================================


def welch(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Welch t-test on b - a, with the 95% interval reported whatever the verdict.

    The interval is not decoration. A failure with a tight interval around zero is evidence of
    blindness; a failure with a wide one is an underpowered experiment, and the two must not look
    the same in the report.
    """
    from scipy import stats

    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    diff = b.mean() - a.mean()
    se = float(np.sqrt(va / na + vb / nb))
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    t = diff / se
    p = 2 * stats.t.sf(abs(t), df)
    half = stats.t.ppf(0.975, df) * se
    return {
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "difference": float(diff), "se": se, "t": float(t), "df": float(df),
        "p_two_sided": float(p),
        "ci95": [float(diff - half), float(diff + half)],
        "n_a": int(na), "n_b": int(nb),
        "passed": bool(p <= CONTROL_ALPHA),
    }


def sigma_s(table: np.ndarray) -> dict:
    """Between-subset SD of the random-616 draws, with a bootstrap interval over subsets."""
    components = variance_components(table)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = table.shape[0]
    draws = np.array([variance_components(table[rng.integers(0, n, size=n)])["between_sd"]
                      for _ in range(N_BOOTSTRAP)])
    return {
        "sigma_s": components["between_sd"],
        "sigma_e": components["within_sd"],
        "icc_single_run": components["icc_single_run"],
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "share_of_resamples_above_reference": float((draws >= SIGMA_S_REFERENCE).mean()),
        "exploratory_reference": SIGMA_S_REFERENCE,
        "anchor_b_minus_c": ANCHOR_B_MINUS_C,
        "status": "exploratory diagnostic reference, not a confirmatory threshold",
        "provenance": "half of a v1 C-vs-B gap observed on a test set and known before this "
                      "reference was written; it orients the reading and decides nothing",
        # Deliberately not called "passed". Meeting it is an orientation, not a verdict, and
        # missing it does not by itself rule selection out.
        "reference_met": bool(components["between_sd"] >= SIGMA_S_REFERENCE),
    }


def _complete_table(rows: list[dict], ids: list[str], seeds: list[int], metric: str) -> np.ndarray:
    """Refuse a partial or duplicated grid rather than analysing the complete part of it."""
    cells: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        cells.setdefault((row["subset_id"], int(row["seed"])), []).append(float(row["metrics"][metric]))
    want = [(i, s) for i in ids for s in seeds]
    missing = [k for k in want if k not in cells]
    duplicated = [k for k, v in cells.items() if len(v) > 1]
    if missing:
        raise HeadroomError(
            f"the grid is incomplete: {len(missing)} of {len(want)} cells missing "
            f"(e.g. {missing[:3]}). Finish --phase measure; nothing partial is analysed.")
    if duplicated:
        raise HeadroomError(f"duplicated cells: {duplicated[:5]}. The runs file holds a cell twice.")
    return np.array([[cells[(i, s)][0] for s in seeds] for i in ids])


# ===================================================================================================
# Phases
# ===================================================================================================


def _random_draws(pool: list[str]) -> dict[str, list[str]]:
    """Fifteen independent draws of 616 from the candidate pool, frozen by DRAW_SEED.

    They overlap, because 616 x 15 far exceeds the pool. That is not a flaw: the quantity being
    estimated is the spread of the distribution of random 616-selections, and independent draws from
    that distribution naturally share images.
    """
    rng = np.random.default_rng(DRAW_SEED)
    ordered = sorted(pool)
    return {f"random_616_{i:02d}":
            sorted(rng.choice(ordered, size=SELECTION_SIZE, replace=False).tolist())
            for i in range(N_RANDOM_SUBSETS)}


def run_plan(namespace: str, selected_csv: Path, out_dir: Path) -> dict:
    import pandas as pd

    from scripts.asism.ham10000_ranking import load_candidate_pool
    from scripts.utils.config import load_named_config
    from scripts.utils.manifest import get_git_commit_hash

    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "headroom_plan.json"
    if plan_path.is_file():
        raise HeadroomError(f"a frozen plan already exists at {plan_path}. It is never re-drawn: "
                            "delete the whole output directory to start this experiment again.")
    if not selected_csv.is_file():
        raise HeadroomError(f"no ASISM selection at {selected_csv}; C cannot be placed without it")

    stage2 = load_named_config("ham10000_stage2.yaml", "ham_stage2")
    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    frame, _, _ = load_candidate_pool(namespace, stage3, Path(stage2.paths.stage2_root), verbose=False)
    pool = sorted(frame["image_id"].astype(str).tolist())

    selected = pd.read_csv(selected_csv)["image_id"].astype(str).tolist()
    if len(selected) != SELECTION_SIZE:
        raise HeadroomError(f"the ASISM selection holds {len(selected)} images, "
                            f"expected {SELECTION_SIZE}")
    missing = sorted(set(selected) - set(pool))
    if missing:
        raise HeadroomError(f"{len(missing)} selected images are not in the candidate pool "
                            f"(e.g. {missing[:3]}); the two artifacts disagree")

    arms: dict[str, list[str]] = {ARM_REAL_ONLY: [], ARM_ALL_SYNTHETIC: pool,
                                 ARM_SELECTED: sorted(selected)}
    arms.update(_random_draws(pool))

    plan = {
        "experiment": "selection_headroom",
        "namespace": namespace,
        "proxy_variant": PROXY_VARIANT,
        "proxy": {**variant_proxy(PROXY_VARIANT), "architecture": V1_ARCHITECTURE},
        "budget_protocol": "optimizer steps, not epochs (docs/stages2_to_5_plan.md:454)",
        "label_metric": LABEL_METRIC,
        "candidate_pool_size": len(pool),
        "positive_control": {
            "arms": [ARM_REAL_ONLY, ARM_ALL_SYNTHETIC],
            "seeds": list(CONTROL_SEEDS),
            "test": "two-sided Welch t on balanced accuracy",
            "alpha": CONTROL_ALPHA,
            "anchor_stage4": CONTROL_ANCHOR_STAGE4,
            "anchor_note": "reference only; the measured effect is whatever it is, and the 95% "
                           "interval is reported whether the test passes or fails",
        },
        "headroom": {
            "subsets": sorted(k for k in arms if k.startswith("random_616_")),
            "selected_arm": ARM_SELECTED,
            "seeds": list(HEADROOM_SEEDS),
            "selection_size": SELECTION_SIZE,
            "draw_seed": DRAW_SEED,
            "exploratory_reference": {
                "sigma_s": SIGMA_S_REFERENCE,
                "anchor_b_minus_c": ANCHOR_B_MINUS_C,
                "reasoning": "for selection to beat random by B - C, that gap must lie within "
                             "about 2 SD of the random-616 distribution",
                "status": "exploratory diagnostic reference, NOT a confirmatory threshold and NOT "
                          "a claim about the smallest utility variation that matters",
                "provenance": "half of a v1 C-vs-B gap observed on a test set, known before this "
                              "reference was written",
            },
        },
        "what_this_cannot_do": [
            "it is not a Stage 5 result and is not reported as one",
            "no C2 is produced and no ASISM, ranking or selection change follows from it alone",
            "C is located within the random distribution, never used to set or move a reference",
            "a sigma_s below the reference does not by itself close the question",
        ],
        "attempts": "one. If the control fails, the proxy is blind and no further measurement with "
                    "it is interpreted. sigma_s is not re-estimated on a different draw, a "
                    "different size or a different metric after the number is seen.",
        "arms": arms,
        "asism_selected_sha256": _sha256(selected_csv),
        "git_commit_hash": get_git_commit_hash(),
    }
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    runs = 2 * len(CONTROL_SEEDS) + (N_RANDOM_SUBSETS + 1) * len(HEADROOM_SEEDS)
    print(f"Frozen: {runs} runs -> {plan_path}", flush=True)
    print(f"  positive control {2 * len(CONTROL_SEEDS)}   headroom "
          f"{N_RANDOM_SUBSETS * len(HEADROOM_SEEDS)}   C {len(HEADROOM_SEEDS)}", flush=True)
    return {"plan_path": str(plan_path), "runs_planned": runs, "candidate_pool_size": len(pool)}


def run_measure(namespace: str, out_dir: Path, device: str | None) -> dict:
    import pandas as pd

    from scripts.asism.ham10000_03_build_utility_subsets import (
        _candidate_records, _measure, _split_records,
    )
    from scripts.asism.ham10000_ranking import load_candidate_pool
    from scripts.utils.config import load_named_config
    from scripts.utils.manifest import get_git_commit_hash

    plan_path = out_dir / "headroom_plan.json"
    if not plan_path.is_file():
        raise HeadroomError(f"no frozen plan at {plan_path}; run --phase plan first.")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    stage1 = load_named_config("ham10000_stage1.yaml", "ham_stage1")
    stage2 = load_named_config("ham10000_stage2.yaml", "ham_stage2")
    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    splits_cfg = load_named_config("splits_ham10000.yaml", "ham_splits")
    learned = stage3.learned_asism
    real_split = str(learned.real_train_split)
    if FORBIDDEN_SPLIT in {real_split, TUNING_SPLIT}:
        raise HeadroomError("the protected split cannot be used to measure utility")
    check_proxy_config(learned.proxy)
    proxy = variant_proxy(PROXY_VARIANT)
    if plan["proxy"] != {**proxy, "architecture": V1_ARCHITECTURE}:
        raise HeadroomError("the frozen plan records a different proxy than this code builds")

    splits_dir = Path(splits_cfg.paths.splits_root) / namespace
    stage2_manifest = Path(stage2.paths.stage2_root) / namespace / "all_candidates.csv"
    status = freeze_measure_inputs(out_dir / "measure_inputs.json", {
        "git_commit_hash": get_git_commit_hash(),
        "proxy": {key: str(proxy[key]) for key in V1_PROXY},
        f"{real_split}_csv_sha256": _sha256(splits_dir / f"{real_split}.csv"),
        f"{TUNING_SPLIT}_csv_sha256": _sha256(splits_dir / f"{TUNING_SPLIT}.csv"),
        "all_candidates_sha256": _sha256(stage2_manifest),
    })
    print(f"measure inputs: {status}", flush=True)

    images_root = Path(stage1.paths.images_dir) / namespace
    real_records = _split_records(splits_dir, images_root, real_split)
    tuning_records = _split_records(splits_dir, images_root, TUNING_SPLIT)
    frame, _, _ = load_candidate_pool(namespace, stage3, Path(stage2.paths.stage2_root), verbose=False)
    candidates = _candidate_records(frame, pd.read_csv(stage2_manifest))

    arms: dict[str, list[str]] = plan["arms"]
    control_ids = [ARM_REAL_ONLY, ARM_ALL_SYNTHETIC]
    headroom_ids = list(plan["headroom"]["subsets"]) + [plan["headroom"]["selected_arm"]]
    # Seed-major within each block, so an interrupted pod leaves every arm at the same depth.
    jobs = [(s, i) for s in plan["positive_control"]["seeds"] for i in control_ids]
    jobs += [(s, i) for s in plan["headroom"]["seeds"] for i in headroom_ids]

    runs_path = out_dir / "headroom_runs.jsonl"
    done = ({(r["subset_id"], int(r["seed"])) for r in _read_jsonl(runs_path)}
            if runs_path.is_file() else set())
    pending = [(s, i) for s, i in jobs if (i, s) not in done]
    gpu = _gpu_name()
    print(f"{len(pending)} of {len(jobs)} proxy runs pending on {gpu}", flush=True)

    steps, batch = int(proxy["max_steps"]), int(proxy["batch_size"])
    for position, (seed, arm) in enumerate(pending, start=1):
        synthetic = [candidates[i] for i in arms[arm]]
        train = real_records + synthetic
        epochs = steps * batch / len(train)
        started = time.perf_counter()
        metrics = _measure(train, tuning_records, SimpleNamespace(**proxy), seed, device)
        seconds = time.perf_counter() - started
        _append_jsonl(runs_path, {
            "subset_id": arm,
            "seed": int(seed),
            "n_synthetic": len(synthetic),
            "metrics": {m: float(metrics[m]) for m in METRICS},
            "per_class_recall": {label: float(v["recall"])
                                 for label, v in metrics["per_class"].items()},
            # The fixed-step protocol gives the arms different epochs on purpose. Recorded, not hidden.
            "effective_dataset_size": len(train),
            "optimizer_steps": steps,
            "epochs": round(epochs, 3),
            "seconds": round(seconds, 1),
            "gpu": gpu,
            "git_commit_hash": get_git_commit_hash(),
        })
        print(f"  [{position}/{len(pending)}] {arm} seed={seed} "
              f"bal.acc={metrics[LABEL_METRIC]:.4f} epochs={epochs:.2f} ({seconds:.0f}s)", flush=True)
    return {"runs_path": str(runs_path), "completed": len(jobs)}


def run_analyze(out_dir: Path) -> dict:
    plan = json.loads((out_dir / "headroom_plan.json").read_text(encoding="utf-8"))
    rows = _read_jsonl(out_dir / "headroom_runs.jsonl")

    control = _complete_table(rows, [ARM_REAL_ONLY, ARM_ALL_SYNTHETIC],
                              plan["positive_control"]["seeds"], LABEL_METRIC)
    head_seeds = plan["headroom"]["seeds"]
    table = _complete_table(rows, list(plan["headroom"]["subsets"]), head_seeds, LABEL_METRIC)
    selected = _complete_table(rows, [plan["headroom"]["selected_arm"]], head_seeds, LABEL_METRIC)[0]
    print(f"complete grid: {control.shape[1]} control seeds per arm, "
          f"{table.shape[0]} random draws x {table.shape[1]} seeds, C x {len(selected)}\n")

    pc = welch(control[0], control[1])
    hr = sigma_s(table)
    draw_means = table.mean(axis=1)
    c_mean = float(selected.mean())
    spread = float(draw_means.std(ddof=1))
    position = {
        "c_mean": c_mean,
        "random_mean": float(draw_means.mean()),
        "random_min": float(draw_means.min()),
        "random_max": float(draw_means.max()),
        "c_minus_random_mean": float(c_mean - draw_means.mean()),
        "share_of_random_draws_below_c": float((draw_means < c_mean).mean()),
        "z_of_c_in_random_distribution": float((c_mean - draw_means.mean()) / spread) if spread > 0 else None,
    }

    report = {
        "experiment": "selection_headroom",
        "label_metric": LABEL_METRIC,
        "positive_control": pc,
        "headroom": hr,
        "selection_position": position,
        "epochs_by_arm": {arm: sorted({r["epochs"] for r in rows if r["subset_id"] == arm})
                          for arm in [ARM_REAL_ONLY, ARM_ALL_SYNTHETIC,
                                      plan["headroom"]["selected_arm"]]},
        "verdict": {
            "control_passed": pc["passed"],
            "headroom_reference_met": hr["reference_met"],
            "meaning": "the control is the only confirmatory test here: it decides whether the "
                       "proxy can see a large effect at all. The headroom figure is an exploratory "
                       "reading of how much room a selection rule could have, and C's position is "
                       "descriptive. Nothing downstream changes on the strength of either.",
        },
    }
    (out_dir / "headroom_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Positive control  B - A = {pc['difference']:+.4f}  95% CI "
          f"[{pc['ci95'][0]:+.4f}, {pc['ci95'][1]:+.4f}]  p = {pc['p_two_sided']:.4g}"
          f"  -> {'PASS' if pc['passed'] else 'FAIL'}")
    print(f"                  (v1 Stage 4 reference {CONTROL_ANCHOR_STAGE4:+.4f}, a different "
          f"recipe and a different evaluation split)")
    print(f"Headroom          sigma_s = {hr['sigma_s']:.4f}  95% CI "
          f"[{hr['ci95'][0]:.4f}, {hr['ci95'][1]:.4f}]  exploratory reference "
          f"{SIGMA_S_REFERENCE:.4f} -> {'met' if hr['reference_met'] else 'not met'}")
    print("                  (exploratory only: this reference decides nothing)")
    print(f"C                 {c_mean:.4f}  against random mean {position['random_mean']:.4f} "
          f"(range {position['random_min']:.4f}-{position['random_max']:.4f}), "
          f"{position['share_of_random_draws_below_c']:.0%} of draws below it")
    print(f"\nCONFIRMATORY: positive control {'PASS' if pc['passed'] else 'FAIL'}")
    print(f"EXPLORATORY : headroom reference {'met' if hr['reference_met'] else 'not met'}; "
          "C located, not judged. Diagnostic evidence only, not a Stage 5 result.")
    return {"report": str(out_dir / "headroom_report.json"),
            "control_passed": pc["passed"], "headroom_reference_met": hr["reference_met"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--phase", required=True, choices=["plan", "measure", "analyze"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--selected-csv", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    from scripts.utils.config import load_named_config

    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    base = Path(stage3.paths.outputs_dir)
    out_dir = (Path(args.out_dir) if args.out_dir
               else base.parent / "followup" / args.namespace / "selection_headroom")
    selected = (Path(args.selected_csv) if args.selected_csv
                else base / args.namespace / "asism_selected.csv")

    if args.phase == "plan":
        result = run_plan(args.namespace, selected, out_dir)
    elif args.phase == "measure":
        result = run_measure(args.namespace, out_dir, args.device)
    else:
        result = run_analyze(out_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
