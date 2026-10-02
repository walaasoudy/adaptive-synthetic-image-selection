#!/usr/bin/env python3
"""Experiment 2 of the utility supervision redesign: is the cheap proxy a usable instrument?

Round 3 (§11) showed the proxy utility is not reliable enough at any of three budgets. The audit
(§14) traced that to the instrument rather than to the idea: balanced accuracy on a split holding 13
df images cannot step finer than 0.011, while the between-subset spread being resolved is 0.0134.

This script tests a REPLACEMENT instrument against two criteria that were fixed numerically in
docs/ham10000_utility_supervision_redesign.md §9.1 before any run existed:

    Criterion R (reliability)  repeats_needed for reliability 0.80 must be <= 20
    Criterion A (agreement)    Spearman(new label, round-3 Stage-4 means) >= 0.60 AND p <= 0.05

BOTH must pass. Reliability alone is explicitly not sufficient: a label that repeats itself
perfectly while disagreeing with the real training recipe is a precise measurement of the wrong
thing.

WHAT IS NEW relative to the noise-floor diagnostic:
  - the label is macro one-vs-rest AUROC, not balanced accuracy (§3);
  - no baseline is subtracted from it (§5);
  - 20 repeats of one subset, not 4 seeds (§2 — this ceiling is for the 224 px / 300 step proxy
    ONLY; MAX_AFFORDABLE_REPEATS = 5 still stands for the Stage 4 recipe and round 3 is still a
    failure at 8);
  - per-class AUROC is recorded for every run, so a subset that helps only df is visible;
  - the target side costs nothing: round 3 already measured these 12 subsets at the Stage 4 recipe.

WHAT IS DELIBERATELY NOT HERE. No subset is built (that is Experiment 1, §8, and it is CPU work that
this script does not do). The ranking network is not touched. No selection is produced. C2, D2 and
every v2 stage are untouched. final_eval_heldout is unreachable: the guard below refuses if it ever
appears as the training or the evaluation split.

WHAT IS REUSED, NOT COPIED. The proxy training path is `_measure` from
scripts/asism/ham10000_03_build_utility_subsets — byte-for-byte the code that produced the v1
labels. The variance maths is imported from ham10000_proxy_noise_floor, which is left unmodified so
that its analyze phase keeps reproducing §11's published numbers exactly.

THREE PHASES, deliberately separate commands:

    --phase plan      CPU, seconds. Freezes the 12 subsets, the 20 seeds, the target values taken
                      from round 3, and both acceptance thresholds. Nothing is measured.
    --phase measure   GPU, ~1 hour. 240 proxy runs (+20 more if --with-baseline). Resumable.
    --phase analyze   CPU, seconds. REFUSES unless the grid is exactly complete, then evaluates
                      Criterion R and Criterion A and writes the verdict.

Usage:
    python scripts/followup/ham10000_utility_instrument_check.py --namespace ham-stratified-v1 --phase plan
    python scripts/followup/ham10000_utility_instrument_check.py --namespace ham-stratified-v1 --phase measure --device cuda
    python scripts/followup/ham10000_utility_instrument_check.py --namespace ham-stratified-v1 --phase analyze
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.followup.ham10000_proxy_noise_floor import (  # noqa: E402
    BASELINE_ID,
    FORBIDDEN_SPLIT,
    TUNING_SPLIT,
    V1_ARCHITECTURE,
    V1_PROXY,
    _append_jsonl,
    _gpu_name,
    _read_jsonl,
    _sha256,
    check_proxy_config,
    freeze_measure_inputs,
    reliability,
    repeats_needed,
    variance_components,
    variant_proxy,
)

# ---- pre-registered in docs/ham10000_utility_supervision_redesign.md §9.1, before any run --------
N_REPEATS = 20
SEEDS = tuple(range(42, 42 + N_REPEATS))        # 42..61; 42 is v1's own seed
PROXY_VARIANT = "v1"                            # 224 px / 300 steps — the cheap recipe, §2
LABEL_METRIC = "macro_auroc_ovr"                # §3: continuous, not quantised by 13 df images
METRICS = ("balanced_accuracy", "macro_auroc_ovr", "macro_f1", "accuracy")

RELIABILITY_TARGET = 0.80
MAX_CHEAP_REPEATS = 20                          # §2. The Stage 4 ceiling of 5 is NOT changed.
AGREEMENT_MIN_RHO = 0.60
AGREEMENT_MAX_P = 0.05
N_PERMUTATIONS = 200_000
PERMUTATION_SEED = 42
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0

# Where the target comes from: round 3, already measured and already paid for.
TARGET_ROUND_DIRNAME = "proxy_noise_floor_stage4size"
TARGET_RECIPE = "512 px / 3000 steps (Stage 4 recipe)"


class InstrumentCheckError(RuntimeError):
    pass


# ==============================================================================================
# Pure statistics (no torch, no files): unit-testable
# ==============================================================================================


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, so Spearman is the Pearson correlation of these."""
    values = np.asarray(values, dtype=float)
    order = values.argsort(kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # average the ranks of any tied group
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(unique))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def spearman(a, b) -> float:
    ra, rb = average_ranks(a), average_ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denominator = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / denominator) if denominator else float("nan")


def permutation_p_value(a, b, n_permutations: int = N_PERMUTATIONS,
                        seed: int = PERMUTATION_SEED) -> dict:
    """One-sided (H1: rho > 0) permutation test on Spearman's rho.

    The p-value counts permutations whose rho is at least the observed one, with the observed
    arrangement itself included in both numerator and denominator (the +1 convention), so a p-value
    of exactly zero is never reported.
    """
    ra, rb = average_ranks(a), average_ranks(b)
    observed = spearman(a, b)
    ra_centred = ra - ra.mean()
    rb_centred = rb - rb.mean()
    scale = float(np.sqrt((ra_centred ** 2).sum() * (rb_centred ** 2).sum()))
    rng = np.random.default_rng(seed)
    tiled = np.tile(rb_centred, (n_permutations, 1))
    permuted = rng.permuted(tiled, axis=1)
    rhos = permuted @ ra_centred / scale
    at_least = int((rhos >= observed - 1e-12).sum())
    return {
        "rho": observed,
        "p_one_sided": (at_least + 1) / (n_permutations + 1),
        "n_permutations": n_permutations,
        "permutation_seed": seed,
        "alternative": "rho > 0",
    }


def bootstrap_over_subsets(table: np.ndarray, max_repeats: int = MAX_CHEAP_REPEATS) -> dict:
    """Resample SUBSETS — the unit we generalise over — to show how uncertain 12 of them leave us."""
    table = np.asarray(table, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    iccs, needs = [], []
    for _ in range(N_BOOTSTRAP):
        sample = table[rng.integers(0, len(table), len(table))]
        component = variance_components(sample)
        iccs.append(component["icc_single_run"])
        need = repeats_needed(component["between_variance"], component["within_variance"],
                              RELIABILITY_TARGET, max_repeats)
        needs.append(max_repeats + 1 if need is None else need)   # "more than the ceiling"
    needs = np.asarray(needs, dtype=float)
    return {
        "icc_ci95": [float(np.quantile(iccs, 0.025)), float(np.quantile(iccs, 0.975))],
        "repeats_needed_median": float(np.median(needs)),
        "repeats_needed_p90": float(np.quantile(needs, 0.9)),
        "share_of_resamples_within_ceiling": float(np.mean(needs <= max_repeats)),
    }


def criterion_reliability(table: np.ndarray) -> dict:
    """Criterion R: repeats_needed for reliability 0.80 must be <= MAX_CHEAP_REPEATS."""
    component = variance_components(table)
    between, within = component["between_variance"], component["within_variance"]
    needed = repeats_needed(between, within, RELIABILITY_TARGET, MAX_CHEAP_REPEATS)
    return {
        **component,
        "per_subset_sd": [float(x) for x in np.asarray(table, dtype=float).std(axis=1, ddof=1)],
        "reliability_by_repeats": {str(r): reliability(between, within, r)
                                   for r in (1, 2, 5, 10, 15, 20)},
        "repeats_needed": needed,
        "ceiling": MAX_CHEAP_REPEATS,
        "target": RELIABILITY_TARGET,
        "bootstrap": bootstrap_over_subsets(table),
        "passed": bool(needed is not None and needed <= MAX_CHEAP_REPEATS),
    }


def criterion_agreement(new_label: np.ndarray, target: np.ndarray) -> dict:
    """Criterion A: rho >= 0.60 AND one-sided permutation p <= 0.05, against the Stage 4 recipe."""
    test = permutation_p_value(new_label, target)
    return {
        **test,
        "min_rho": AGREEMENT_MIN_RHO,
        "max_p": AGREEMENT_MAX_P,
        "target_recipe": TARGET_RECIPE,
        "passed": bool(test["rho"] >= AGREEMENT_MIN_RHO and test["p_one_sided"] <= AGREEMENT_MAX_P),
    }


# ==============================================================================================
# Phases
# ==============================================================================================


def _target_from_round3(round3_dir: Path, metric: str = LABEL_METRIC) -> tuple[list[str], dict]:
    """The 12 subsets and their Stage-4-recipe means. Refuses anything but a complete 12 x 4 grid."""
    report_path = round3_dir / "noise_floor_report.json"
    if not report_path.is_file():
        raise InstrumentCheckError(
            f"no round-3 report at {report_path}. Experiment 2 compares against round 3's "
            "Stage-4-recipe measurements; without them there is nothing to agree with."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    per_subset = report["per_subset"][metric]
    subsets = sorted(sid for sid in per_subset if sid != BASELINE_ID)
    if len(subsets) != 12:
        raise InstrumentCheckError(f"round 3 holds {len(subsets)} subsets, expected 12")
    incomplete = {sid: len(per_subset[sid]) for sid in subsets if len(per_subset[sid]) != 4}
    if incomplete:
        raise InstrumentCheckError(f"round 3 subsets without 4 seeds: {incomplete}")
    return subsets, {sid: float(np.mean(per_subset[sid])) for sid in subsets}


def run_plan(namespace: str, v1_dir: Path, round3_dir: Path, out_dir: Path,
             with_baseline: bool) -> dict:
    from scripts.utils.manifest import get_git_commit_hash

    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "instrument_check_plan.json"
    if plan_path.is_file():
        raise InstrumentCheckError(
            f"a frozen plan already exists at {plan_path}. It is never re-drawn: delete the whole "
            "output directory if this experiment is genuinely being started again."
        )

    subsets, target = _target_from_round3(round3_dir)
    plan = {
        "experiment": "utility_instrument_check",
        "design_document": "docs/ham10000_utility_supervision_redesign.md §9",
        "namespace": namespace,
        "chosen_subsets": subsets,
        "target_metric": LABEL_METRIC,
        "target_recipe": TARGET_RECIPE,
        "target_values": target,
        "target_source": str(round3_dir / "noise_floor_report.json"),
        "target_source_sha256": _sha256(round3_dir / "noise_floor_report.json"),
        "seeds": list(SEEDS),
        "n_repeats": N_REPEATS,
        "with_baseline": bool(with_baseline),
        "proxy_variant": PROXY_VARIANT,
        "proxy": {**variant_proxy(PROXY_VARIANT), "architecture": V1_ARCHITECTURE},
        "criteria": {
            "reliability": {"target": RELIABILITY_TARGET, "max_repeats": MAX_CHEAP_REPEATS},
            "agreement": {"min_rho": AGREEMENT_MIN_RHO, "max_p": AGREEMENT_MAX_P,
                          "n_permutations": N_PERMUTATIONS, "permutation_seed": PERMUTATION_SEED,
                          "alternative": "rho > 0"},
            "combined": "both must pass; reliability alone is not sufficient",
            "attempts": "one. If it fails, R is not raised, the metric is not swapped, the subset "
                        "sample is not changed and these thresholds are not revisited.",
        },
        "v1_utility_subsets_sha256": _sha256(v1_dir / "utility_subsets.jsonl"),
        "git_commit_hash": get_git_commit_hash(),
    }
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    runs = len(subsets) * N_REPEATS + (N_REPEATS if with_baseline else 0)
    print(f"Frozen: {len(subsets)} subsets x {N_REPEATS} repeats"
          f"{' + baseline' if with_baseline else ''} = {runs} runs -> {plan_path}", flush=True)
    return {"plan_path": str(plan_path), "runs_planned": runs, "subsets": len(subsets)}


def run_measure(namespace: str, v1_dir: Path, out_dir: Path, device: str | None) -> dict:
    # Reused, not copied: the proxy training code must be the one that produced the v1 labels.
    import pandas as pd

    from scripts.asism.ham10000_03_build_utility_subsets import (
        _candidate_records, _measure, _split_records,
    )
    from scripts.asism.ham10000_ranking import load_candidate_pool
    from scripts.utils.config import load_named_config
    from scripts.utils.manifest import get_git_commit_hash

    plan_path = out_dir / "instrument_check_plan.json"
    if not plan_path.is_file():
        raise InstrumentCheckError(f"no frozen plan at {plan_path}; run --phase plan first.")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["v1_utility_subsets_sha256"] != _sha256(v1_dir / "utility_subsets.jsonl"):
        raise InstrumentCheckError("v1 utility_subsets.jsonl changed since the plan was frozen")

    stage1 = load_named_config("ham10000_stage1.yaml", "ham_stage1")
    stage2 = load_named_config("ham10000_stage2.yaml", "ham_stage2")
    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    splits_cfg = load_named_config("splits_ham10000.yaml", "ham_splits")
    learned = stage3.learned_asism
    real_split = str(learned.real_train_split)
    if FORBIDDEN_SPLIT in {real_split, TUNING_SPLIT}:
        raise InstrumentCheckError("the protected split cannot be used to measure utility")
    check_proxy_config(learned.proxy)
    proxy = variant_proxy(PROXY_VARIANT)
    if plan["proxy"] != {**proxy, "architecture": V1_ARCHITECTURE}:
        raise InstrumentCheckError("the frozen plan records a different proxy than this code builds")

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
    subsets = {row["subset_id"]: row for row in _read_jsonl(v1_dir / "utility_subsets.jsonl")}

    runs_path = out_dir / "instrument_check_runs.jsonl"
    done = {(r["subset_id"], r["seed"]) for r in _read_jsonl(runs_path)} if runs_path.is_file() else set()
    ids = ([BASELINE_ID] if plan["with_baseline"] else []) + list(plan["chosen_subsets"])
    # Seed-major: an interrupted pod still leaves every subset at the same number of repeats, which
    # is what the balanced variance decomposition needs.
    jobs = [(seed, sid) for seed in plan["seeds"] for sid in ids]
    pending = [(seed, sid) for seed, sid in jobs if (sid, seed) not in done]
    gpu = _gpu_name()
    print(f"{len(pending)} of {len(jobs)} proxy runs pending on {gpu}", flush=True)

    for position, (seed, sid) in enumerate(pending, start=1):
        synthetic = [] if sid == BASELINE_ID else [candidates[i] for i in subsets[sid]["image_ids"]]
        started = time.perf_counter()
        metrics = _measure(real_records + synthetic, tuning_records, SimpleNamespace(**proxy), seed, device)
        seconds = time.perf_counter() - started
        _append_jsonl(runs_path, {
            "subset_id": sid,
            "seed": int(seed),
            "n_synthetic": len(synthetic),
            "metrics": {m: float(metrics[m]) for m in METRICS},
            # §4: a subset that helps only df must not be invisible.
            "per_class_auroc": {label: float(values["auroc_ovr"])
                                for label, values in metrics["per_class"].items()},
            "per_class_recall": {label: float(values["recall"])
                                 for label, values in metrics["per_class"].items()},
            "seconds": round(seconds, 1),
            "gpu": gpu,
            "git_commit_hash": get_git_commit_hash(),
        })
        print(f"  [{position}/{len(pending)}] {sid} seed={seed} "
              f"auroc={metrics['macro_auroc_ovr']:.4f} bal.acc={metrics['balanced_accuracy']:.4f} "
              f"({seconds:.0f}s)", flush=True)
    return {"runs_path": str(runs_path), "runs": len(jobs)}


def _complete_table(rows: list[dict], subsets: list[str], seeds: list[int], metric: str) -> np.ndarray:
    """(n_subsets, n_repeats) or a refusal naming exactly what is missing.

    Round 3 taught this: its analyser silently dropped incomplete seeds, so a killed pod would have
    produced a report that looked finished. Nothing partial is analysed here.
    """
    cells = {}
    duplicates = []
    for row in rows:
        key = (row["subset_id"], int(row["seed"]))
        if key in cells:
            duplicates.append(key)
        cells[key] = row["metrics"][metric]
    if duplicates:
        raise InstrumentCheckError(f"duplicate runs for {sorted(set(duplicates))}")
    missing = [(sid, seed) for sid in subsets for seed in seeds if (sid, seed) not in cells]
    if missing:
        raise InstrumentCheckError(
            f"the grid is incomplete: {len(missing)} of {len(subsets) * len(seeds)} cells missing "
            f"(e.g. {missing[:5]}). Finish --phase measure; nothing partial is analysed."
        )
    return np.array([[cells[(sid, seed)] for seed in seeds] for sid in subsets], dtype=float)


def run_analyze(out_dir: Path) -> dict:
    plan_path = out_dir / "instrument_check_plan.json"
    if not plan_path.is_file():
        raise InstrumentCheckError(f"no frozen plan at {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(out_dir / "instrument_check_runs.jsonl")
    subsets, seeds = list(plan["chosen_subsets"]), [int(s) for s in plan["seeds"]]

    subset_rows = [r for r in rows if r["subset_id"] != BASELINE_ID]
    table = _complete_table(subset_rows, subsets, seeds, LABEL_METRIC)
    print(f"complete grid: {len(subsets)} subsets x {len(seeds)} repeats = {table.size} runs",
          flush=True)

    new_label = table.mean(axis=1)
    target = np.array([plan["target_values"][sid] for sid in subsets], dtype=float)

    reliability_result = criterion_reliability(table)
    agreement_result = criterion_agreement(new_label, target)
    both = bool(reliability_result["passed"] and agreement_result["passed"])

    secondary = {}
    for metric in METRICS:
        if metric == LABEL_METRIC:
            continue
        try:
            other = _complete_table(subset_rows, subsets, seeds, metric)
        except InstrumentCheckError as error:
            secondary[metric] = {"skipped": str(error)}
            continue
        component = variance_components(other)
        secondary[metric] = {
            "icc_single_run": component["icc_single_run"],
            "repeats_needed": repeats_needed(component["between_variance"],
                                             component["within_variance"],
                                             RELIABILITY_TARGET, MAX_CHEAP_REPEATS),
            "spearman_vs_round3_label_metric": spearman(other.mean(axis=1), target),
        }

    baseline_rows = [row for row in rows if row["subset_id"] == BASELINE_ID]
    report = {
        "experiment": "utility_instrument_check",
        "design_document": "docs/ham10000_utility_supervision_redesign.md §9",
        "label_metric": LABEL_METRIC,
        "n_subsets": len(subsets),
        "n_repeats": len(seeds),
        "criterion_R_reliability": reliability_result,
        "criterion_A_agreement": agreement_result,
        "verdict": {
            "reliability_passed": reliability_result["passed"],
            "agreement_passed": agreement_result["passed"],
            "accepted": both,
            "meaning": ("the cheap proxy is accepted as the utility instrument; Experiment 1 and "
                        "the full supervision may proceed at R = %s repeats"
                        % reliability_result["repeats_needed"])
            if both else
            ("the cheap proxy is REJECTED. Per §9.2 this is not retried at a different R, metric or "
             "sample; the redesign moves to the next candidate instrument (§9.3)."),
        },
        "secondary_readings": {
            "role": "reported so nothing is hidden; no power to overturn R or A (§9.2)",
            "per_metric": secondary,
        },
        "baseline": {
            "role": "interpretation only; never subtracted from the label (§5)",
            "runs": len(baseline_rows),
            "mean": float(np.mean([b["metrics"][LABEL_METRIC] for b in baseline_rows]))
            if baseline_rows else None,
            "sd": float(np.std([b["metrics"][LABEL_METRIC] for b in baseline_rows], ddof=1))
            if len(baseline_rows) > 1 else None,
        },
        "new_label_per_subset": {sid: float(value) for sid, value in zip(subsets, new_label)},
        "target_per_subset": plan["target_values"],
        "timing": {"runs": len(rows),
                   "mean_seconds": float(np.mean([row["seconds"] for row in rows])) if rows else None},
    }
    report_path = out_dir / "instrument_check_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nCriterion R (reliability): repeats_needed = {reliability_result['repeats_needed']} "
          f"(ceiling {MAX_CHEAP_REPEATS}), ICC {reliability_result['icc_single_run']:.3f} "
          f"-> {'PASS' if reliability_result['passed'] else 'FAIL'}")
    print(f"Criterion A (agreement)  : rho = {agreement_result['rho']:+.3f} "
          f"(min {AGREEMENT_MIN_RHO}), p = {agreement_result['p_one_sided']:.4f} "
          f"(max {AGREEMENT_MAX_P}) -> {'PASS' if agreement_result['passed'] else 'FAIL'}")
    print(f"\nVERDICT: the cheap proxy is {'ACCEPTED' if both else 'REJECTED'}")
    return {"report": str(report_path), "accepted": both}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--phase", required=True, choices=["plan", "measure", "analyze"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--v1-dir", default=None)
    parser.add_argument("--round3-dir", default=None,
                        help=f"default: outputs/ham10000/followup/<ns>/{TARGET_ROUND_DIRNAME}")
    parser.add_argument("--out-dir", default=None,
                        help="default: outputs/ham10000/followup/<ns>/utility_instrument_check")
    parser.add_argument("--with-baseline", action="store_true",
                        help="also train the real-only baseline 20 times (§5: reported, never "
                             "subtracted). Adds 20 runs, about 5 minutes.")
    args = parser.parse_args()

    if args.v1_dir and args.out_dir and args.round3_dir:
        v1_dir, out_dir = Path(args.v1_dir), Path(args.out_dir)
        round3_dir = Path(args.round3_dir)
    else:
        from scripts.utils.config import load_named_config

        stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
        base = Path(stage3.paths.outputs_dir)
        followup = base.parent / "followup" / args.namespace
        v1_dir = Path(args.v1_dir) if args.v1_dir else base / args.namespace / "learned"
        round3_dir = Path(args.round3_dir) if args.round3_dir else followup / TARGET_ROUND_DIRNAME
        out_dir = Path(args.out_dir) if args.out_dir else followup / "utility_instrument_check"

    if args.phase == "plan":
        result = run_plan(args.namespace, v1_dir, round3_dir, out_dir, args.with_baseline)
    elif args.phase == "measure":
        result = run_measure(args.namespace, v1_dir, out_dir, args.device)
    else:
        result = run_analyze(out_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
