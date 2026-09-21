#!/usr/bin/env python3
"""Follow-up diagnostic (post-test): how much of a Stage 3 utility label is signal, and how much noise?

WHY THIS EXISTS. In ham-final-v1 every utility subset was measured by ONE proxy training run (seed 42)
against ONE real-only baseline (seed 42). The set model trained on those 80 labels reached a
validation Spearman of -0.36. One explanation is that the labels were mostly seed noise: if training
the same subset twice moves balanced accuracy as much as swapping one subset for another, there was
nothing for the set model to learn. This script measures exactly that, and nothing else.

WHAT IT DOES.
    12 of the 80 v1 subsets, each re-measured under 4 seeds (42..45), with the SAME proxy recipe as
    v1, plus the real-only baseline under the same 4 seeds: 52 proxy runs in total. Twelve subsets
    rather than eight because the between-subset spread is the hard part to estimate and only more
    subsets narrow it; four seeds already pin down the within-subset noise.

    within-subset variance   same images, different seed        -> noise
    between-subset variance  different images, averaged seeds    -> the signal the set model needs
    ICC = between / (between + within)                           -> share of one label that is signal
    reliability(R) = between / (between + within / R)            -> if each label averaged R runs
    repeats_needed = smallest R with reliability >= 0.80         -> the number Stage 3 v2 should use
    correlation ceiling ~ sqrt(reliability)                      -> an APPROXIMATE reference for how
                                                                    well any model could track such
                                                                    labels; not a strict bound on
                                                                    Spearman (see correlation_ceiling)

    Sensitivity only: the same analysis restricted to subsets of equal size ("same-size ICC"). With
    3, 4 and 5 subsets per size it is a rough check on whether subset size alone drives the spread,
    not an independent estimate, and it never enters the decision rule below.

    Every run records all four metrics, so the same 52 runs also show which utility metric is least
    noisy. Balanced accuracy moves in jumps on asism_tuning_heldout (df has 13 images: one image is
    ~0.011 of balanced accuracy); macro AUROC reads the probabilities and may be far smoother.

THE DECISION RULE IS FIXED HERE, BEFORE ANY RUN (RELIABILITY_TARGET, MAX_AFFORDABLE_REPEATS):
    per metric, on the v1-style label:
    * between-subset variance <= 0      -> no detectable signal at this proxy budget; repetitions
                                           cannot fix it; the proxy itself must change.
    * repeats_needed <= 5               -> use that many repetitions in Stage 3 v2.
    * repeats_needed  > 5               -> repetitions alone are too expensive; lengthen the proxy.
    The v2 utility metric is the one with the highest single-run ICC among those whose verdict is
    "repeat"; if none is, v2 keeps balanced_accuracy and lengthens the proxy.

WHAT IT NEVER DOES. It does not modify, re-select or re-rank anything from ham-final-v1: the 616
selected images, the v1 utility files and the v1 code are read-only here. It never reads
final_eval_heldout. Nothing it measures is a thesis result; it is reported as a post-hoc diagnostic.

THREE PHASES, as in the v1 utility script:
    --phase plan      CPU, seconds. Picks the 12 subsets and freezes the plan. Refuses to overwrite a
                      different frozen plan, so the subsets cannot be re-picked after seeing results.
    --phase measure   GPU. 52 proxy runs, resumable: every finished run is appended at once.
    --phase analyze   CPU, seconds. Runs anywhere (also on a laptop) from the saved runs.

Usage:
    python scripts/followup/ham10000_proxy_noise_floor.py --namespace ham-stratified-v1 --phase plan
    python scripts/followup/ham10000_proxy_noise_floor.py --namespace ham-stratified-v1 --phase measure
    python scripts/followup/ham10000_proxy_noise_floor.py --namespace ham-stratified-v1 --phase analyze
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ---- pre-registered, fixed before any run ------------------------------------------------------
SEEDS = (42, 43, 44, 45)              # 42 repeats v1's own seed: a same-seed rerun shows GPU noise
N_SUBSETS = 12
SELECTION_SEED = 20260921             # picks one subset per utility-rank bin; never re-drawn
METRICS = ("balanced_accuracy", "macro_auroc_ovr", "macro_f1", "accuracy")
V1_METRIC = "balanced_accuracy"       # the metric v1's labels were measured in
RELIABILITY_TARGET = 0.80
MAX_AFFORDABLE_REPEATS = 5
MAX_REPEATS_REPORTED = 10
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0

# The v1 proxy recipe (configs/ham10000_stage3.yaml at b6187d5, learned_asism.proxy). measure refuses
# to run if the current config differs, so this cannot silently measure the noise of another proxy.
# The seed is excluded: varying it is the point. The backbone is not a config key: build_model in
# scripts/utils/classifier.py is DenseNet-121, and the commit is frozen in measure_inputs.json.
V1_PROXY = {
    "resolution": 224,
    "max_steps": 300,
    "batch_size": 32,
    "learning_rate": 1.0e-4,
    "weight_decay": 1.0e-4,
    "dropout_p": 0.2,
    "pretrained_source": "imagenet",
}
V1_ARCHITECTURE = "densenet121"

TUNING_SPLIT = "asism_tuning_heldout"
FORBIDDEN_SPLIT = "final_eval_heldout"
BASELINE_ID = "__baseline__"


class NoiseFloorError(RuntimeError):
    pass


# ==============================================================================================
# Pure statistics (no torch, no files): unit-tested
# ==============================================================================================


def pick_subsets(v1_results: pd.DataFrame, n: int = N_SUBSETS, seed: int = SELECTION_SEED) -> list[str]:
    """One subset from each of n equal bins of the v1 utility ranking.

    Picking the top, middle and bottom subsets by hand would stretch the between-subset spread and
    make the signal look larger than it is. A purely random draw could land all in the middle. One
    random draw per rank bin covers the whole range while staying close to a random sample of the 80.

    The v1 utility is used ONLY to stratify coverage for this diagnostic. The choice is made and
    frozen (run_plan) before any new run, and nothing measured by the 52 new runs feeds back into it.
    """
    utility = v1_results["augmented_balanced_accuracy"] - v1_results["real_only_balanced_accuracy"]
    ranked = v1_results.assign(_u=utility).sort_values(["_u", "subset_id"])["subset_id"].to_numpy()
    rng = np.random.default_rng(seed)
    return [str(rng.choice(bin_ids)) for bin_ids in np.array_split(ranked, n)]


def variance_components(table: np.ndarray, groups: np.ndarray | None = None) -> dict:
    """One-way random-effects ANOVA on a (subsets x seeds) table.

    within  = mean square within subsets                  (seed noise of ONE run)
    between = (mean square between - within) / k          (true spread between subsets)

    With `groups` (one label per subset, here its size), the between term is measured around each
    group's own mean instead of the grand mean, with n - g degrees of freedom: the spread between
    subsets of the same size. SENSITIVITY ANALYSIS ONLY. With a handful of subsets per group the
    group means are themselves noisy, so this indicates whether size alone could explain the spread;
    it is not a strong, independent estimate of the signal about which images were chosen.

    The between estimate can come out negative when the subsets differ less than the noise predicts;
    it is reported as it is and treated as "no detectable signal", never clipped silently.
    """
    table = np.asarray(table, dtype=float)
    n, k = table.shape
    if n < 2 or k < 2:
        raise NoiseFloorError(f"need at least 2 subsets and 2 seeds, got {n}x{k}")
    subset_means = table.mean(axis=1)
    if groups is None:
        centres, df_between = np.full(n, table.mean()), n - 1
    else:
        groups = np.asarray(groups)
        centres = np.array([subset_means[groups == g].mean() for g in groups])
        df_between = n - len(np.unique(groups))
        if df_between < 2:
            raise NoiseFloorError(f"too few subsets per group for a grouped estimate (df={df_between})")
    ms_between = k * np.sum((subset_means - centres) ** 2) / df_between
    ms_within = np.sum((table - subset_means[:, None]) ** 2) / (n * (k - 1))
    between = (ms_between - ms_within) / k
    icc = between / (between + ms_within) if between > 0 else 0.0
    return {
        "n_subsets": int(n),
        "n_seeds": int(k),
        "within_variance": float(ms_within),
        "within_sd": float(np.sqrt(ms_within)),
        "between_variance": float(between),
        "between_sd": float(np.sqrt(between)) if between > 0 else 0.0,
        "icc_single_run": float(icc),
    }


def reliability(between: float, within: float, repeats: int) -> float:
    """Spearman-Brown: reliability of a label that averages `repeats` independent runs."""
    if between <= 0:
        return 0.0
    return float(between / (between + within / repeats))


def correlation_ceiling(between: float, within: float, repeats: int = 1) -> float:
    """Approximate reference ceiling on how well a predictor can correlate with labels this noisy.

    sqrt(reliability) bounds the PEARSON correlation only under the classical measurement model
    (noise independent of the predictor and of the true utility). For Spearman it is an
    approximation, not a strict mathematical upper bound, and it inherits the uncertainty of the
    reliability estimate itself (12 subsets). Report it as a diagnostic reference: with repeats=1 it
    indicates roughly how far v1's set model could have got against single-run labels.
    """
    return float(np.sqrt(reliability(between, within, repeats)))


def repeats_needed(between: float, within: float, target: float = RELIABILITY_TARGET,
                   max_repeats: int = MAX_REPEATS_REPORTED) -> int | None:
    """Smallest R whose reliability reaches the target; None if no R up to max_repeats does."""
    for repeats in range(1, max_repeats + 1):
        if reliability(between, within, repeats) >= target:
            return repeats
    return None


def verdict(between: float, needed: int | None) -> str:
    if between <= 0:
        return "no_detectable_signal"          # repetitions cannot help; change the proxy
    if needed is not None and needed <= MAX_AFFORDABLE_REPEATS:
        return "repeat"                         # average `needed` runs per subset in v2
    return "lengthen_proxy"                     # repetitions alone too expensive


def bootstrap_over_subsets(table: np.ndarray, n_resamples: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> dict:
    """Resample SUBSETS (the unit we generalise over) to show how uncertain 12 subsets leave us."""
    table = np.asarray(table, dtype=float)
    rng = np.random.default_rng(seed)
    iccs, needs = [], []
    for _ in range(n_resamples):
        sample = table[rng.integers(0, len(table), len(table))]
        comp = variance_components(sample)
        iccs.append(comp["icc_single_run"])
        need = repeats_needed(comp["between_variance"], comp["within_variance"])
        # "more than MAX_REPEATS_REPORTED" is stored as one past it, so quantiles stay finite
        needs.append(MAX_REPEATS_REPORTED + 1 if need is None else need)
    needs = np.asarray(needs, dtype=float)
    return {
        "icc_ci95": [float(np.quantile(iccs, 0.025)), float(np.quantile(iccs, 0.975))],
        "repeats_needed_median": float(np.median(needs)),
        "repeats_needed_p90": float(np.quantile(needs, 0.9)),   # 11 means "more than 10"
        "share_of_resamples_within_affordable": float(np.mean(needs <= MAX_AFFORDABLE_REPEATS)),
    }


def analyse_table(table: np.ndarray, sizes: np.ndarray | None = None) -> dict:
    table = np.asarray(table, dtype=float)
    comp = variance_components(table)
    between, within = comp["between_variance"], comp["within_variance"]
    need = repeats_needed(between, within)
    result = {
        **comp,
        "per_subset_sd": [float(x) for x in table.std(axis=1, ddof=1)],
        "reliability_by_repeats": {str(r): reliability(between, within, r) for r in range(1, MAX_REPEATS_REPORTED + 1)},
        "correlation_ceiling_by_repeats": {str(r): correlation_ceiling(between, within, r) for r in (1, 2, 3, 4, 5)},
        "repeats_needed": need,
        "verdict": verdict(between, need),
        "bootstrap": bootstrap_over_subsets(table),
    }
    if sizes is not None:
        sizes = np.asarray(sizes)
        result["mean_by_size"] = {str(s): float(table[sizes == s].mean()) for s in sorted(np.unique(sizes))}
        try:
            within_size = variance_components(table, groups=sizes)
            result["same_size_only"] = {
                "role": "sensitivity",
                "note": "few subsets per size, no bootstrap CI; indicative only, never used by the decision rule",
                "subsets_per_size": {str(s): int((sizes == s).sum()) for s in sorted(np.unique(sizes))},
                **within_size,
                "correlation_ceiling_single_run": correlation_ceiling(
                    within_size["between_variance"], within_size["within_variance"]),
                "repeats_needed": repeats_needed(within_size["between_variance"], within_size["within_variance"]),
            }
        except NoiseFloorError as error:
            result["same_size_only"] = {"skipped": str(error)}
    return result


def check_proxy_config(proxy_cfg) -> None:
    """Refuse unless the configured proxy is exactly v1's (seed aside)."""
    differing = {}
    for key, expected in V1_PROXY.items():
        current = proxy_cfg[key]
        same = current == expected if isinstance(expected, str) else np.isclose(float(current), float(expected))
        if not same:
            differing[key] = (current, expected)
    if differing:
        raise NoiseFloorError(f"proxy config differs from v1 (current, v1): {differing}; refusing")


def freeze_measure_inputs(path: Path, inputs: dict) -> str:
    """Write the measure inputs on the first run; on a resume, refuse if any of them changed."""
    if path.is_file():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if frozen != inputs:
            changed = sorted(k for k in set(frozen) | set(inputs) if frozen.get(k) != inputs.get(k))
            raise NoiseFloorError(f"measure inputs changed since the first run ({changed}); refusing to mix runs")
        return "identical"
    path.write_text(json.dumps(inputs, indent=2), encoding="utf-8")
    return "frozen"


def choose_v2_metric(per_metric: dict) -> dict:
    """Apply the pre-registered metric rule to the v1-style analyses."""
    eligible = {m: block for m, block in per_metric.items() if block["verdict"] == "repeat"}
    if not eligible:
        return {"metric": V1_METRIC, "repeats": None, "reason": "no metric reaches the target within "
                f"{MAX_AFFORDABLE_REPEATS} repeats; keep {V1_METRIC} and lengthen the proxy"}
    best = max(eligible, key=lambda m: eligible[m]["icc_single_run"])
    return {"metric": best, "repeats": eligible[best]["repeats_needed"],
            "reason": "highest single-run ICC among metrics whose verdict is 'repeat'"}


# ==============================================================================================
# Files
# ==============================================================================================


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def run_plan(v1_dir: Path, out_dir: Path) -> dict:
    results_path, subsets_path = v1_dir / "utility_results.jsonl", v1_dir / "utility_subsets.jsonl"
    for path in (results_path, subsets_path):
        if not path.is_file():
            raise NoiseFloorError(f"missing v1 file {path}")
    v1 = pd.DataFrame(_read_jsonl(results_path))
    chosen = pick_subsets(v1)
    sizes = v1.set_index("subset_id").loc[chosen, "size"].astype(int).tolist()
    plan = {
        "diagnostic": "proxy_noise_floor",
        "post_hoc": True,
        "chosen_subsets": chosen,
        "chosen_subset_sizes": sizes,
        "seeds": list(SEEDS),
        "metrics": list(METRICS),
        "selection_seed": SELECTION_SEED,
        "selection_basis": "v1 utility ranks, used only to stratify coverage; frozen before any new run, "
                           "and the new runs never influence the choice",
        "proxy": {**V1_PROXY, "architecture": V1_ARCHITECTURE},
        "reliability_target": RELIABILITY_TARGET,
        "max_affordable_repeats": MAX_AFFORDABLE_REPEATS,
        "v1_utility_results_sha256": _sha256(results_path),
        "v1_utility_subsets_sha256": _sha256(subsets_path),
    }
    plan_path = out_dir / "noise_floor_plan.json"
    if plan_path.is_file():
        frozen = json.loads(plan_path.read_text(encoding="utf-8"))
        if frozen != plan:
            raise NoiseFloorError(f"{plan_path} is frozen and differs from what this run would write; refusing")
        return {"plan_path": str(plan_path), "chosen_subsets": chosen, "status": "already frozen, identical"}
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return {"plan_path": str(plan_path), "chosen_subsets": chosen, "status": "frozen"}


def _gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - only a label for the cost estimate
        return "unknown"


def run_measure(namespace: str, v1_dir: Path, out_dir: Path, device: str | None) -> dict:
    # Reused, not copied: the proxy recipe must be byte-for-byte the one that produced the v1 labels.
    from scripts.asism.ham10000_03_build_utility_subsets import _candidate_records, _measure, _split_records
    from scripts.asism.ham10000_ranking import load_candidate_pool
    from scripts.utils.config import load_named_config
    from scripts.utils.manifest import get_git_commit_hash

    plan = json.loads((out_dir / "noise_floor_plan.json").read_text(encoding="utf-8"))
    if plan["v1_utility_subsets_sha256"] != _sha256(v1_dir / "utility_subsets.jsonl"):
        raise NoiseFloorError("v1 utility_subsets.jsonl changed since the plan was frozen")

    stage1 = load_named_config("ham10000_stage1.yaml", "ham_stage1")
    stage2 = load_named_config("ham10000_stage2.yaml", "ham_stage2")
    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    splits_cfg = load_named_config("splits_ham10000.yaml", "ham_splits")
    learned = stage3.learned_asism
    if FORBIDDEN_SPLIT in {str(learned.real_train_split), TUNING_SPLIT}:
        raise NoiseFloorError("the protected split cannot be used to measure utility")
    check_proxy_config(learned.proxy)
    if plan["proxy"] != {**V1_PROXY, "architecture": V1_ARCHITECTURE}:
        raise NoiseFloorError("the frozen plan records a different proxy than this code; refusing")

    splits_dir = Path(splits_cfg.paths.splits_root) / namespace
    stage2_manifest = Path(stage2.paths.stage2_root) / namespace / "all_candidates.csv"
    real_split = str(learned.real_train_split)
    status = freeze_measure_inputs(out_dir / "measure_inputs.json", {
        "git_commit_hash": get_git_commit_hash(),
        "proxy": {key: str(learned.proxy[key]) for key in V1_PROXY},
        f"{real_split}_csv_sha256": _sha256(splits_dir / f"{real_split}.csv"),
        f"{TUNING_SPLIT}_csv_sha256": _sha256(splits_dir / f"{TUNING_SPLIT}.csv"),
        "all_candidates_sha256": _sha256(stage2_manifest),
    })
    print(f"measure inputs: {status}", flush=True)

    images_root = Path(stage1.paths.images_dir) / namespace
    real_records = _split_records(splits_dir, images_root, real_split)
    tuning_records = _split_records(splits_dir, images_root, TUNING_SPLIT)
    frame, _, _ = load_candidate_pool(namespace, stage3, Path(stage2.paths.stage2_root), verbose=False)
    manifest = pd.read_csv(stage2_manifest)
    candidates = _candidate_records(frame, manifest)
    subsets = {row["subset_id"]: row for row in _read_jsonl(v1_dir / "utility_subsets.jsonl")}

    runs_path = out_dir / "noise_floor_runs.jsonl"
    done = {(r["subset_id"], r["seed"]) for r in _read_jsonl(runs_path)} if runs_path.is_file() else set()
    # Seed-major order: an interrupted pod still leaves every subset at the same number of seeds,
    # which is what the balanced ANOVA needs.
    jobs = [(seed, sid) for seed in plan["seeds"] for sid in [BASELINE_ID, *plan["chosen_subsets"]]]
    pending = [(seed, sid) for seed, sid in jobs if (sid, seed) not in done]
    gpu = _gpu_name()
    print(f"{len(pending)} of {len(jobs)} proxy runs pending on {gpu}", flush=True)

    for position, (seed, sid) in enumerate(pending, start=1):
        synthetic = [] if sid == BASELINE_ID else [candidates[i] for i in subsets[sid]["image_ids"]]
        started = time.perf_counter()
        metrics = _measure(real_records + synthetic, tuning_records, learned.proxy, seed, device)
        seconds = time.perf_counter() - started
        _append_jsonl(runs_path, {
            "subset_id": sid,
            "seed": int(seed),
            "n_synthetic": len(synthetic),
            "metrics": {m: float(metrics[m]) for m in METRICS},
            "seconds": round(seconds, 1),
            "gpu": gpu,
            "git_commit_hash": get_git_commit_hash(),
        })
        print(f"  [{position}/{len(pending)}] {sid} seed={seed} bal.acc={metrics['balanced_accuracy']:.4f} "
              f"auroc={metrics['macro_auroc_ovr']:.4f} ({seconds:.0f}s)", flush=True)
    return {"runs_path": str(runs_path), "runs": len(jobs)}


def run_analyze(v1_dir: Path, out_dir: Path) -> dict:
    plan = json.loads((out_dir / "noise_floor_plan.json").read_text(encoding="utf-8"))
    rows = _read_jsonl(out_dir / "noise_floor_runs.jsonl")
    runs = pd.DataFrame([{"subset_id": r["subset_id"], "seed": r["seed"], **r["metrics"]} for r in rows])
    chosen, sizes = plan["chosen_subsets"], np.asarray(plan["chosen_subset_sizes"])

    first = runs.pivot_table(index="subset_id", columns="seed", values=V1_METRIC)
    complete_seeds = [s for s in plan["seeds"] if s in first.columns
                      and first.reindex([BASELINE_ID, *chosen])[s].notna().all()]
    if len(complete_seeds) < 2:
        raise NoiseFloorError(f"not enough finished runs to analyse (complete seeds: {complete_seeds})")

    per_metric, paired_per_metric, per_subset = {}, {}, {}
    for metric in plan["metrics"]:
        wide = runs.pivot_table(index="subset_id", columns="seed", values=metric)
        baseline = wide.loc[BASELINE_ID, complete_seeds].to_numpy()
        augmented = wide.loc[chosen, complete_seeds].to_numpy()
        # v1-style: every subset minus one fixed baseline, so its noise is the augmented run's alone.
        # paired: subset minus the baseline trained under the SAME seed, cancelling any shared noise.
        per_metric[metric] = analyse_table(augmented, sizes)
        per_metric[metric]["baseline_sd_across_seeds"] = float(np.std(baseline, ddof=1))
        paired_per_metric[metric] = analyse_table(augmented - baseline[None, :], sizes)
        per_subset[metric] = {sid: [float(x) for x in wide.loc[sid, complete_seeds]] for sid in [BASELINE_ID, *chosen]}

    v1 = pd.DataFrame(_read_jsonl(v1_dir / "utility_results.jsonl")).set_index("subset_id")
    same_seed = {}
    if 42 in complete_seeds:
        rerun = first.loc[chosen, 42]
        original = v1.loc[chosen, "augmented_balanced_accuracy"]
        same_seed = {"max_abs_difference": float((rerun - original).abs().max()),
                     "mean_abs_difference": float((rerun - original).abs().mean())}

    seconds = [r["seconds"] for r in rows if "seconds" in r]
    report = {
        "diagnostic": "proxy_noise_floor",
        "post_hoc": True,
        "complete_seeds": complete_seeds,
        "chosen_subsets": chosen,
        "chosen_subset_sizes": sizes.tolist(),
        "per_subset": per_subset,
        "v1_style_label": per_metric,
        "paired_label": paired_per_metric,
        "v2_metric_rule": choose_v2_metric(per_metric),
        "seed42_rerun_vs_v1": same_seed,
        "timing": {"runs": len(seconds), "mean_seconds": float(np.mean(seconds)) if seconds else None,
                   "gpu": sorted({r.get("gpu", "unknown") for r in rows})},
        "decision_rule": {"reliability_target": RELIABILITY_TARGET, "max_affordable_repeats": MAX_AFFORDABLE_REPEATS},
        "interpretation": {
            "correlation_ceiling": "approximate reference, sqrt(reliability); bounds Pearson only under the "
                                   "classical noise model, not a strict upper bound on Spearman",
            "same_size_only": "sensitivity analysis with 3-5 subsets per size; indicative, not an independent estimate",
            "v2_metric_rule": "a recommendation for Stage 3 v2 only; nothing in ham-final-v1 is changed",
        },
    }
    (out_dir / "noise_floor_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"complete seeds: {complete_seeds}")
    for label, blocks in (("v1-style label", per_metric), ("paired label", paired_per_metric)):
        print(f"\n{label}")
        for metric, b in blocks.items():
            same = b.get("same_size_only", {})
            same_icc = f"{same['icc_single_run']:.2f}" if "icc_single_run" in same else "n/a"
            print(f"  {metric:<18} within SD {b['within_sd']:.4f}  between SD {b['between_sd']:.4f}  "
                  f"ICC {b['icc_single_run']:.2f} [{b['bootstrap']['icc_ci95'][0]:.2f}, {b['bootstrap']['icc_ci95'][1]:.2f}]  "
                  f"same-size ICC (sensitivity) {same_icc}  ceiling~(1 run) {b['correlation_ceiling_by_repeats']['1']:.2f}  "
                  f"repeats {b['repeats_needed']}  -> {b['verdict']}")
    print(f"\nv2 metric rule: {report['v2_metric_rule']}")
    if same_seed:
        print(f"seed-42 rerun vs v1: max |diff| {same_seed['max_abs_difference']:.4f}")
    if seconds:
        print(f"mean {np.mean(seconds):.0f}s per proxy run on {report['timing']['gpu']}")
    return {"report": str(out_dir / "noise_floor_report.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--phase", required=True, choices=["plan", "measure", "analyze"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--v1-dir", default=None, help="v1 learned/ directory (default: from the Stage 3 config)")
    parser.add_argument("--out-dir", default=None, help="default: outputs/ham10000/followup/<ns>/proxy_noise_floor")
    args = parser.parse_args()

    if args.v1_dir and args.out_dir:
        v1_dir, out_dir = Path(args.v1_dir), Path(args.out_dir)
    else:
        from scripts.utils.config import load_named_config

        stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
        base = Path(stage3.paths.outputs_dir)
        v1_dir = Path(args.v1_dir) if args.v1_dir else base / args.namespace / "learned"
        out_dir = Path(args.out_dir) if args.out_dir else base.parent / "followup" / args.namespace / "proxy_noise_floor"

    if args.phase == "plan":
        result = run_plan(v1_dir, out_dir)
    elif args.phase == "measure":
        result = run_measure(args.namespace, v1_dir, out_dir, args.device)
    else:
        result = run_analyze(v1_dir, out_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
