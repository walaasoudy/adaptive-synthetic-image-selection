"""Does a larger evaluation set reduce sigma_e? Measured, not inferred by subtraction.

Pre-registered in `docs/ham10000_noise_decomposition_prereg.md`.

Every reliability figure in sections 15 and 16 turns on ICC = sigma_s^2 / (sigma_s^2 + sigma_e^2).
sigma_s has been measured repeatedly; sigma_e never interrogated. If a larger evaluation set would
shrink it, the next question is about splits and datasets. If it would not, repeated training is the
only lever and the repeat counts already measured stand.

An earlier design tried to obtain sigma_train as sqrt(sigma_total^2 - sigma_eval^2), with sigma_eval
from a per-model bootstrap. It was withdrawn: every run is scored on the SAME fixed 1,377 images, so
the variance across seeds contains only the part of the evaluation error that models disagree on,
while a per-model bootstrap measures the whole of it. The subtraction therefore overstates the
evaluation component by an unknown amount. See prereg section 3.1.

What runs instead: score the same models on nested evaluation subsets of 12.5%, 25%, 50% and 100%,
and watch sigma_e as the evaluation sample grows. Falling curve -> a larger evaluation set could
help. Flat curve -> it could not. No subtraction, no independence assumption, four measured points.

The measurement path is the frozen one. `_measure` in ham10000_03_build_utility_subsets.py builds a
TrainingBudget, calls train_classifier, then predict_probabilities, then full_metric_suite. This
calls those same four things with those same arguments and differs in exactly one respect: it keeps
the probability matrix instead of discarding it. `--phase verify` proves that by reproducing
`_measure`'s own output at the same seed, and the grid refuses to run until it has.

Phases:
  --phase plan      laptop, seconds. Freezes the two subsets and the evaluation-curve settings.
  --phase verify    GPU, about 30 s. Two runs. Must match `_measure` exactly or nothing proceeds.
  --phase measure   GPU, about 6 minutes. 24 runs. Resumable.
  --phase analyze   laptop, a few minutes. Refuses an incomplete grid, then builds the curve.
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
    freeze_measure_inputs, variant_proxy,
)

PROXY_VARIANT = "v1"
PRIMARY_METRIC = "macro_auroc_ovr"
METRICS = ("balanced_accuracy", "macro_auroc_ovr", "macro_f1", "accuracy")

SMALL_SIZE = 150
SMALL_ARM = "random_150_decomp"
LARGE_ARM = "random_616_00"            # reused unchanged from the selection-headroom plan
DRAW_SEED = 20260925                   # frozen before any run: which synthetic images
# 42..53, 12 per subset -> 24 runs. Twelve, not thirty: the answer now comes from a trend across
# four evaluation sizes rather than from a difference between two close numbers, and twelve is what
# section 16 used for its random arms, so the two are directly comparable. Seeds 42-51 of the 616
# arm were already run there, so those ten are a free reproducibility check.
SEEDS = tuple(range(42, 54))

# The evaluation-size curve, frozen before any run.
EVAL_FRACTIONS = (0.125, 0.25, 0.5, 1.0)
EVAL_DRAW_SEED = 20260926
N_EVAL_DRAWS = 20

N_BOOTSTRAP = 2000                     # descriptive only; not an input to any subtraction
BOOTSTRAP_SEED = 0
PROJECTION_K = (2, 4)
VERIFY_SEED = 42


class DecompositionError(RuntimeError):
    """Raised instead of continuing on an input or a grid that cannot carry the conclusion."""


# ===================================================================================================
# The measurement, which must be the frozen one
# ===================================================================================================


def measure_keeping_probabilities(train_records, tuning_records, proxy_cfg, seed: int,
                                  device: str | None):
    """Line for line `_measure`, except that the probability matrix is returned as well.

    Kept deliberately parallel rather than clever: any divergence from the frozen recipe would make
    every number produced here incomparable with sections 15 and 16. `--phase verify` checks the
    equality rather than trusting this comment.
    """
    from scripts.utils.classifier import TrainingBudget
    from scripts.utils.ham10000_classifier import (
        predict_probabilities, train_classifier, true_class_indices,
    )
    from scripts.utils.ham10000_metrics import full_metric_suite

    budget = TrainingBudget(
        max_steps=int(proxy_cfg.max_steps),
        batch_size=int(proxy_cfg.batch_size),
        learning_rate=float(proxy_cfg.learning_rate),
        weight_decay=float(proxy_cfg.weight_decay),
        seed=int(seed),
        eval_every_n_steps=10**9,
    )
    model, _ = train_classifier(
        train_records,
        budget,
        float(proxy_cfg.dropout_p),
        str(proxy_cfg.pretrained_source),
        int(proxy_cfg.resolution),
        device=device,
        progress_desc="proxy",
    )
    probabilities = predict_probabilities(model, tuning_records, int(proxy_cfg.resolution),
                                          device=device)
    # Everything below this line is retention, not computation: full_metric_suite is called on
    # exactly the matrix `_measure` calls it on, and the extras are derived from that same matrix.
    metrics = full_metric_suite(probabilities, true_class_indices(tuning_records))
    probabilities = np.asarray(probabilities)
    return {
        "metrics": metrics,
        "probabilities": probabilities,
        "predictions": probabilities.argmax(axis=1),   # the decision the model actually makes
        "y_true": np.asarray(true_class_indices(tuning_records)),
    }


# ===================================================================================================
# The evaluation-size curve
# ===================================================================================================


def nested_eval_indices(y_true: np.ndarray, fractions=EVAL_FRACTIONS, n_draws: int = N_EVAL_DRAWS,
                        seed: int = EVAL_DRAW_SEED) -> dict:
    """Nested, class-proportional evaluation subsets: 12.5% inside 25% inside 50% inside 100%.

    Each class is permuted once per draw, and fraction f takes the first round(f * n_c) of it. The
    nesting is a property of the construction, not something checked afterwards, and the class
    proportions of `asism_tuning_heldout` are preserved at every size because those counts are fixed
    by the split rather than drawn at random: the smaller sets should differ from the full one in
    size alone.

    At f = 1.0 every draw is the whole evaluation set, so the draws collapse to one, as they should.
    """
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(y_true == c) for c in np.unique(y_true)]
    out: dict[float, list[np.ndarray]] = {float(f): [] for f in fractions}
    for _ in range(n_draws):
        permuted = [rng.permutation(idx) for idx in by_class]
        for f in fractions:
            take = [p[:max(1, int(round(float(f) * len(p))))] for p in permuted]
            out[float(f)].append(np.sort(np.concatenate(take)))
    return out


def sigma_e_curve(points_by_seed: dict, y_true: np.ndarray, probabilities_by_seed: dict,
                  metric: str = PRIMARY_METRIC) -> dict:
    """sigma_e at each evaluation fraction: SD across seeds, on each nested subset.

    `points_by_seed` is only carried through for the record; the curve is recomputed from the stored
    probability matrices so that the f = 1.0 entry is derived the same way as the others.
    """
    from scripts.utils.ham10000_metrics import full_metric_suite

    seeds = sorted(probabilities_by_seed)
    draws = nested_eval_indices(y_true)
    curve = {}
    for f, index_sets in draws.items():
        per_draw = []
        for pick in index_sets:
            values = [float(full_metric_suite(probabilities_by_seed[s][pick], y_true[pick])[metric])
                      for s in seeds]
            per_draw.append(float(np.std(values, ddof=1)))
        per_draw = np.asarray(per_draw)
        curve[f] = {
            "n_images": int(len(index_sets[0])),
            "sigma_e_mean": float(per_draw.mean()),
            "sigma_e_p05_p95": [float(np.percentile(per_draw, 5)),
                                float(np.percentile(per_draw, 95))],
            "n_draws": int(len(per_draw)),
            "per_draw": [float(v) for v in per_draw],
        }
    return curve


def fit_inverse_f(curve: dict) -> dict:
    """Least squares of sigma_e^2 against 1/f, i.e. sigma_e^2(f) = a + b/f.

    SECONDARY. The measured curve is the result; this is an extrapolation from four points. `a` is
    the part that does not shrink with evaluation size and `b/f` the part that does, but they are
    fitted, not separated, and a fit that does not describe the points is reported as such.
    """
    fractions = sorted(curve)
    x = np.array([1.0 / f for f in fractions])
    y = np.array([curve[f]["sigma_e_mean"] ** 2 for f in fractions])
    design = np.column_stack([np.ones_like(x), x])
    (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ np.array([a, b])
    residual = y - predicted
    denominator = float(((y - y.mean()) ** 2).sum())
    return {
        "a_floor_variance": float(a),
        "b_per_fraction": float(b),
        "r_squared": float(1 - (residual ** 2).sum() / denominator) if denominator > 0 else None,
        "max_abs_residual_in_sigma": float(np.max(np.abs(np.sqrt(np.maximum(y, 0))
                                                         - np.sqrt(np.maximum(predicted, 0))))),
        "floor_sigma": float(np.sqrt(max(a, 0.0))),
    }


def project(fit: dict, sigma_s: float, k_values=PROJECTION_K) -> dict:
    """What the fitted curve implies for evaluation sets k times the current one.

    Arithmetic, not a promise, and built on the secondary fit rather than on measured points.
    """
    out = {}
    for k in (1, *k_values):
        variance = max(fit["a_floor_variance"] + fit["b_per_fraction"] / k, 0.0)
        s_e = float(np.sqrt(variance))
        icc = sigma_s**2 / (sigma_s**2 + s_e**2) if (sigma_s or s_e) else 0.0
        need = 0.8 * (1 - icc) / (0.2 * icc) if icc > 0 else None
        out[str(k)] = {
            "sigma_e": s_e,
            "icc": float(icc),
            "repeats_for_080": (int(np.ceil(need)) if need is not None and need < 1e6 else None),
        }
    return out


def stratified_bootstrap_sd(probabilities: np.ndarray, y_true: np.ndarray,
                            metric: str = PRIMARY_METRIC, n: int = N_BOOTSTRAP,
                            seed: int = BOOTSTRAP_SEED) -> dict:
    """DESCRIPTIVE ONLY: how precisely 1,377 images pin down ONE model's macro AUROC.

    This is not an input to any subtraction and no component is derived from it. It measures the
    total evaluation error of a single model, most of which is common to all the models here because
    they are scored on the same images, and therefore cancels in the seed-to-seed spread.

    Stratified within class for the same reason the nested subsets are: the class counts of
    `asism_tuning_heldout` are fixed by the split, not drawn at random.
    """
    from scripts.utils.ham10000_metrics import full_metric_suite

    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(y_true == c) for c in np.unique(y_true)]
    values = []
    for _ in range(n):
        pick = np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in by_class])
        values.append(float(full_metric_suite(probabilities[pick], y_true[pick])[metric]))
    values = np.asarray(values)
    return {
        "sd": float(values.std(ddof=1)),
        "mean": float(values.mean()),
        "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
        "n_bootstrap": int(n),
    }


def _complete_table(rows: list[dict], arms: list[str], seeds: list[int]) -> dict:
    """Refuse a partial or duplicated grid rather than analysing the complete part of it."""
    cells: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        cells.setdefault((row["subset_id"], int(row["seed"])), []).append(row)
    want = [(a, s) for a in arms for s in seeds]
    missing = [k for k in want if k not in cells]
    duplicated = [k for k, v in cells.items() if len(v) > 1]
    if missing:
        raise DecompositionError(
            f"the grid is incomplete: {len(missing)} of {len(want)} cells missing "
            f"(e.g. {missing[:3]}). Finish --phase measure; nothing partial is analysed.")
    if duplicated:
        raise DecompositionError(f"duplicated cells: {duplicated[:5]}.")
    return {k: v[0] for k, v in cells.items()}


# ===================================================================================================
# Phases
# ===================================================================================================


def _load_pool(namespace: str):
    from scripts.asism.ham10000_ranking import load_candidate_pool
    from scripts.utils.config import load_named_config

    stage2 = load_named_config("ham10000_stage2.yaml", "ham_stage2")
    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    frame, _, _ = load_candidate_pool(namespace, stage3, Path(stage2.paths.stage2_root),
                                      verbose=False)
    return sorted(frame["image_id"].astype(str).tolist())


def run_plan(namespace: str, headroom_plan: Path, out_dir: Path) -> dict:
    from scripts.utils.manifest import get_git_commit_hash

    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "decomposition_plan.json"
    if plan_path.is_file():
        raise DecompositionError(f"a frozen plan already exists at {plan_path}; delete the whole "
                                 "output directory to start this experiment again.")
    if not headroom_plan.is_file():
        raise DecompositionError(f"no selection-headroom plan at {headroom_plan}; {LARGE_ARM} is "
                                 "reused from it unchanged and cannot be reconstructed here.")

    pool = _load_pool(namespace)
    prior = json.loads(headroom_plan.read_text(encoding="utf-8"))
    if LARGE_ARM not in prior["arms"]:
        raise DecompositionError(f"{LARGE_ARM} is not in {headroom_plan}")
    large = sorted(prior["arms"][LARGE_ARM])

    rng = np.random.default_rng(DRAW_SEED)
    small = sorted(rng.choice(pool, size=SMALL_SIZE, replace=False).tolist())

    plan = {
        "experiment": "evaluation_size_curve",
        "pre_registration": "docs/ham10000_noise_decomposition_prereg.md",
        "supersedes": "the subtraction design registered in commit 964f0e5 (see prereg 3.1)",
        "namespace": namespace,
        "proxy": {**variant_proxy(PROXY_VARIANT), "architecture": V1_ARCHITECTURE},
        "budget_protocol": "optimizer steps, not epochs (docs/stages2_to_5_plan.md:454)",
        "primary_metric": PRIMARY_METRIC,
        "seeds": list(SEEDS),
        "arms": {SMALL_ARM: small, LARGE_ARM: large},
        "arm_sizes": {SMALL_ARM: len(small), LARGE_ARM: len(large)},
        "draw_seed": DRAW_SEED,
        "large_arm_source": str(headroom_plan),
        "large_arm_source_sha256": _sha256(headroom_plan),
        "evaluation_curve": {
            "fractions": list(EVAL_FRACTIONS),
            "n_draws": N_EVAL_DRAWS,
            "draw_seed": EVAL_DRAW_SEED,
            "nested": "each class permuted once per draw; fraction f takes the first round(f*n_c), "
                      "so 12.5% is inside 25% is inside 50% is inside 100%",
            "class_proportional": True,
        },
        "bootstrap": {"n": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED, "stratified_within_class": True,
                      "role": "descriptive only; not an input to any subtraction"},
        "projection_k": list(PROJECTION_K),
        "verify_seed": VERIFY_SEED,
        "no_threshold": "this experiment estimates a curve and tests no hypothesis; no pass or fail "
                        "is declared, and the procedure is what is frozen",
        "what_this_cannot_do": [
            "it changes no split and evaluates only on asism_tuning_heldout and nested subsets of it",
            "it does not propose a dataset change; it measures whether one could help",
            "it does not touch the Neural learned utility ranking",
            "it produces no selection and no C2, and is not a Stage 5 result",
            "it does not claim to have separated sigma_train from sigma_eval",
        ],
        "git_commit_hash": get_git_commit_hash(),
    }
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    runs = len(plan["arms"]) * len(SEEDS)
    print(f"Frozen: {len(plan['arms'])} subsets x {len(SEEDS)} seeds = {runs} runs -> {plan_path}",
          flush=True)
    for arm, ids in plan["arms"].items():
        print(f"  {arm:<22} {len(ids):>4} synthetic", flush=True)
    print(f"  evaluation fractions {list(EVAL_FRACTIONS)} x {N_EVAL_DRAWS} nested draws "
          f"(seed {EVAL_DRAW_SEED})", flush=True)
    return {"plan_path": str(plan_path), "runs_planned": runs}


def _prepare(namespace: str, out_dir: Path):
    """Everything both verify and measure need, including the guards."""
    import pandas as pd

    from scripts.asism.ham10000_03_build_utility_subsets import _candidate_records, _split_records
    from scripts.asism.ham10000_ranking import load_candidate_pool
    from scripts.utils.config import load_named_config
    from scripts.utils.manifest import get_git_commit_hash

    plan_path = out_dir / "decomposition_plan.json"
    if not plan_path.is_file():
        raise DecompositionError(f"no frozen plan at {plan_path}; run --phase plan first.")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    stage1 = load_named_config("ham10000_stage1.yaml", "ham_stage1")
    stage2 = load_named_config("ham10000_stage2.yaml", "ham_stage2")
    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    splits_cfg = load_named_config("splits_ham10000.yaml", "ham_splits")
    learned = stage3.learned_asism
    real_split = str(learned.real_train_split)
    if FORBIDDEN_SPLIT in {real_split, TUNING_SPLIT}:
        raise DecompositionError("the protected split cannot be used to measure utility")
    check_proxy_config(learned.proxy)
    proxy = variant_proxy(PROXY_VARIANT)
    if plan["proxy"] != {**proxy, "architecture": V1_ARCHITECTURE}:
        raise DecompositionError("the frozen plan records a different proxy than this code builds")

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
    frame, _, _ = load_candidate_pool(namespace, stage3, Path(stage2.paths.stage2_root),
                                      verbose=False)
    candidates = _candidate_records(frame, pd.read_csv(stage2_manifest))
    return plan, proxy, real_records, tuning_records, candidates


def run_verify(namespace: str, out_dir: Path, device: str | None) -> dict:
    """Prove the local measurement path is the frozen one before any of the grid runs."""
    from scripts.asism.ham10000_03_build_utility_subsets import _measure

    plan, proxy, real_records, tuning_records, candidates = _prepare(namespace, out_dir)
    arm = SMALL_ARM
    train = real_records + [candidates[i] for i in plan["arms"][arm]]
    cfg = SimpleNamespace(**proxy)

    print(f"verifying against _measure on {arm}, seed {VERIFY_SEED} ...", flush=True)
    frozen = _measure(train, tuning_records, cfg, VERIFY_SEED, device)
    local = measure_keeping_probabilities(train, tuning_records, cfg, VERIFY_SEED, device)["metrics"]

    differences = {m: (float(frozen[m]), float(local[m])) for m in METRICS
                   if float(frozen[m]) != float(local[m])}
    result = {
        "arm": arm, "seed": VERIFY_SEED,
        "frozen": {m: float(frozen[m]) for m in METRICS},
        "local": {m: float(local[m]) for m in METRICS},
        "identical": not differences,
        "differences": differences,
    }
    (out_dir / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if differences:
        raise DecompositionError(
            f"the local measurement path does NOT reproduce _measure: {differences}. "
            "Nothing is measured until it does.")
    print("  identical on all four metrics. The measurement path is the frozen one.", flush=True)
    for m in METRICS:
        print(f"    {m:<20} {float(frozen[m]):.10f}", flush=True)
    return result


def run_measure(namespace: str, out_dir: Path, device: str | None) -> dict:
    verification = out_dir / "verification.json"
    if not verification.is_file():
        raise DecompositionError("run --phase verify first: the grid does not run until the local "
                                 "measurement path has been shown to reproduce _measure exactly.")
    if not json.loads(verification.read_text(encoding="utf-8"))["identical"]:
        raise DecompositionError("verification on record did not pass; nothing is measured.")

    plan, proxy, real_records, tuning_records, candidates = _prepare(namespace, out_dir)
    runs_path = out_dir / "decomposition_runs.jsonl"
    probs_dir = out_dir / "probabilities"
    probs_dir.mkdir(parents=True, exist_ok=True)

    done = ({(r["subset_id"], int(r["seed"])) for r in _read_jsonl(runs_path)}
            if runs_path.is_file() else set())
    arms = list(plan["arms"])
    jobs = [(s, a) for s in plan["seeds"] for a in arms]      # seed-major, so a stop stays balanced
    pending = [(s, a) for s, a in jobs if (a, s) not in done]
    gpu = _gpu_name()
    print(f"{len(pending)} of {len(jobs)} proxy runs pending on {gpu}", flush=True)

    steps, batch = int(proxy["max_steps"]), int(proxy["batch_size"])
    for position, (seed, arm) in enumerate(pending, start=1):
        train = real_records + [candidates[i] for i in plan["arms"][arm]]
        started = time.perf_counter()
        kept = measure_keeping_probabilities(train, tuning_records, SimpleNamespace(**proxy),
                                             seed, device)
        seconds = time.perf_counter() - started
        metrics, y_true = kept["metrics"], kept["y_true"]
        stem = f"{arm}_seed{seed}"
        np.savez_compressed(probs_dir / f"{stem}.npz",
                            probabilities=kept["probabilities"].astype(np.float32),
                            predictions=kept["predictions"].astype(np.int16),
                            y_true=y_true.astype(np.int16))
        _append_jsonl(runs_path, {
            "subset_id": arm,
            "seed": int(seed),
            "n_synthetic": len(plan["arms"][arm]),
            "metrics": {m: float(metrics[m]) for m in METRICS},
            "probabilities_file": f"probabilities/{stem}.npz",
            "n_eval_images": int(len(y_true)),
            "effective_dataset_size": len(train),
            "optimizer_steps": steps,
            "epochs": round(steps * batch / len(train), 3),
            "seconds": round(seconds, 1),
            "gpu": gpu,
        })
        print(f"  [{position}/{len(pending)}] {arm} seed={seed} "
              f"auroc={metrics[PRIMARY_METRIC]:.4f} ({seconds:.0f}s)", flush=True)
    return {"runs_path": str(runs_path), "completed": len(jobs)}


def run_analyze(out_dir: Path, sigma_s_small: float | None, sigma_s_large: float) -> dict:
    plan = json.loads((out_dir / "decomposition_plan.json").read_text(encoding="utf-8"))
    rows = _read_jsonl(out_dir / "decomposition_runs.jsonl")
    arms, seeds = list(plan["arms"]), plan["seeds"]
    cells = _complete_table(rows, arms, seeds)
    print(f"complete grid: {len(arms)} subsets x {len(seeds)} seeds = {len(cells)} runs\n")

    report = {"experiment": "evaluation_size_curve", "primary_metric": PRIMARY_METRIC, "arms": {}}
    sigma_s_for = {SMALL_ARM: sigma_s_small, LARGE_ARM: sigma_s_large}
    for arm in arms:
        probabilities_by_seed, points, y_true, bootstrap_sds = {}, {}, None, {}
        for seed in seeds:
            row = cells[(arm, seed)]
            blob = np.load(out_dir / row["probabilities_file"])
            probabilities_by_seed[seed] = blob["probabilities"].astype(np.float64)
            y_true = blob["y_true"].astype(int)
            points[str(seed)] = row["metrics"][PRIMARY_METRIC]
            bootstrap_sds[str(seed)] = stratified_bootstrap_sd(probabilities_by_seed[seed],
                                                               y_true)["sd"]

        curve = sigma_e_curve(points, y_true, probabilities_by_seed)
        fit = fit_inverse_f(curve)
        entry = {
            "size": plan["arm_sizes"][arm],
            "points": points,
            "curve": {str(f): v for f, v in curve.items()},
            "fit_secondary": fit,
            "single_model_bootstrap_sd_descriptive": bootstrap_sds,
        }
        if sigma_s_for[arm] is not None:
            entry["projection_secondary"] = project(fit, sigma_s_for[arm])
            entry["sigma_s_used"] = sigma_s_for[arm]
        report["arms"][arm] = entry

    (out_dir / "decomposition_report.json").write_text(json.dumps(report, indent=2),
                                                      encoding="utf-8")

    print("PRIMARY RESULT: sigma_e measured at four evaluation sizes\n")
    for arm in arms:
        entry = report["arms"][arm]
        print(f"{arm}  (training subset {entry['size']} synthetic, {len(seeds)} seeds)")
        print(f"  {'fraction':>9}{'images':>8}{'sigma_e':>10}{'5th-95th across draws':>26}")
        for f in sorted(curve):
            v = entry["curve"][str(f)]
            span = f"[{v['sigma_e_p05_p95'][0]:.4f}, {v['sigma_e_p05_p95'][1]:.4f}]"
            print(f"  {f:>9.3f}{v['n_images']:>8}{v['sigma_e_mean']:>10.4f}{span:>26}")
        mean = entry["single_model_bootstrap_sd_descriptive"]
        print(f"  descriptive: one model's macro AUROC on 1,377 images is pinned to "
              f"+-{np.mean(list(mean.values())):.4f} (bootstrap SD, not a component)")
        print()

    print("SECONDARY (extrapolation from four points, labelled as such):")
    for arm in arms:
        entry = report["arms"][arm]
        fit = entry["fit_secondary"]
        # R^2 is undefined when the four points are identical, which is itself the flat-curve
        # answer: a larger evaluation set would not help. That case must read, not crash.
        if fit["r_squared"] is None:
            r2, quality = "n/a", "the curve is flat; a larger evaluation set would not help"
        elif fit["r_squared"] > 0.9:
            r2, quality = f"{fit['r_squared']:.3f}", "describes the points"
        else:
            r2 = f"{fit['r_squared']:.3f}"
            quality = "does NOT describe the points; the extrapolation is not used"
        print(f"\n  {arm}: sigma_e^2 = {fit['a_floor_variance']:.3e} + "
              f"{fit['b_per_fraction']:.3e}/f    R^2 {r2}  -> {quality}")
        print(f"    floor as an evaluation set grows without limit: sigma_e -> "
              f"{fit['floor_sigma']:.4f}")
        if "projection_secondary" not in entry:
            continue
        print(f"    at sigma_s = {entry['sigma_s_used']:.4f}:")
        print(f"      {'eval set':>10}{'sigma_e':>10}{'ICC':>9}{'repeats for 0.80':>19}")
        for k, v in entry["projection_secondary"].items():
            need = v["repeats_for_080"] if v["repeats_for_080"] is not None else "unreachable"
            print(f"      {('x' + k):>10}{v['sigma_e']:>10.4f}{v['icc']:>9.4f}{str(need):>19}")

    print("\nNo threshold was declared for this experiment and none is applied: it estimates a "
          "curve and tests no hypothesis. No claim is made that sigma_train and sigma_eval have "
          "been separated.")
    return {"report": str(out_dir / "decomposition_report.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--phase", required=True, choices=["plan", "verify", "measure", "analyze"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--headroom-plan", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--sigma-s-small", type=float, default=None,
                        help="sigma_s at size 150, once the small-budget diagnostic has measured "
                             "it. Omitted until then, and the projection is skipped.")
    parser.add_argument("--sigma-s-large", type=float, default=0.0010,
                        help="sigma_s at 616, measured in section 16")
    args = parser.parse_args()

    from scripts.utils.config import load_named_config

    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    followup = Path(stage3.paths.outputs_dir).parent / "followup" / args.namespace
    out_dir = Path(args.out_dir) if args.out_dir else followup / "noise_decomposition"
    headroom = (Path(args.headroom_plan) if args.headroom_plan
                else followup / "selection_headroom" / "headroom_plan.json")

    if args.phase == "plan":
        result = run_plan(args.namespace, headroom, out_dir)
    elif args.phase == "verify":
        result = run_verify(args.namespace, out_dir, args.device)
    elif args.phase == "measure":
        result = run_measure(args.namespace, out_dir, args.device)
    else:
        result = run_analyze(out_dir, args.sigma_s_small, args.sigma_s_large)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
