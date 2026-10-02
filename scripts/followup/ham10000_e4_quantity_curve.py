"""E4: can the marginal utility of adding synthetic images be resolved above the noise?

Design and decision rule: docs/ham10000_e4_quantity_design.md, approved by Walaa on 2026-10-02
(D1-D7 and section 4) before this code existed. Every constant below is a transcription of that
document. Changing one after any run exists is changing the experiment.

  Recipe      the Stage 4 recipe, 512 px / 3,000 optimizer steps (D1), checked field by field
              against configs/ham10000_stage4.yaml before any run.
  Pool        the project's safe pool: the safety gate of load_candidate_pool, i.e. invalid IQA and
              near-duplicates of real images removed (D2).
  Sizes       0, 250, 500, 1000, 2000 and N = the whole safe pool (D3).
  Draws       two nested chains, each one permutation of the safe pool; size s is the first s images
              of the chain (D4).
  Seeds       real only: 42..51. Every other size: 2 chains x 42..46. 10 runs per size (D5).
  Metric      macro AUROC (one-vs-rest) on asism_tuning_heldout, per run (D6).
  Rule        GO / COARSE / NO from a Welch test on U(N) - U(0) and a Holm-corrected one-sided
              Welch test on each step beyond the first (section 4).

Every run trains on classifier_train plus the drawn synthetic images, and nothing else changes
between runs. final_eval_heldout and classifier_val are never read.

Phases:
  --phase plan      laptop, seconds. Freezes the pool, the chains and the 60 cells.
  --phase measure   GPU, about 9.6 hours. Resumable; a crashed cell is re-run as the same cell.
  --phase analyze   laptop, seconds. Refuses an incomplete grid, then applies the frozen rule once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np

from scripts.followup.ham10000_proxy_noise_floor import (
    FORBIDDEN_SPLIT, TUNING_SPLIT, V1_ARCHITECTURE, V1_PROXY,
    _append_jsonl, _gpu_name, _read_jsonl, _sha256, check_proxy_config,
    freeze_measure_inputs, variant_proxy,
)

EXPERIMENT = "e4_quantity_curve"
DESIGN_DOCUMENT = "docs/ham10000_e4_quantity_design.md"

# ---- D1: the Stage 4 recipe -----------------------------------------------------------------------
PROXY_VARIANT = "stage4size"
VALIDATION_SPLIT = "classifier_val"

# ---- D3: sizes. The last one is N, the size of the safe pool, known only once the pool is built ---
FIXED_SIZES = (0, 250, 500, 1000, 2000)

# ---- D4: two nested chains, frozen before any run --------------------------------------------------
CHAIN_DRAW_SEEDS = {1: 20261003, 2: 20261004}

# ---- D5: seeds -------------------------------------------------------------------------------------
BASELINE_SEEDS = tuple(range(42, 52))              # size 0: 42..51
CHAIN_SEEDS = tuple(range(42, 47))                 # every other size: 42..46 per chain
RUNS_PER_SIZE = 10

# ---- D6: metrics -----------------------------------------------------------------------------------
PRIMARY_METRIC = "macro_auroc_ovr"
METRICS = ("macro_auroc_ovr", "balanced_accuracy", "macro_f1", "accuracy")

# ---- section 4: the decision rule ------------------------------------------------------------------
OVERALL_CONFIDENCE = 0.95                          # two-sided interval on U(N) - U(0)
STEP_ALPHA = 0.05                                  # Holm familywise alpha over steps 2..5
TESTED_STEPS = (2, 3, 4, 5)                        # step 1 (0 -> 250) is reported, not tested
VERDICT_GO, VERDICT_COARSE, VERDICT_NO = "GO", "COARSE", "NO"


class E4Error(RuntimeError):
    """Raised instead of continuing on an input or a grid that cannot carry the conclusion."""


# ===================================================================================================
# Design, as pure functions
# ===================================================================================================


def sizes_for_pool(pool_size: int) -> tuple[int, ...]:
    """The six sizes of D3. The pool has to be larger than the largest fixed size, or the last
    step (2000 -> N) would be empty and the curve would not be the one that was approved."""
    if pool_size <= FIXED_SIZES[-1]:
        raise E4Error(f"the safe pool holds {pool_size} images; D3 needs more than {FIXED_SIZES[-1]}")
    return FIXED_SIZES + (int(pool_size),)


def build_chains(pool_ids: list[str]) -> dict[str, list[str]]:
    """One random permutation of the safe pool per chain (D4). Sorting first makes the chain depend
    only on the pool's membership and the frozen draw seed, never on the order a file was read in."""
    ordered = sorted(str(i) for i in pool_ids)
    if len(set(ordered)) != len(ordered):
        raise E4Error("the safe pool holds duplicated image ids")
    return {str(chain): [ordered[k] for k in np.random.default_rng(seed).permutation(len(ordered))]
            for chain, seed in CHAIN_DRAW_SEEDS.items()}


def cell_id(size: int, chain: int | None, seed: int) -> str:
    return f"s{size:04d}_base_seed{seed}" if size == 0 else f"s{size:04d}_c{chain}_seed{seed}"


def plan_cells(sizes: tuple[int, ...]) -> list[dict]:
    """The 60 cells of D5: 10 seeds at size 0, then 2 chains x 5 seeds at every other size."""
    cells = [{"cell_id": cell_id(0, None, s), "size": 0, "chain": None, "seed": s} for s in BASELINE_SEEDS]
    for size in sizes[1:]:
        for chain in CHAIN_DRAW_SEEDS:
            for seed in CHAIN_SEEDS:
                cells.append({"cell_id": cell_id(size, chain, seed), "size": int(size),
                              "chain": chain, "seed": seed})
    return cells


def subset_ids(cell: dict, chains: dict) -> list[str]:
    """The synthetic images of one cell: the first `size` images of its chain. Nested by
    construction, so each step adds images to the previous size and replaces none of them."""
    if cell["size"] == 0:
        return []
    return list(chains[str(cell["chain"])][: cell["size"]])


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def check_stage4_recipe(stage4_cfg, proxy: dict) -> None:
    """D1: the proxy is the Stage 4 recipe, field by field. Refuse on any difference."""
    expected = {
        "resolution": stage4_cfg.model.resolution,
        "max_steps": stage4_cfg.training.max_steps,
        "batch_size": stage4_cfg.training.batch_size,
        "learning_rate": stage4_cfg.training.learning_rate,
        "weight_decay": stage4_cfg.training.weight_decay,
        "dropout_p": stage4_cfg.model.dropout_p,
        "pretrained_source": stage4_cfg.model.pretrained_source,
    }
    differing = {}
    for key, value in expected.items():
        current = proxy[key]
        same = (str(current) == str(value) if isinstance(value, str)
                else np.isclose(float(current), float(value)))
        if not same:
            differing[key] = (current, value)
    if str(stage4_cfg.model.architecture) != V1_ARCHITECTURE:
        differing["architecture"] = (V1_ARCHITECTURE, str(stage4_cfg.model.architecture))
    # The proxy path (_measure) trains unweighted; Stage 4 must too, or they are different recipes.
    if str(stage4_cfg.training.class_weighting) != "none":
        differing["class_weighting"] = ("none", str(stage4_cfg.training.class_weighting))
    if differing:
        raise E4Error(f"the E4 proxy is not the Stage 4 recipe (proxy, stage4): {differing}; refusing")


# ===================================================================================================
# Statistics: section 4, exactly
# ===================================================================================================


def welch(a, b) -> dict:
    """b - a with Welch's SE and Welch-Satterthwaite df. Runs are independent: no pairing."""
    from scipy import stats

    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    diff = float(b.mean() - a.mean())
    se = float(np.sqrt(va / na + vb / nb))
    if se == 0.0:
        # Both groups constant: the difference is known exactly and the interval is a point.
        df, half = float(na + nb - 2), 0.0
        p_one = 0.0 if diff > 0 else 1.0
    else:
        df = float((va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)))
        half = float(stats.t.ppf(0.5 + OVERALL_CONFIDENCE / 2, df) * se)
        p_one = float(stats.t.sf(diff / se, df))
    return {
        "mean_a": float(a.mean()), "mean_b": float(b.mean()), "difference": diff,
        "se": se, "df": df, "ci95_two_sided": [diff - half, diff + half],
        "p_one_sided_greater": p_one, "n_a": int(na), "n_b": int(nb),
    }


def values_by_size(rows: list[dict], cells: list[dict], metric: str) -> dict[int, np.ndarray]:
    """Each size's 10 per-run values. Refuses a missing or a duplicated cell, and a row the plan
    does not know: no verdict is ever given on part of the grid."""
    known = {c["cell_id"]: c for c in cells}
    seen: dict[str, float] = {}
    for row in rows:
        cid = row["cell_id"]
        if cid not in known:
            raise E4Error(f"the runs file holds a cell the plan does not define: {cid}")
        if cid in seen:
            raise E4Error(f"duplicated cell {cid}: the runs file holds it twice")
        seen[cid] = float(row["metrics"][metric])
    missing = [cid for cid in known if cid not in seen]
    if missing:
        raise E4Error(f"the grid is incomplete: {len(missing)} of {len(known)} cells missing "
                      f"(e.g. {missing[:3]}). Finish --phase measure; nothing partial is analysed.")
    out: dict[int, list[float]] = {}
    for cid, cell in known.items():
        out.setdefault(int(cell["size"]), []).append(seen[cid])
    for size, values in out.items():
        if len(values) != RUNS_PER_SIZE:
            raise E4Error(f"size {size} holds {len(values)} runs; the design has {RUNS_PER_SIZE}")
    return {size: np.asarray(values, float) for size, values in sorted(out.items())}


def decide(values: dict[int, np.ndarray], sizes: tuple[int, ...]) -> dict:
    """Apply the section 4 rule once. Pure: the verdict depends only on these values."""
    from scripts.utils.metrics import holm_bonferroni

    if tuple(sorted(values)) != tuple(sizes):
        raise E4Error(f"the values cover sizes {sorted(values)}, the design has {list(sizes)}")

    overall = welch(values[sizes[0]], values[sizes[-1]])
    overall["passes"] = bool(overall["ci95_two_sided"][0] > 0)

    steps = {}
    for k in range(1, len(sizes)):
        entry = welch(values[sizes[k - 1]], values[sizes[k]])
        entry.update({"from_size": int(sizes[k - 1]), "to_size": int(sizes[k]),
                      "tested": k in TESTED_STEPS})
        steps[k] = entry

    holm = holm_bonferroni({f"step_{k}": steps[k]["p_one_sided_greater"] for k in TESTED_STEPS}, STEP_ALPHA)
    for k in TESTED_STEPS:
        adjusted = holm[f"step_{k}"]["adjusted_p_value"]
        steps[k]["holm_adjusted_p"] = adjusted
        steps[k]["resolved"] = bool(adjusted <= STEP_ALPHA)
    resolved = [k for k in TESTED_STEPS if steps[k]["resolved"]]

    if not overall["passes"]:
        verdict = VERDICT_NO
    elif resolved:
        verdict = VERDICT_GO
    else:
        verdict = VERDICT_COARSE

    return {
        "overall": overall,
        "steps": {str(k): v for k, v in steps.items()},
        "resolved_steps": resolved,
        "verdict": verdict,
        "rule": "GO: overall lower 95% bound > 0 and >= 1 of steps 2-5 Holm-resolved at 0.05 "
                "(one-sided Welch). COARSE: overall passes, no step resolved. NO: overall fails.",
    }


def describe_curve(values: dict[int, np.ndarray]) -> dict:
    return {str(size): {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "n": int(len(v))}
            for size, v in values.items()}


# ===================================================================================================
# Phases
# ===================================================================================================


def run_plan(namespace: str, out_dir: Path, pool_loader: Callable | None = None) -> dict:
    """Freeze the safe pool, both chains and every cell. Never re-drawn: a second plan refuses."""
    from scripts.utils.manifest import get_git_commit_hash

    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "e4_plan.json"
    if plan_path.is_file():
        raise E4Error(f"a frozen plan already exists at {plan_path}. It is never re-drawn: "
                      "delete the whole output directory to start this experiment again.")

    pool_ids, pool_report = (pool_loader or _load_safe_pool)(namespace)
    applied = pool_report.get("safety_checks_applied", {})
    if not (applied.get("reject_invalid_iqa") and applied.get("reject_near_duplicates")):
        raise E4Error(f"D2 needs both safety checks on; the pool was built with {applied}")
    sizes = sizes_for_pool(len(pool_ids))
    chains = build_chains(pool_ids)
    cells = plan_cells(sizes)

    plan = {
        "experiment": EXPERIMENT,
        "design_document": DESIGN_DOCUMENT,
        "namespace": namespace,
        "proxy_variant": PROXY_VARIANT,
        "proxy": {**variant_proxy(PROXY_VARIANT), "architecture": V1_ARCHITECTURE},
        "budget_protocol": "optimizer steps, not epochs (docs/stages2_to_5_plan.md:454)",
        "primary_metric": PRIMARY_METRIC,
        "metrics": list(METRICS),
        "safe_pool": {
            "n": len(pool_ids),
            "ids_sha256": ids_sha256(pool_ids),
            "safety_checks_applied": applied,
            "safety_gate_removals": pool_report.get("safety_gate_removals", {}),
            "candidates_before_safety_gate": pool_report.get("candidates_before_safety_gate"),
            "per_class_counts": pool_report.get("per_class_counts", {}),
        },
        "sizes": list(sizes),
        "chain_draw_seeds": {str(k): v for k, v in CHAIN_DRAW_SEEDS.items()},
        "baseline_seeds": list(BASELINE_SEEDS),
        "chain_seeds": list(CHAIN_SEEDS),
        "decision_rule": {
            "overall": f"two-sided {OVERALL_CONFIDENCE:.0%} Welch interval on U(N) - U(0); passes if lower > 0",
            "steps": f"one-sided Welch per step {list(TESTED_STEPS)}, Holm at familywise {STEP_ALPHA}",
            "verdicts": {VERDICT_GO: "overall passes and >= 1 step resolved",
                         VERDICT_COARSE: "overall passes and no step resolved",
                         VERDICT_NO: "overall does not pass"},
        },
        "attempts": "one. The rule is applied once to the complete grid; a crashed cell is re-run "
                    "as the same cell; no cell is added, replaced or dropped after any result.",
        "chains": chains,
        "cells": [{**c, "subset_sha256": ids_sha256(subset_ids(c, chains))} for c in cells],
        "git_commit_hash": get_git_commit_hash(),
    }
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Frozen: {len(cells)} runs, safe pool N = {len(pool_ids)}, sizes {list(sizes)} -> {plan_path}",
          flush=True)
    return {"plan_path": str(plan_path), "runs_planned": len(cells), "safe_pool_size": len(pool_ids)}


def _load_safe_pool(namespace: str) -> tuple[list[str], dict]:
    from scripts.asism.ham10000_ranking import load_candidate_pool
    from scripts.utils.config import load_named_config

    stage2 = load_named_config("ham10000_stage2.yaml", "ham_stage2")
    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    frame, _, report = load_candidate_pool(namespace, stage3, Path(stage2.paths.stage2_root), verbose=True)
    return sorted(frame["image_id"].astype(str).tolist()), report


def _load_plan(out_dir: Path) -> dict:
    plan_path = out_dir / "e4_plan.json"
    if not plan_path.is_file():
        raise E4Error(f"no frozen plan at {plan_path}; run --phase plan first.")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for cell in plan["cells"]:
        if ids_sha256(subset_ids(cell, plan["chains"])) != cell["subset_sha256"]:
            raise E4Error(f"cell {cell['cell_id']} no longer matches its frozen subset hash")
    return plan


def job_order(cells: list[dict]) -> list[dict]:
    """Seed-major, so an interrupted pod leaves every size at about the same depth."""
    return sorted(cells, key=lambda c: (c["seed"], c["size"], c["chain"] or 0))


def run_measure(namespace: str, out_dir: Path, device: str | None,
                measure_fn: Callable | None = None, inputs: dict | None = None) -> dict:
    """Train and score every pending cell. `measure_fn` and `inputs` exist for the CPU tests; on
    the pod both are None and the audited _measure path and the real records are used."""
    plan = _load_plan(out_dir)
    if inputs is None:
        inputs = _measure_inputs(namespace, out_dir, plan)
    real_records, tuning_records, candidates = inputs["real"], inputs["tuning"], inputs["candidates"]
    measure = measure_fn or _default_measure
    proxy = variant_proxy(PROXY_VARIANT)

    runs_path = out_dir / "e4_runs.jsonl"
    done = {r["cell_id"] for r in _read_jsonl(runs_path)} if runs_path.is_file() else set()
    pending = [c for c in job_order(plan["cells"]) if c["cell_id"] not in done]
    gpu = _gpu_name()
    print(f"{len(pending)} of {len(plan['cells'])} E4 runs pending on {gpu}", flush=True)

    from scripts.utils.manifest import get_git_commit_hash

    steps, batch = int(proxy["max_steps"]), int(proxy["batch_size"])
    for position, cell in enumerate(pending, start=1):
        ids = subset_ids(cell, plan["chains"])
        unknown = [i for i in ids if i not in candidates]
        if unknown:
            raise E4Error(f"cell {cell['cell_id']}: {len(unknown)} images are not candidates (e.g. {unknown[:3]})")
        train = real_records + [candidates[i] for i in ids]
        started = time.perf_counter()
        metrics = measure(train, tuning_records, SimpleNamespace(**proxy), int(cell["seed"]), device)
        seconds = time.perf_counter() - started
        _append_jsonl(runs_path, {
            "cell_id": cell["cell_id"], "size": cell["size"], "chain": cell["chain"],
            "seed": int(cell["seed"]), "n_synthetic": len(ids), "subset_sha256": ids_sha256(ids),
            "metrics": {m: float(metrics[m]) for m in METRICS},
            "per_class_recall": {label: float(v["recall"]) for label, v in metrics["per_class"].items()},
            "effective_dataset_size": len(train),
            "optimizer_steps": steps,
            "epochs": round(steps * batch / len(train), 3),
            "seconds": round(seconds, 1),
            "gpu": gpu,
            "git_commit_hash": get_git_commit_hash(),
        })
        print(f"  [{position}/{len(pending)}] {cell['cell_id']} n_syn={len(ids)} "
              f"auroc={metrics[PRIMARY_METRIC]:.4f} ({seconds:.0f}s)", flush=True)
    return {"runs_path": str(runs_path), "completed": len(plan["cells"])}


def _default_measure(train, tuning, proxy, seed, device):
    from scripts.asism.ham10000_03_build_utility_subsets import _measure

    return _measure(train, tuning, proxy, seed, device)


def _measure_inputs(namespace: str, out_dir: Path, plan: dict) -> dict:
    import pandas as pd

    from scripts.asism.ham10000_03_build_utility_subsets import _candidate_records, _split_records
    from scripts.asism.ham10000_ranking import load_candidate_pool
    from scripts.utils.config import load_named_config
    from scripts.utils.manifest import get_git_commit_hash

    stage1 = load_named_config("ham10000_stage1.yaml", "ham_stage1")
    stage2 = load_named_config("ham10000_stage2.yaml", "ham_stage2")
    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    stage4 = load_named_config("ham10000_stage4.yaml", "ham_stage4")
    splits_cfg = load_named_config("splits_ham10000.yaml", "ham_splits")
    learned = stage3.learned_asism
    real_split = str(learned.real_train_split)
    if {real_split, TUNING_SPLIT} & {FORBIDDEN_SPLIT, VALIDATION_SPLIT}:
        raise E4Error("E4 trains on classifier_train and measures on asism_tuning_heldout only")
    check_proxy_config(learned.proxy)          # the Stage 3 config itself is still v1's
    proxy = variant_proxy(PROXY_VARIANT)
    check_stage4_recipe(stage4, proxy)         # D1
    if plan["proxy"] != {**proxy, "architecture": V1_ARCHITECTURE}:
        raise E4Error("the frozen plan records a different proxy than this code builds")

    frame, _, report = load_candidate_pool(namespace, stage3, Path(stage2.paths.stage2_root), verbose=False)
    pool = sorted(frame["image_id"].astype(str).tolist())
    if ids_sha256(pool) != plan["safe_pool"]["ids_sha256"]:
        raise E4Error("the safe pool differs from the one frozen in the plan; refusing to mix pools")

    splits_dir = Path(splits_cfg.paths.splits_root) / namespace
    stage2_manifest = Path(stage2.paths.stage2_root) / namespace / "all_candidates.csv"
    status = freeze_measure_inputs(out_dir / "measure_inputs.json", {
        "git_commit_hash": get_git_commit_hash(),
        "proxy": {key: str(proxy[key]) for key in V1_PROXY},
        f"{real_split}_csv_sha256": _sha256(splits_dir / f"{real_split}.csv"),
        f"{TUNING_SPLIT}_csv_sha256": _sha256(splits_dir / f"{TUNING_SPLIT}.csv"),
        "all_candidates_sha256": _sha256(stage2_manifest),
        "safe_pool_ids_sha256": plan["safe_pool"]["ids_sha256"],
    })
    print(f"measure inputs: {status}", flush=True)

    images_root = Path(stage1.paths.images_dir) / namespace
    return {
        "real": _split_records(splits_dir, images_root, real_split),
        "tuning": _split_records(splits_dir, images_root, TUNING_SPLIT),
        "candidates": _candidate_records(frame, pd.read_csv(stage2_manifest)),
    }


def run_analyze(out_dir: Path) -> dict:
    plan = _load_plan(out_dir)
    rows = _read_jsonl(out_dir / "e4_runs.jsonl")
    frozen = {c["cell_id"]: c["subset_sha256"] for c in plan["cells"]}
    for row in rows:
        if row["cell_id"] in frozen and row.get("subset_sha256") != frozen[row["cell_id"]]:
            raise E4Error(f"run {row['cell_id']} trained on a different subset than the plan froze")
    sizes = tuple(plan["sizes"])
    cells = plan["cells"]

    primary = values_by_size(rows, cells, PRIMARY_METRIC)
    result = decide(primary, sizes)

    per_chain = {}
    for chain in plan["chain_draw_seeds"]:
        per_chain[chain] = {
            str(size): float(np.mean([r["metrics"][PRIMARY_METRIC] for r in rows
                                      if r["size"] == size and str(r["chain"]) == chain]))
            for size in sizes[1:]
        }
    secondary = {m: describe_curve(values_by_size(rows, cells, m)) for m in METRICS if m != PRIMARY_METRIC}
    labels = sorted(rows[0]["per_class_recall"]) if rows else []
    per_class = {label: {str(size): float(np.mean([r["per_class_recall"][label] for r in rows
                                                   if r["size"] == size]))
                         for size in sizes} for label in labels}

    report = {
        "experiment": EXPERIMENT,
        "design_document": DESIGN_DOCUMENT,
        "primary_metric": PRIMARY_METRIC,
        "sizes": list(sizes),
        "curve": describe_curve(primary),
        "real_only_baseline": describe_curve({0: primary[0]})["0"],
        **result,
        "reported_not_deciding": {
            "per_chain_curve": per_chain,
            "secondary_metrics": secondary,
            "per_class_recall": per_class,
        },
        "epochs_by_size": {str(size): sorted({r["epochs"] for r in rows if r["size"] == size})
                           for size in sizes},
    }
    report_path = out_dir / "e4_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"E4 quantity curve ({PRIMARY_METRIC}, {RUNS_PER_SIZE} runs per size)")
    for size in sizes:
        c = report["curve"][str(size)]
        print(f"  size {size:>5}  U = {c['mean']:.4f}  sd = {c['sd']:.4f}")
    o = result["overall"]
    print(f"Overall U(N) - U(0) = {o['difference']:+.4f}  95% CI "
          f"[{o['ci95_two_sided'][0]:+.4f}, {o['ci95_two_sided'][1]:+.4f}]  "
          f"-> {'passes' if o['passes'] else 'does not pass'}")
    for k, s in result["steps"].items():
        tail = (f"Holm p = {s['holm_adjusted_p']:.4g} -> {'resolved' if s['resolved'] else 'not resolved'}"
                if s["tested"] else "reported, not tested")
        print(f"  step {k}  {s['from_size']:>5} -> {s['to_size']:<5} {s['difference']:+.4f}  {tail}")
    print(f"\nVERDICT: {result['verdict']}")
    return {"report": str(report_path), "verdict": result["verdict"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--phase", required=True, choices=["plan", "measure", "analyze"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    from scripts.utils.config import load_named_config

    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(stage3.paths.outputs_dir).parent / "followup" / args.namespace / EXPERIMENT)

    if args.phase == "plan":
        result = run_plan(args.namespace, out_dir)
    elif args.phase == "measure":
        result = run_measure(args.namespace, out_dir, args.device)
    else:
        result = run_analyze(out_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
