#!/usr/bin/env python3
"""Build pre-registered controlled-random subset recipes for learned ASISM.

This command does not invent utility labels. Each recipe must subsequently be evaluated by the
same proxy classifier protocol and written to utility_results.jsonl.

Two phases:
  --phase feasibility  Compute candidate-pool feasibility numbers (per-label/per-quantile-band
                       counts, achievable subset sizes, train/val-pool sizes, compute-budget
                       estimate) WITHOUT writing any subset recipe. Writes subset_design_report.json.
  --phase build        FAILS CLOSED unless a subset_design_report.json exists, matches the current
                       config (config_hash), reports zero feasibility failures, and the implied
                       compute-budget estimate is within learned_asism.compute_budget. No ratio or
                       size is ever adjusted automatically to make a failing report pass — a failing
                       report must be fixed by re-declaring subset_design/feasibility_thresholds in
                       configs/stage3_asism.yaml and re-running --phase feasibility. After building,
                       every written record is re-verified (no duplicate images, no train/val
                       overlap, every subset matches its declared size, subset counts match the plan)
                       before the recipe file is written to disk.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.asism.learned import (  # noqa: E402
    active_feature_columns, build_role_conditioned_subsets, pool_feasibility_report, split_image_pool, verify_built_subsets,
)
from scripts.asism.candidate_pool import load_candidate_pool  # noqa: E402
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.artifact_contracts import stage2_paths, stage3_paths  # noqa: E402
from scripts.utils.labels import PRIMARY_ENDPOINT_LABELS  # noqa: E402
from scripts.utils.manifest import get_git_commit_hash, hash_dict, read_json  # noqa: E402


def preflight_check_candidate_pool_inputs(cfg) -> None:
    """Fail closed with a clear, actionable list of every missing upstream artifact and the exact
    command that produces it — never a raw FileNotFoundError traceback from deep inside
    load_candidate_pool. These artifacts require GPU generation + signal computation (RunPod);
    nothing here invents or waits for them."""
    missing: list[str] = []

    scores_dir = Path(cfg.paths.scores_dir)
    if not scores_dir.is_dir() or not any(scores_dir.glob("*_scores.parquet")):
        missing.append(
            f"  - ASISM signal score artifacts: {scores_dir}/*_scores.parquet\n"
            "    -> python scripts/asism/01_compute_signals.py"
        )

    gonogo_path = Path(cfg.paths.gonogo_report)
    if not gonogo_path.is_file():
        missing.append(
            f"  - Go/No-Go report: {gonogo_path}\n"
            "    -> python scripts/asism/02_gonogo.py"
        )

    stage2_cfg = load_named_config("stage2_generation.yaml", "stage2")
    namespace = str(cfg.split_namespace)
    stage2_manifest = stage2_paths(stage2_cfg, namespace)["manifest_path"]
    if not stage2_manifest.is_file():
        missing.append(
            f"  - Stage 2 synthetic generation manifest: {stage2_manifest}\n"
            "    -> python scripts/generate/01_sample_label_recipes.py\n"
            "    -> python scripts/generate/02_generate_synthetic_images.py"
        )

    if missing:
        raise SystemExit(
            "UPSTREAM GATE: learned-ASISM needs artifacts that don't exist yet for namespace "
            f"{cfg.split_namespace!r}:\n\n" + "\n".join(missing) +
            "\n\nThese require GPU generation and signal computation (RunPod) — not available "
            "locally. Run the listed commands in order, then re-run this command."
        )


def subset_design_config_hash(design) -> str:
    """Hashes only the fields that determine feasibility, so an unrelated config edit elsewhere in
    the file doesn't spuriously invalidate a still-valid feasibility report."""
    relevant = {
        "subset_sizes": list(design.subset_sizes),
        "total_subsets": int(design.total_subsets),
        "random_fraction": float(design.random_fraction),
        "single_signal_fraction": float(design.single_signal_fraction),
        "mixed_fraction": float(design.mixed_fraction),
        "quantile_bins": int(design.quantile_bins),
        "val_pool_fraction": float(design.val_pool_fraction),
        "feasibility_thresholds": OmegaConf.to_container(design.feasibility_thresholds, resolve=True),
    }
    return hash_dict(relevant, length=32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--phase", choices=["feasibility", "build"], default="feasibility")
    args = parser.parse_args()
    cfg = load_named_config("stage3_asism.yaml", "stage3")
    namespace = args.namespace or str(cfg.split_namespace)
    for key, value in stage3_paths(cfg, namespace).items():
        if key in cfg.paths:
            cfg.paths[key] = str(value)
    cfg.split_namespace = namespace

    learned = cfg.learned_asism
    design = learned.subset_design
    val_fraction = float(design.val_pool_fraction)
    report_path = Path(cfg.paths.subset_design_report)
    config_hash = subset_design_config_hash(design)

    preflight_check_candidate_pool_inputs(cfg)

    if args.phase == "feasibility":
        merged, intended, surviving = load_candidate_pool(cfg)
        columns = active_feature_columns(list(learned.feature_columns), surviving)
        thresholds = OmegaConf.to_container(design.feasibility_thresholds, resolve=True)
        report = pool_feasibility_report(
            merged, columns, intended, list(PRIMARY_ENDPOINT_LABELS),
            list(design.subset_sizes), int(design.quantile_bins), val_fraction, int(design.seed),
            int(design.total_subsets), thresholds,
        )
        # Compute-budget failure is evaluated here too so --phase build can gate on ONE report,
        # not re-derive it — reuses 04b's own estimator so the two scripts never disagree.
        budget_module = _load_04b()
        budget_estimate = budget_module.compute_budget_estimate(cfg, int(design.total_subsets))
        if not budget_estimate["within_budget"]:
            report["failures"].append(
                f"compute estimate above cap: {budget_estimate['estimated_gpu_hours']}h > "
                f"{budget_estimate['max_gpu_hours']}h for {budget_estimate['total_proxy_runs']} "
                "proxy runs implied by subset_design.total_subsets"
            )
            report["passed"] = False
        report["compute_budget_estimate"] = budget_estimate
        report["config_hash"] = config_hash
        report["surviving_signals"] = sorted(surviving)
        report["active_feature_columns"] = columns
        report["git_commit_hash"] = get_git_commit_hash()

        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        print(f"\n-> {report_path}", flush=True)
        if report["passed"]:
            print("PASSED. Review this before --phase build.", flush=True)
            return 0
        print(f"FAILED — {len(report['failures'])} failure(s). --phase build will refuse to run "
              "until subset_design/feasibility_thresholds are revised and this is re-run.", flush=True)
        return 2

    # --phase build: fail closed.
    if not report_path.is_file():
        raise SystemExit(
            f"UPSTREAM GATE: no feasibility report at {report_path}.\n"
            "Run: python scripts/asism/04_build_utility_subsets.py --phase feasibility"
        )
    report = read_json(report_path)
    if report.get("config_hash") != config_hash:
        raise SystemExit(
            "STALE FEASIBILITY REPORT: configs/stage3_asism.yaml's subset_design/"
            "feasibility_thresholds changed since the report was generated "
            f"(report config_hash={report.get('config_hash')!r}, current={config_hash!r}).\n"
            "Re-run: python scripts/asism/04_build_utility_subsets.py --phase feasibility"
        )
    if not report.get("passed", False):
        raise SystemExit(
            f"FEASIBILITY GATE: {report_path} reports {len(report.get('failures', []))} failure(s):\n"
            + "\n".join(f"  - {reason}" for reason in report.get("failures", []))
            + "\n\nsubset_design/feasibility_thresholds must be revised in configs/stage3_asism.yaml "
              "and --phase feasibility re-run BEFORE --phase build. No automatic adjustment."
        )

    merged, _, surviving = load_candidate_pool(cfg)
    columns = active_feature_columns(list(learned.feature_columns), surviving)
    if report.get("surviving_signals") != sorted(surviving) or report.get("active_feature_columns") != columns:
        raise SystemExit(
            "STALE FEASIBILITY REPORT: Go/No-Go admitted signals changed since feasibility was measured.\n"
            "Re-run: python scripts/asism/04_build_utility_subsets.py --phase feasibility"
        )
    train_frame, val_frame = split_image_pool(merged, val_fraction, int(design.seed))
    train_total = round(int(design.total_subsets) * (1 - val_fraction))
    val_total = int(design.total_subsets) - train_total
    records = build_role_conditioned_subsets(
        train_frame, val_frame, columns, train_total, val_total,
        list(design.subset_sizes), float(design.random_fraction),
        float(design.single_signal_fraction), float(design.mixed_fraction),
        int(design.seed), quantile_bins=int(design.quantile_bins),
    )
    verify_built_subsets(records, train_total, val_total)  # last check before anything is written

    output = Path(cfg.paths.utility_subsets)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    n_train = sum(1 for r in records if r["role"] == "train")
    n_val = sum(1 for r in records if r["role"] == "val")
    print(f"{len(records)} frozen subset recipes ({n_train} train-role, {n_val} val-role) -> {output}")
    print("Next: run identical proxy training for every recipe and write utility_results.jsonl.")
    return 0


def _load_04b():
    import importlib.util
    script = Path(__file__).with_name("04b_evaluate_utility_subsets.py")
    spec = importlib.util.spec_from_file_location("utility_subsets_04b", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
