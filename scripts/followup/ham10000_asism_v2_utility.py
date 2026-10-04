#!/usr/bin/env python3
"""ASISM v2 utility measurement: the labels the learned ranker is supervised by (contract §8).

Design: docs/asism_v2_quantity_design_check_2026-10-03.md §8 and §9, approved by Walaa on 2026-10-03.
Every value comes from scripts/asism_v2/prereg.py; nothing is chosen here.

  --phase plan      laptop, seconds. Freezes the roles, the 200 designed subsets and the 1,000 cells.
  --phase measure   GPU, about 4 hours. THE ONLY GPU STEP OF THE RANKER. Refused until
                    MEASUREMENT_APPROVED is set by a dated commit recording Walaa's approval, and
                    then only with --i-understand-this-trains-real-models. Resumable.
  --phase gate      laptop, seconds. G1 on the train + validation subsets, written once. The test
                    subsets' outcomes are kept in a separate file that this phase never opens.

Each run trains the 224 px / 300 step proxy on classifier_train plus one subset and reads macro
AUROC on asism_tuning_heldout. classifier_val and final_eval_heldout are never read.

Usage:
    python -m scripts.followup.ham10000_asism_v2_utility --phase plan \
        --candidates <stage2/all_candidates.csv> --scores-dir <four_signal/gonogo_signal_set>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from scripts.asism_v2.contracts import fingerprint, validate_measurements, write_new_json
from scripts.asism_v2.gates import reliability_gate
from scripts.asism_v2.prereg import load_prereg
from scripts.asism_v2.preserve import sha256
from scripts.asism_v2.supervision import (PLAN_NAME, build_plan, describe, freeze_plan, ids_sha256,
                                          load_signal_table)
from scripts.followup.ham10000_proxy_noise_floor import (V1_ARCHITECTURE, _append_jsonl, _gpu_name,
                                                         _read_jsonl, freeze_measure_inputs, variant_proxy)

# Approved by Walaa on 2026-10-04 (contract §8, "MEASUREMENT APPROVED"), recorded in the contract
# first and set here in a commit of its own; the flag alone never starts a run.
MEASUREMENT_APPROVED = True
REAL_MODELS_FLAG = "--i-understand-this-trains-real-models"

PROXY_VARIANT = "v1"                                  # the 224 px / 300 step recipe
FORBIDDEN_SPLITS = frozenset({"final_eval_heldout", "classifier_val"})
FIT_ROLES = ("train", "validation")
FIT_RUNS, TEST_RUNS = "utility_runs_fit.jsonl", "utility_runs_test.jsonl"
INPUTS_NAME, GATE_NAME = "measure_inputs.json", "g1_reliability.json"
MEASUREMENT_CODE = ("scripts/followup/ham10000_asism_v2_utility.py",
                    "scripts/asism/ham10000_03_build_utility_subsets.py",
                    "scripts/utils/ham10000_classifier.py", "scripts/utils/classifier.py")


class UtilityError(RuntimeError):
    """Raised instead of measuring, gating or fitting on inputs that cannot carry the result."""


def default_out_dir(prereg: dict, namespace: str) -> Path:
    if namespace != prereg["split_namespace"]:
        raise UtilityError(f"namespace {namespace!r} is not the pre-registered {prereg['split_namespace']!r}")
    return Path(prereg["paths"]["outputs_dir"]) / namespace


def proxy_recipe(prereg: dict) -> dict:
    proxy = {**variant_proxy(PROXY_VARIANT), "architecture": V1_ARCHITECTURE}
    for key, value in prereg["supervision"]["proxy"].items():
        if int(proxy[key]) != int(value):
            raise UtilityError(f"the proxy's {key} is {proxy[key]}, the pre-registration says {value}")
    return proxy


def run_plan(namespace: str, candidates: Path, scores_dir: Path, out_dir: Path | None = None) -> dict:
    prereg = load_prereg()
    out_dir = Path(out_dir or default_out_dir(prereg, namespace))
    frame, report = load_signal_table(candidates, scores_dir, prereg["safety"])
    plan = build_plan(frame, prereg, report)
    path = freeze_plan(out_dir, plan)
    return {"plan": str(path), **describe(plan)}


def load_plan(out_dir: Path, prereg: dict) -> dict:
    path = Path(out_dir) / PLAN_NAME
    if not path.is_file():
        raise UtilityError(f"no frozen plan at {path}; run --phase plan first")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan["prereg_sha256"] != prereg["prereg_sha256"]:
        raise UtilityError("the frozen plan was drawn under another pre-registration")
    for sid, subset in plan["subsets"].items():
        if ids_sha256(subset["image_ids"]) != subset["members_sha256"]:
            raise UtilityError(f"{sid} no longer matches its frozen member hash")
    return plan


def job_order(plan: dict) -> list[dict]:
    """Seed-major, so an interrupted pod leaves every subset at about the same depth."""
    return sorted(plan["cells"], key=lambda c: (c["seed"], c["subset_id"]))


def measurement_protocol(plan: dict, prereg: dict, hashes: dict, plan_sha256: str) -> dict:
    """What every measured row is bound to. The instrument decision is added after G1, not here."""
    return {"dataset": "ham10000", "namespace": prereg["split_namespace"],
            "metric": prereg["supervision"]["metric"], "recipe_sha256": plan_sha256,
            "classifier_recipe_sha256": fingerprint(proxy_recipe(prereg)),
            "feature_artifacts_sha256": fingerprint(plan["pool"]["signal_artifact_sha256"]),
            "candidate_pool_sha256": plan["pool"]["candidates_csv_sha256"],
            "outcome_split_sha256": hashes["outcome_split"],
            "real_train_split_sha256": hashes["real_train_split"],
            "measurement_code_sha256": hashes["measurement_code"],
            "prereg_sha256": prereg["prereg_sha256"]}


def run_measure(namespace: str, device: str | None, flag_given: bool, candidates: Path | None = None,
                out_dir: Path | None = None, measure_fn: Callable | None = None,
                inputs: dict | None = None) -> dict:
    """Train and score every pending cell. `measure_fn` and `inputs` exist for the CPU tests."""
    if not MEASUREMENT_APPROVED:
        raise UtilityError("the utility measurement is a GPU run that has NOT been approved "
                           "(contract §8: 'the measurement needs its own approval'). Nothing is trained.")
    if not flag_given:
        raise UtilityError(f"this phase trains real models on a GPU; pass {REAL_MODELS_FLAG}")
    prereg = load_prereg()
    out_dir = Path(out_dir or default_out_dir(prereg, namespace))
    plan = load_plan(out_dir, prereg)
    if {prereg["real_train_split"], prereg["utility_split"]} & FORBIDDEN_SPLITS:
        raise UtilityError("the measurement trains on classifier_train and reads asism_tuning_heldout only")
    if inputs is None:
        inputs = _measure_inputs(namespace, plan, prereg, Path(candidates))
    protocol = measurement_protocol(plan, prereg, inputs["hashes"], sha256(out_dir / PLAN_NAME))
    print(f"measure inputs: {freeze_measure_inputs(out_dir / INPUTS_NAME, protocol)}", flush=True)
    stamp = fingerprint(protocol)
    proxy = proxy_recipe(prereg)
    measure = measure_fn or _default_measure
    metric = prereg["supervision"]["metric"]

    done = {(r["subset_id"], r["seed"]) for name in (FIT_RUNS, TEST_RUNS) if (out_dir / name).is_file()
            for r in _read_jsonl(out_dir / name)}
    pending = [c for c in job_order(plan) if (c["subset_id"], c["seed"]) not in done]
    gpu = _gpu_name()
    print(f"{len(pending)} of {len(plan['cells'])} utility runs pending on {gpu}", flush=True)
    for position, cell in enumerate(pending, start=1):
        subset = plan["subsets"][cell["subset_id"]]
        train = inputs["real"] + [inputs["candidates"][i] for i in subset["image_ids"]]
        started = time.perf_counter()
        metrics = measure(train, inputs["tuning"], SimpleNamespace(**proxy), int(cell["seed"]), device)
        row = {"subset_id": cell["subset_id"], "seed": int(cell["seed"]), "role": subset["role"],
               "size": subset["size"], "members_sha256": subset["members_sha256"],
               "protocol_sha256": stamp, "augmented_metric": float(metrics[metric]),
               "seconds": round(time.perf_counter() - started, 1), "gpu": gpu}
        _append_jsonl(out_dir / (FIT_RUNS if subset["role"] in FIT_ROLES else TEST_RUNS), row)
        print(f"  [{position}/{len(pending)}] {cell['subset_id']} seed {cell['seed']} "
              f"{metric}={row['augmented_metric']:.4f} ({row['seconds']:.0f}s)", flush=True)
    return {"out_dir": str(out_dir), "completed": len(plan["cells"])}


def _default_measure(train, tuning, proxy, seed, device):
    from scripts.asism.ham10000_03_build_utility_subsets import _measure

    return _measure(train, tuning, proxy, seed, device)


def _measure_inputs(namespace: str, plan: dict, prereg: dict, candidates: Path) -> dict:
    import pandas as pd

    from scripts.asism.ham10000_03_build_utility_subsets import _candidate_records, _split_records
    from scripts.utils.config import load_named_config

    if sha256(candidates) != plan["pool"]["candidates_csv_sha256"]:
        raise UtilityError(f"{candidates} is not the candidate pool the plan was drawn from")
    stage1 = load_named_config("ham10000_stage1.yaml", "ham_stage1")
    splits_cfg = load_named_config("splits_ham10000.yaml", "ham_splits")
    splits_dir = Path(splits_cfg.paths.splits_root) / namespace
    images_root = Path(stage1.paths.images_dir) / namespace
    real_split, tuning_split = prereg["real_train_split"], prereg["utility_split"]
    manifest = pd.read_csv(candidates)
    manifest["image_id"] = manifest["image_id"].astype(str)
    pool = manifest[manifest["image_id"].isin(plan["roles"])]
    if len(pool) != len(plan["roles"]):
        raise UtilityError("the candidate manifest does not hold every image of the plan")
    repo = Path(__file__).resolve().parents[2]
    return {"real": _split_records(splits_dir, images_root, real_split),
            "tuning": _split_records(splits_dir, images_root, tuning_split),
            "candidates": _candidate_records(pool, manifest),
            "hashes": {"outcome_split": sha256(splits_dir / f"{tuning_split}.csv"),
                       "real_train_split": sha256(splits_dir / f"{real_split}.csv"),
                       "measurement_code": fingerprint({name: sha256(repo / name) for name in MEASUREMENT_CODE})}}


def _rows(out_dir: Path, name: str, protocol: dict, stamp_as: dict) -> list[dict]:
    """Rows of one runs file, each checked against the measurement it claims, then bound to
    `stamp_as` (the same protocol once the instrument decision has been added)."""
    path = Path(out_dir) / name
    if not path.is_file():
        raise UtilityError(f"no measurements at {path}")
    measured, bound = fingerprint(protocol), fingerprint(stamp_as)
    rows = _read_jsonl(path)
    if any(row["protocol_sha256"] != measured for row in rows):
        raise UtilityError(f"{name} holds a row measured under another protocol; refusing to mix runs")
    return [{**row, "protocol_sha256": bound} for row in rows]


def _measured_protocol(out_dir: Path) -> dict:
    path = Path(out_dir) / INPUTS_NAME
    if not path.is_file():
        raise UtilityError(f"no {INPUTS_NAME} in {out_dir}: nothing has been measured")
    return json.loads(path.read_text(encoding="utf-8"))


def run_gate(namespace: str, out_dir: Path | None = None) -> dict:
    """G1, once, on the train + validation subsets. A failed gate is the result: no fit follows."""
    prereg = load_prereg()
    out_dir = Path(out_dir or default_out_dir(prereg, namespace))
    plan = load_plan(out_dir, prereg)
    protocol = _measured_protocol(out_dir)
    members = {sid: s["image_ids"] for sid, s in plan["subsets"].items() if s["role"] in FIT_ROLES}
    values = validate_measurements(_rows(out_dir, FIT_RUNS, protocol, protocol), members,
                                   plan["training_seeds"], protocol)
    gate = reliability_gate(values, {sid: plan["subsets"][sid]["size"] for sid in members},
                            prereg["reliability"]["min_reliability_of_subset_means"])
    gate.update(prereg_sha256=prereg["prereg_sha256"], measurement_protocol_sha256=fingerprint(protocol),
                test_subsets_read=False,
                if_not_passed="the labels are not learned from; this failure is the reported result")
    write_new_json(out_dir / GATE_NAME, gate)
    return gate


def accepted_protocol(out_dir: Path) -> dict:
    """The protocol fit_ranker requires: the measured one plus a passed, hash-bound G1."""
    gate_path = Path(out_dir) / GATE_NAME
    if not gate_path.is_file():
        raise UtilityError("G1 has not been run; run --phase gate")
    gate, protocol = json.loads(gate_path.read_text(encoding="utf-8")), _measured_protocol(out_dir)
    if gate["measurement_protocol_sha256"] != fingerprint(protocol):
        raise UtilityError("G1 was computed on other measurements")
    if not gate["passed"]:
        raise UtilityError(f"G1 failed (reliability {gate['reliability_of_mean']:.3f} < {gate['threshold']}): "
                           "the ranker is not fitted")
    return {**protocol, "instrument_decision": "accepted", "instrument_gate_sha256": sha256(gate_path)}


def fit_measurements(out_dir: Path) -> tuple[list[dict], dict]:
    protocol = accepted_protocol(out_dir)
    return _rows(out_dir, FIT_RUNS, _measured_protocol(out_dir), protocol), protocol


def heldout_measurements(out_dir: Path) -> list[dict]:
    """Read only by the acceptance check, which is itself written once."""
    return _rows(out_dir, TEST_RUNS, _measured_protocol(out_dir), accepted_protocol(out_dir))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", default="ham-stratified-v1")
    parser.add_argument("--phase", required=True, choices=["plan", "measure", "gate"])
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--scores-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(REAL_MODELS_FLAG, dest="real_models", action="store_true")
    args = parser.parse_args()
    if args.phase == "plan":
        if not (args.candidates and args.scores_dir):
            parser.error("--phase plan needs --candidates and --scores-dir")
        result = run_plan(args.namespace, args.candidates, args.scores_dir, args.out_dir)
    elif args.phase == "measure":
        if MEASUREMENT_APPROVED and not args.candidates:
            parser.error("--phase measure needs --candidates")
        result = run_measure(args.namespace, args.device, args.real_models, args.candidates, args.out_dir)
    else:
        result = run_gate(args.namespace, args.out_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
