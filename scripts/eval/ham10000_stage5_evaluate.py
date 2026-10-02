#!/usr/bin/env python3
"""Stage 5 — the protected final evaluation of one protocol's conditions on final_eval_heldout.

This is the ONLY place in the HAM10000 pipeline that reads final_eval_heldout for an outcome. Every
earlier stage refuses it by name, and the access here goes through the same declared-purpose gate
the CheXpert path uses, which requires an explicit run id nothing else can supply.

WHAT THIS FILE DOES AND DOES NOT DO. It loads each frozen Stage 4 checkpoint, predicts on the
protected split, and writes one prediction table per run. It computes no comparison and no p-value:
that is ham10000_compare_conditions.py, which runs on the predictions alone, on a laptop, so the
analysis can be iterated without ever re-reading the split. Separating them is what makes "we looked
at the held-out data once" a checkable statement rather than an intention.

THE PRECONDITIONS, ALL REFUSALS RATHER THAN WARNINGS
  * an explicit --final-eval-run-id, which becomes the output directory: a second evaluation is a
    SEPARATE result, never an overwrite of the first;
  * every Stage 4 run manifest asserts it did not read the protected split, and none of them
    selected on it;
  * the conditions trained under one protocol and differ only in data — the Stage 4 aggregator's
    invariant, re-checked here, because a confounded comparison does not become sound by being
    measured on held-out data;
  * the selection manifest behind the selected condition(s) exists and names this namespace, so the
    run being evaluated is the one the selector actually produced.

WHICH CONDITIONS. --protocol names the condition set and, with it, the Stage 4 config, the selection
manifest to check and the directory predictions are written to. It defaults to v1, the primary
result, so every command written before protocols existed still means exactly what it meant. v2
writes under stage5_v2/: a follow-up evaluation can never land on top of the protected v1 result.

A prediction file that already exists for this run id is REUSED after validation, so an interrupted
evaluation resumes. That is an engineering property. Re-running under a changed method after seeing
results is not resuming — it needs a new run id, and produces a separate result.

Usage:
    python scripts/eval/ham10000_stage5_evaluate.py --namespace ham-stratified-v1 \
        --final-eval-run-id ham-stage5-2026-09-19-a            # v1, the primary result
    python scripts/eval/ham10000_stage5_evaluate.py --namespace ham-stratified-v1 \
        --protocol v2 --final-eval-run-id ham-stage5-v2-...    # the A/B/C2/D2 follow-up
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.classify.ham10000_aggregate_conditions import (  # noqa: E402
    check_fairness,
    check_protected_split,
    check_training_data_differs,
    discover_runs,
)
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS, normalize_diagnosis  # noqa: E402
from scripts.utils.ham10000_conditions import DEFAULT_PROTOCOL, PROTOCOLS, get_protocol  # noqa: E402
from scripts.utils.manifest import get_git_commit_hash, read_json, sha256_file, write_json  # noqa: E402
from scripts.utils.splits import assert_final_eval_access_allowed  # noqa: E402

FINAL_EVAL_SPLIT = "final_eval_heldout"


def conditions_phrase(conditions) -> str:
    """condition C / conditions C2 and D2, so a refusal names what it is actually about."""
    names = list(conditions)
    if len(names) == 1:
        return f"condition {names[0]}"
    return "conditions " + ", ".join(names[:-1]) + " and " + names[-1]


class PreconditionFailed(SystemExit):
    """A Stage 5 precondition does not hold. Nothing is evaluated and nothing is written."""


def enforce_preconditions(namespace: str, run_id: str, selection_cfg, stage4, condition_protocol) -> dict:
    if not run_id:
        raise PreconditionFailed(
            "Stage 5 requires an explicit --final-eval-run-id. It names the output directory, so a "
            "second evaluation is a separate result rather than an overwrite of the first."
        )
    # The gate every outcome-bearing read of the protected split passes through, shared with the
    # CheXpert path so there is one rule rather than two that might drift apart.
    assert_final_eval_access_allowed(
        "load_labels_for_evaluation", final_eval_run_id=run_id, caller="ham10000_stage5_evaluate"
    )

    seeds = [int(seed) for seed in stage4.seeds]
    conditions = condition_protocol.conditions
    runs, missing = discover_runs(Path(stage4.paths.results_dir), namespace, seeds, conditions=conditions)
    if missing:
        raise PreconditionFailed(
            f"Stage 5 evaluates the complete {'/'.join(conditions)} grid or nothing. Missing: "
            + ", ".join(f"{condition}/seed{seed}" for condition, seed in missing)
            + "\nEvaluating a partial grid on the protected split spends the one look on an "
              "incomplete comparison."
        )

    # A confounded comparison does not become sound by being measured on held-out data.
    check_protected_split(runs)
    shared = check_fairness(runs)
    data_per_condition = check_training_data_differs(runs, condition_protocol)

    spec = condition_protocol.selection_manifest
    covered = conditions_phrase(spec.covers)
    selection_manifest_path = Path(selection_cfg.paths.outputs_dir) / namespace / spec.filename
    if not selection_manifest_path.is_file():
        raise PreconditionFailed(
            f"{covered}'s selection manifest is missing at {selection_manifest_path}.\n"
            "Run: " + spec.rebuild_command.format(namespace=namespace)
        )
    selection = read_json(selection_manifest_path)
    if str(selection.get("namespace")) != namespace:
        raise PreconditionFailed(
            f"the selection manifest was produced for namespace {selection.get('namespace')!r}, not "
            f"{namespace!r}; {covered} would be evaluated against another experiment's selection"
        )

    return {
        "final_eval_run_id": run_id,
        "namespace": namespace,
        "runs": runs,
        "shared_protocol": shared,
        "training_data_per_condition": data_per_condition,
        "selection_manifest": str(selection_manifest_path),
        "selection_manifest_sha256": sha256_file(selection_manifest_path),
        **{f"selection_{key}": selection.get(key) for key in spec.evidence_keys},
    }


def load_protected_split(splits_dir: Path, images_root: Path, run_id: str):
    """The protected split's records, plus the lesion id every image belongs to.

    The lesion id is loaded here, with the split, because it is what the analysis resamples over:
    HAM10000 contains several images of the same lesion, and treating them as independent would
    shrink every confidence interval by pretending there is more evidence than there is.
    """
    assert_final_eval_access_allowed(
        "load_images", final_eval_run_id=run_id, caller="ham10000_stage5_evaluate"
    )
    from scripts.utils.ham10000_classifier import records_from_split

    path = Path(splits_dir) / f"{FINAL_EVAL_SPLIT}.csv"
    if not path.is_file():
        raise PreconditionFailed(f"missing protected split at {path}")
    frame = pd.read_csv(path)
    if "lesion_id" not in frame.columns:
        raise PreconditionFailed(
            f"{path} has no lesion_id column. Image-level resampling would treat several images of "
            "one lesion as independent evidence and understate every interval."
        )
    records = records_from_split(frame, Path(images_root) / FINAL_EVAL_SPLIT)
    if len(records) != len(frame):
        raise PreconditionFailed(
            f"{len(frame) - len(records)} of {len(frame)} protected-split images are absent from "
            f"{images_root / FINAL_EVAL_SPLIT}. Evaluating on whatever happens to be present would "
            "silently change the evaluation set."
        )
    lesion_of = dict(zip(frame["image_id"].astype(str), frame["lesion_id"].astype(str)))
    split_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return records, lesion_of, split_sha256


def predict(checkpoint: Path, records, resolution: int, dropout_p: float, pretrained: str, device):
    import torch

    from scripts.utils.classifier import build_model
    from scripts.utils.ham10000_classifier import predict_probabilities

    model = build_model(len(CLASSIFIER_TARGET_LABELS), dropout_p, pretrained)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    return predict_probabilities(model, records, resolution, device=device)


def run(namespace: str, run_id: str, device: str | None = None,
        protocol_name: str = DEFAULT_PROTOCOL) -> dict:
    condition_protocol = get_protocol(protocol_name)
    spec = condition_protocol.selection_manifest
    selection_cfg = load_named_config(spec.config, spec.section)
    stage4 = load_named_config(condition_protocol.stage4_config, "ham_stage4")
    splits_cfg = load_named_config("splits_ham10000.yaml", "ham_splits")

    evidence = enforce_preconditions(namespace, run_id, selection_cfg, stage4, condition_protocol)
    runs = evidence.pop("runs")

    splits_dir = Path(splits_cfg.paths.splits_root) / namespace
    images_root = Path(stage4.paths.images_root) / namespace
    records, lesion_of, split_sha256 = load_protected_split(splits_dir, images_root, run_id)

    out_dir = Path(stage4.paths.results_dir).parent / condition_protocol.stage5_dirname / namespace / run_id
    predictions_dir = out_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    image_ids = [record["image_id"] for record in records]
    truth = np.asarray([int(record["class_index"]) for record in records])
    resolution = int(stage4.model.resolution)

    written, reused = [], []
    for (condition, seed), entry in sorted(runs.items()):
        tag = f"{condition}_seed{seed}"
        path = predictions_dir / f"{tag}.parquet"
        if path.is_file():
            existing = pd.read_parquet(path)
            if list(existing["image_id"].astype(str)) == image_ids:
                reused.append(tag)
                continue
            raise PreconditionFailed(
                f"existing predictions for {tag} cover a different image set; refusing to reuse "
                "them. Use a new --final-eval-run-id."
            )

        checkpoint = Path(entry["run_dir"]) / "model.pt"
        if not checkpoint.is_file():
            raise PreconditionFailed(f"missing Stage 4 checkpoint {checkpoint}")
        print(f"  [{tag}] predicting on {len(records)} protected images", flush=True)
        probabilities = predict(
            checkpoint, records, resolution,
            float(stage4.model.dropout_p), str(stage4.model.pretrained_source), device,
        )
        frame = pd.DataFrame(
            {
                "image_id": image_ids,
                "lesion_id": [lesion_of[image_id] for image_id in image_ids],
                "true_class_index": truth,
                "true_dx": [CLASSIFIER_TARGET_LABELS[index] for index in truth],
                "condition": condition,
                "seed": int(seed),
            }
        )
        for index, label in enumerate(CLASSIFIER_TARGET_LABELS):
            frame[f"prob_{label}"] = probabilities[:, index]
        frame.to_parquet(path, index=False)
        write_json(
            path.with_suffix(".provenance.json"),
            {
                "final_eval_run_id": run_id,
                "namespace": namespace,
                "condition": condition,
                "seed": int(seed),
                "checkpoint_sha256": sha256_file(checkpoint),
                "stage4_run_dir": entry["run_dir"],
                "split": FINAL_EVAL_SPLIT,
                "split_csv_sha256": split_sha256,
                "n_images": len(records),
                "git_commit_hash": get_git_commit_hash(),
            },
        )
        written.append(tag)

    manifest = {
        "stage": "ham10000_stage5_evaluate",
        "final_eval_run_id": run_id,
        "namespace": namespace,
        "split": FINAL_EVAL_SPLIT,
        "split_csv_sha256": split_sha256,
        "n_images": len(records),
        "n_lesions": len(set(lesion_of.values())),
        "class_support": {
            label: int((truth == index).sum()) for index, label in enumerate(CLASSIFIER_TARGET_LABELS)
        },
        "protocol": condition_protocol.name,
        "conditions": list(condition_protocol.conditions),
        "seeds": [int(seed) for seed in stage4.seeds],
        "predictions_written": written,
        "predictions_reused": reused,
        "predictions_dir": str(predictions_dir),
        "resampling_unit": "lesion_id",
        "analysis_note": (
            "This file contains no comparison. Run scripts/eval/ham10000_compare_conditions.py on "
            "the predictions; it needs no GPU and never re-reads the protected split."
        ),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": get_git_commit_hash(),
        **evidence,
    }
    write_json(out_dir / "stage5_manifest.json", manifest)

    print(f"Stage 5 — {condition_protocol.name} — {namespace} / {run_id}", flush=True)
    print(f"  protected split: {len(records)} images across {manifest['n_lesions']} lesions", flush=True)
    print(f"  predictions written {len(written)}, reused {len(reused)} -> {predictions_dir}", flush=True)
    print("  no comparison computed here; run ham10000_compare_conditions.py next", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--final-eval-run-id", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL, choices=sorted(PROTOCOLS))
    args = parser.parse_args()

    manifest = run(args.namespace, args.final_eval_run_id, args.device, args.protocol)
    print(json.dumps({key: manifest[key] for key in ("final_eval_run_id", "n_images", "predictions_dir")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
