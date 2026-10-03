#!/usr/bin/env python3
"""Stage 4 aggregation — collect the conditions across seeds, and verify that only the DATA differed.

Stage 4 trains one classifier per (condition, seed). This collects those runs, checks that the
comparison between them is actually fair, and writes one table. It trains nothing and needs no GPU.

WHAT THE CONDITIONS ARE — see scripts/utils/ham10000_conditions.py, which fixes them per protocol
    v1   A  real only    B  real + ALL candidates    C   real + the candidates ASISM selected
    v2   A  real only    B  real + ALL candidates    C2  real + the v2 selection
                                                     D2  real + the same number, drawn at random
B is the condition that makes the thesis's claim testable: without it, C versus A would only show
that synthetic data helps, not that SELECTING it helps, which is the entire contribution. D2 does
the same job one level down in v2: it is C2's size held constant while the selection rule is
removed, so C2 over D2 is selection and not volume.

THE FAIRNESS INVARIANT, CHECKED RATHER THAN ASSUMED. Every condition must have trained under the
same optimizer-step budget, the same architecture and resolution, the same class weighting, the same
real training split and the same selection split. If any of those differ the comparison is
confounded, and this refuses to write a table rather than producing one that reads as though it
were. Equal STEPS and not equal epochs is the crux: at fixed epochs the larger B dataset silently
receives more gradient updates, and "more data" would be indistinguishable from "more training".

THESE ARE NOT RESULTS. Every number here is measured on the selection split — model-selection
evidence, the same split Stage 4 used while training. The comparison that counts happens in Stage 5
on final_eval_heldout, which is read nowhere in this file and nowhere in Stage 4. Each run's manifest
records `final_eval_heldout_read: false`, and that claim is verified here rather than trusted.

MISSING RUNS ARE NAMED, NEVER AVERAGED OVER. A condition with two of its three seeds is reported as
incomplete with the exact commands that would finish it; it is not quietly summarised as though the
third seed had agreed with the other two.

Usage:
    python scripts/classify/ham10000_aggregate_conditions.py --namespace ham-stratified-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS  # noqa: E402
from scripts.utils.ham10000_conditions import DEFAULT_PROTOCOL, PROTOCOLS, get_protocol  # noqa: E402
from scripts.utils.manifest import get_git_commit_hash, read_json, write_json  # noqa: E402

# v1's condition tuple, kept as a module attribute because Stage 5 imports it. The protocol
# object is the source of truth; this is its v1 view.
CONDITIONS = get_protocol("v1").conditions
HEADLINE_METRICS = ("balanced_accuracy", "macro_f1", "macro_auroc_ovr", "accuracy")
FORBIDDEN_SPLIT = "final_eval_heldout"

# Everything that must be identical across conditions for the comparison to mean anything. Each
# entry is a dotted path into a run manifest.
FAIRNESS_KEYS = (
    "budget.max_steps",
    "budget.batch_size",
    "budget.learning_rate",
    "budget.weight_decay",
    "model.architecture",
    "model.pretrained_source",
    "model.resolution",
    "model.dropout_p",
    "class_weighting",
    # The basis and the resulting vector, not only the on/off switch. Two runs can both say
    # "inverse_frequency" and still be weighted differently: with basis condition_training_set the
    # counts come from each condition's own training set, so on ham-stratified-v1 the df/nv weight
    # ratio is 43x for A but 1.3x for B — weighting would cancel most of what the synthetic images
    # add, and the switch alone would not show it. Under the frozen protocol (class_weighting=none)
    # both are None in every manifest, so these two entries cost nothing and close the gap in
    # advance for any sensitivity run that does turn weighting on.
    "class_weight_basis",
    "class_weights",
    "data.real_train_split",
    "data.selection_split",
    "loss",
)


class AggregationError(SystemExit):
    """The runs cannot be compared as they stand. The message says which invariant failed."""


def _dig(manifest: dict, path: str):
    node = manifest
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def discover_runs(results_dir: Path, namespace: str, seeds: list[int], conditions=CONDITIONS) -> tuple[dict, list[tuple[str, int]]]:
    """Every (condition, seed) that produced both a manifest and its metrics."""
    runs, missing = {}, []
    for condition in conditions:
        for seed in seeds:
            run_dir = Path(results_dir) / namespace / condition / f"seed{seed}"
            manifest_path = run_dir / "run_manifest.json"
            metrics_path = run_dir / "selection_metrics.json"
            if manifest_path.is_file() and metrics_path.is_file():
                runs[(condition, int(seed))] = {
                    "run_dir": str(run_dir),
                    "manifest": read_json(manifest_path),
                    "metrics": read_json(metrics_path),
                }
            else:
                missing.append((condition, int(seed)))
    return runs, missing


def check_protected_split(runs: dict) -> None:
    """No Stage 4 run may have touched the split Stage 5 reports on.

    Checked from the manifests rather than trusted: a run that read it would make every Stage 5
    number a measurement on data its own model had seen, and that is not recoverable afterwards.
    """
    offenders = []
    for (condition, seed), entry in runs.items():
        manifest = entry["manifest"]
        if manifest.get("final_eval_heldout_read") is not False:
            offenders.append(f"{condition}/seed{seed}: manifest does not assert final_eval_heldout_read=false")
            continue
        data = manifest.get("data", {})
        for key in ("real_train_split", "selection_split"):
            if str(data.get(key)) == FORBIDDEN_SPLIT:
                offenders.append(f"{condition}/seed{seed}: {key} is {FORBIDDEN_SPLIT}")
    if offenders:
        raise AggregationError(
            "PROTECTED SPLIT: Stage 4 runs touched final_eval_heldout:\n  " + "\n  ".join(offenders)
        )


def check_fairness(runs: dict) -> dict:
    """Only the training DATA may differ between conditions.

    Within a condition the seeds must also agree: a seed trained under a different budget is not a
    repeat of the others and cannot contribute to their mean.
    """
    observed: dict[str, dict] = {}
    violations = []
    for key in FAIRNESS_KEYS:
        values = {}
        for (condition, seed), entry in runs.items():
            values[f"{condition}/seed{seed}"] = _dig(entry["manifest"], key)
        distinct = {json.dumps(value, sort_keys=True, default=str) for value in values.values()}
        observed[key] = {"values": values, "n_distinct": len(distinct)}
        if len(distinct) > 1:
            violations.append(f"{key} differs across runs: {json.dumps(values, default=str)}")

    if violations:
        raise AggregationError(
            "FAIRNESS INVARIANT: the conditions did not train under the same protocol, so the "
            "comparison between them is confounded:\n  " + "\n  ".join(violations)
            + "\nRe-run the affected conditions under one protocol. No table is written from runs "
              "that are not comparable."
        )
    return {key: entry["values"][next(iter(entry["values"]))] for key, entry in observed.items()}


def check_training_data_differs(runs: dict, protocol=None) -> dict:
    """The one thing that SHOULD differ, checked rather than assumed.

    A condition that is supposed to carry synthetic images and has none is not that condition,
    whatever its directory is called. Two arms that are supposed to differ and share a count are
    either reading the same manifest or reporting a selector that kept everything. And a matched
    control that does NOT share its arm's count is not a control: C2 vs D2 would then be comparing
    sizes, which is the confound D2 exists to remove.
    """
    protocol = protocol or get_protocol(DEFAULT_PROTOCOL)
    per_condition = {}
    for condition in protocol.conditions:
        entries = [entry for (name, _), entry in runs.items() if name == condition]
        if not entries:
            continue
        synthetic = {int(entry["manifest"]["data"]["synthetic_images"]) for entry in entries}
        manifests = {str(entry["manifest"]["data"]["synthetic_manifest"]) for entry in entries}
        per_condition[condition] = {
            "synthetic_images": sorted(synthetic),
            "synthetic_manifest": sorted(manifests),
            "real_train_images": sorted({int(entry["manifest"]["data"]["real_train_images"]) for entry in entries}),
        }

    def counts(condition):
        return per_condition.get(condition, {}).get("synthetic_images")

    notes = []
    if counts(protocol.baseline) not in (None, [0]):
        notes.append(
            f"condition {protocol.baseline} was trained with synthetic images; "
            f"{protocol.baseline} is the real-only baseline"
        )
    for condition in protocol.synthetic_conditions:
        if counts(condition) == [0]:
            notes.append(f"condition {condition} was trained with no synthetic images")
    for left, right in protocol.equal_counts_suspect:
        if counts(left) and counts(right) and counts(left) == counts(right):
            notes.append(
                f"conditions {left} and {right} used the same number of synthetic images; either the "
                f"selector kept every candidate or {right} is reading {left}'s manifest"
            )
    for left, right in protocol.equal_counts_required:
        left_counts, right_counts = counts(left), counts(right)
        if not left_counts or not right_counts:
            continue
        if left_counts != right_counts:
            notes.append(
                f"conditions {left} and {right} used different numbers of synthetic images "
                f"({left_counts} vs {right_counts}); {right} is {left}'s size-matched control, so a "
                f"difference between them would be a size difference"
            )
        elif per_condition[left]["synthetic_manifest"] == per_condition[right]["synthetic_manifest"]:
            notes.append(
                f"conditions {left} and {right} read the same manifest; the control is not a "
                "separate draw"
            )
    if notes:
        raise AggregationError("CONDITION DEFINITION:\n  " + "\n  ".join(notes))
    return per_condition


def check_selection_outcome(runs: dict, protocol, selection: dict) -> dict:
    """For a selector that decides its own count, every count is a valid result, including "all"
    and "none". What is checked is that the protocol being aggregated is the one that outcome
    calls for, and that C trained on the images the selection recorded.

    "all" means C is B's training set and "none" means C is A's: neither is a broken run, and
    neither is trained a second time under another name. The outcome is copied into the report.
    """
    outcome = selection.get("selection_outcome")
    if outcome not in protocol.selection_outcomes:
        raise AggregationError(
            f"SELECTION OUTCOME: the selection manifest records {outcome!r}; protocol {protocol.name} "
            f"is for {list(protocol.selection_outcomes)}. Aggregate under --protocol {selection.get('protocol')}."
        )
    for (condition, seed), entry in runs.items():
        if condition == "C" and "C" in protocol.selection_manifest.covers:
            trained_on = entry["manifest"]["data"].get("synthetic_ids_sha256")
            if trained_on != selection.get("c_ids_sha256"):
                raise AggregationError(
                    f"SELECTION OUTCOME: C/seed{seed} trained on images other than the ones the "
                    "selection manifest recorded"
                )
    meaning = {"subset": "C is a proper subset of the candidates; D is its size-matched random control",
               "all": "the selector kept every candidate: C = B, and no C or D is trained separately",
               "none": "the selector kept no candidate: C = A, and no C or D is trained separately"}
    return {"selection_outcome": outcome, "meaning": meaning[outcome],
            "per_class": selection.get("per_class"), "n_selected_c": selection.get("n_selected_c")}


def _selection_manifest(protocol, namespace: str) -> dict:
    spec = protocol.selection_manifest
    path = Path(load_named_config(spec.config, spec.section).paths.outputs_dir) / namespace / spec.filename
    if not path.is_file():
        raise AggregationError(f"no selection manifest at {path}.\nRun: "
                               + spec.rebuild_command.format(namespace=namespace))
    return read_json(path)


def summarise(runs: dict, seeds: list[int], conditions=CONDITIONS) -> tuple[pd.DataFrame, dict]:
    rows = []
    for (condition, seed), entry in sorted(runs.items()):
        metrics = entry["metrics"]
        row = {"condition": condition, "seed": seed}
        row.update({name: float(metrics[name]) for name in HEADLINE_METRICS if name in metrics})
        for label in CLASSIFIER_TARGET_LABELS:
            per_class = metrics.get("per_class", {}).get(label, {})
            row[f"recall_{label}"] = float(per_class.get("recall", float("nan")))
            row[f"support_{label}"] = int(per_class.get("support", 0))
        rows.append(row)
    frame = pd.DataFrame(rows)

    summary = {}
    for condition in conditions:
        subset = frame[frame["condition"] == condition]
        if subset.empty:
            continue
        entry = {
            "seeds": sorted(int(value) for value in subset["seed"]),
            "n_seeds": int(len(subset)),
            "complete": sorted(int(value) for value in subset["seed"]) == sorted(int(s) for s in seeds),
        }
        for name in HEADLINE_METRICS:
            if name not in subset:
                continue
            values = subset[name].to_numpy(dtype=float)
            entry[name] = {
                "mean": float(np.mean(values)),
                # Sample standard deviation over seeds. With three seeds it is a crude spread
                # estimate and is reported as one — never as a confidence interval, which would
                # imply a sampling model three runs cannot support.
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "per_seed": {int(seed): float(value) for seed, value in zip(subset["seed"], values)},
            }
        entry["per_class_recall_mean"] = {
            label: float(subset[f"recall_{label}"].mean()) for label in CLASSIFIER_TARGET_LABELS
        }
        summary[condition] = entry
    return frame, summary


def rare_class_deltas(summary: dict, real_counts: dict[str, int] | None = None, protocol=None) -> dict:
    """Per-class recall change from the baseline, where a rebalancing intervention has to show up.

    Reported per class rather than only as a headline average: a balanced-accuracy gain that comes
    entirely from nv would mean the synthetic data did nothing for the classes it was generated for.
    """
    protocol = protocol or get_protocol(DEFAULT_PROTOCOL)
    if protocol.baseline not in summary:
        return {}
    baseline = summary[protocol.baseline]["per_class_recall_mean"]
    deltas = {}
    for condition in protocol.synthetic_conditions:
        if condition not in summary:
            continue
        deltas[condition] = {
            label: float(summary[condition]["per_class_recall_mean"][label] - baseline[label])
            for label in CLASSIFIER_TARGET_LABELS
        }
    return deltas


def run(namespace: str | None = None, protocol_name: str = DEFAULT_PROTOCOL) -> dict:
    condition_protocol = get_protocol(protocol_name)
    config = load_named_config(condition_protocol.stage4_config, "ham_stage4")
    namespace = namespace or str(config.split_namespace)
    seeds = [int(seed) for seed in config.seeds]
    results_dir = Path(config.paths.results_dir)

    runs, missing = discover_runs(results_dir, namespace, seeds, condition_protocol.conditions)
    if not runs:
        raise AggregationError(
            f"no Stage 4 runs under {results_dir / namespace}.\n"
            f"Run: python scripts/classify/ham10000_train_conditions.py --protocol {condition_protocol.name} "
            f"--condition {condition_protocol.baseline} --seed {seeds[0]}"
        )

    check_protected_split(runs)
    protocol = check_fairness(runs)
    data_per_condition = check_training_data_differs(runs, condition_protocol)
    selection_outcome = (check_selection_outcome(runs, condition_protocol,
                                                 _selection_manifest(condition_protocol, namespace))
                         if condition_protocol.selection_outcomes else None)
    frame, summary = summarise(runs, seeds, condition_protocol.conditions)

    out_dir = results_dir / namespace
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "stage4_condition_table.csv"
    frame.to_csv(table_path, index=False)

    report = {
        "stage": "ham10000_stage4_aggregate",
        "protocol": condition_protocol.name,
        "conditions": list(condition_protocol.conditions),
        "namespace": namespace,
        "measured_on": str(config.selection_split),
        "status": (
            "model-selection evidence, NOT results. Every number here is measured on the split "
            "Stage 4 itself used; the comparison that counts is Stage 5 on final_eval_heldout."
        ),
        "seeds_configured": seeds,
        "runs_found": [f"{condition}/seed{seed}" for condition, seed in sorted(runs)],
        "runs_missing": [f"{condition}/seed{seed}" for condition, seed in missing],
        "complete": not missing,
        "shared_protocol": protocol,
        "training_data_per_condition": data_per_condition,
        **({"selection": selection_outcome} if selection_outcome else {}),
        "per_condition": summary,
        f"per_class_recall_delta_vs_{condition_protocol.baseline}": rare_class_deltas(summary, protocol=condition_protocol),
        "table_path": str(table_path),
        "git_commit_hash": get_git_commit_hash(),
    }
    if missing:
        report["commands_for_missing_runs"] = [
            "python scripts/classify/ham10000_train_conditions.py "
            f"--protocol {condition_protocol.name} --condition {condition} --seed {seed}"
            for condition, seed in missing
        ]
    write_json(out_dir / "stage4_condition_summary.json", report)

    print(f"Stage 4 — {namespace} (measured on {config.selection_split}; NOT results)", flush=True)
    print("=" * 78, flush=True)
    header = f"  {'cond':<6}{'n':<4}" + "".join(f"{name[:14]:>16}" for name in HEADLINE_METRICS)
    print(header, flush=True)
    for condition in condition_protocol.conditions:
        entry = summary.get(condition)
        if not entry:
            print(f"  {condition:<6}{'-':<4}  no runs", flush=True)
            continue
        cells = ""
        for name in HEADLINE_METRICS:
            value = entry.get(name)
            if not value:
                cells += f"{'-':>16}"
            elif value["std"] is None:
                cells += f"{value['mean']:>16.4f}"
            else:
                cells += f"{value['mean']:>10.4f}±{value['std']:.3f}"
        print(f"  {condition:<6}{entry['n_seeds']:<4}{cells}", flush=True)
    print("=" * 78, flush=True)

    deltas = report[f"per_class_recall_delta_vs_{condition_protocol.baseline}"]
    if deltas:
        print(
            f"  per-class recall change vs {condition_protocol.baseline} "
            "(where a rebalancing intervention has to show up):",
            flush=True,
        )
        for condition, values in deltas.items():
            formatted = "  ".join(f"{label}{value:+.3f}" for label, value in values.items())
            print(f"    {condition}: {formatted}", flush=True)
    if missing:
        print(f"\n  INCOMPLETE — {len(missing)} run(s) missing; the table is not a comparison yet:", flush=True)
        for command in report["commands_for_missing_runs"]:
            print(f"    {command}", flush=True)
    print(f"\n  table -> {table_path}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL, choices=sorted(PROTOCOLS))
    args = parser.parse_args()

    report = run(args.namespace, args.protocol)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
