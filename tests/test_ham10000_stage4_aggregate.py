#!/usr/bin/env python3
"""Stage 4 aggregation — the fairness checks that decide whether A/B/C may be compared at all.

The run manifests and metric files here are constructed, because they are exactly what the
aggregator reads and the thing under test is what it does with them: a seed trained under a
different step budget, a condition C that turns out to be identical to B, a manifest that does not
assert the protected split was left alone. Nothing here asserts that any condition performed well;
the aggregator cannot produce a result, and neither can these tests.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fixture_workspace import fixture_workspace  # noqa: E402

NAMESPACE = "ham-stage4-v1"
SEEDS = (42, 43, 44)
LABELS = ("nv", "mel", "bkl", "bcc", "akiec", "vasc", "df")

SYNTHETIC = {"A": 0, "B": 3000, "C": 900}
MANIFEST_PATH = {
    "A": None,
    "B": "outputs/ham10000/stage2/all_candidates.csv",
    "C": "outputs/ham10000/stage3/asism_selected.csv",
}


def _metrics(condition: str, seed: int) -> dict:
    """Arbitrary but distinct per condition and seed, so the aggregation is visibly doing arithmetic
    rather than echoing one number. No assertion below reads these as evidence about a condition."""
    rng = np.random.default_rng(hash((condition, seed)) % 2**31)
    base = {"A": 0.60, "B": 0.62, "C": 0.65}[condition]
    recalls = {label: float(np.clip(base + rng.normal(0, 0.05), 0, 1)) for label in LABELS}
    return {
        "split": "classifier_val",
        "accuracy": base + 0.10,
        "balanced_accuracy": base,
        "macro_f1": base - 0.02,
        "macro_auroc_ovr": base + 0.25,
        "per_class": {
            label: {"recall": recalls[label], "precision": 0.5, "f1": 0.5, "support": 20}
            for label in LABELS
        },
    }


def _manifest(condition: str, seed: int, **overrides) -> dict:
    manifest = {
        "dataset": "ham10000",
        "condition": condition,
        "seed": seed,
        "loss": "cross_entropy",
        "class_weighting": "none",
        "model": {"architecture": "densenet121", "pretrained_source": "imagenet",
                  "resolution": 512, "dropout_p": 0.2},
        "budget": {"max_steps": 3000, "batch_size": 32, "learning_rate": 1e-4, "weight_decay": 1e-4},
        "data": {
            "real_train_split": "classifier_train",
            "selection_split": "classifier_val",
            "real_train_images": 5000,
            "synthetic_manifest": MANIFEST_PATH[condition],
            "synthetic_images": SYNTHETIC[condition],
        },
        "final_eval_heldout_read": False,
    }
    for path, value in overrides.items():
        node = manifest
        parts = path.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return manifest


def _write_run(root: Path, condition: str, seed: int, **overrides) -> Path:
    from scripts.utils.manifest import write_json

    run_dir = root / "outputs/ham10000/stage4" / NAMESPACE / condition / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run_manifest.json", _manifest(condition, seed, **overrides))
    write_json(run_dir / "selection_metrics.json", _metrics(condition, seed))
    return run_dir


def _write_grid(root: Path, conditions=("A", "B", "C"), seeds=SEEDS) -> None:
    for condition in conditions:
        for seed in seeds:
            _write_run(root, condition, seed)


def _run():
    from scripts.classify.ham10000_aggregate_conditions import run

    return run(NAMESPACE)


@pytest.fixture
def workspace(monkeypatch):
    with fixture_workspace("stage4-aggregate") as root:
        monkeypatch.setenv("PROJECT_ROOT", str(root))
        overlay = root / "overlay.yaml"
        overlay.write_text(f"ham_stage4:\n  split_namespace: {NAMESPACE}\n", encoding="utf-8")
        monkeypatch.setenv("THESIS_CONFIG_OVERLAY", str(overlay))
        yield root


# ==============================================================================================
# The complete grid
# ==============================================================================================


def test_the_full_grid_aggregates_to_one_table_marked_as_selection_evidence(workspace):
    _write_grid(workspace)
    report = _run()

    assert report["complete"] is True
    assert set(report["per_condition"]) == {"A", "B", "C"}
    assert all(report["per_condition"][condition]["n_seeds"] == 3 for condition in "ABC")
    assert report["measured_on"] == "classifier_val"
    # The one thing a reader must not mistake: these are not the thesis's results.
    assert "NOT results" in report["status"]
    assert "final_eval_heldout" in report["status"]

    frame = pd.read_csv(report["table_path"])
    assert len(frame) == 9
    assert set(frame["condition"]) == {"A", "B", "C"}
    assert {f"recall_{label}" for label in LABELS} <= set(frame.columns)


def test_the_spread_over_three_seeds_is_a_standard_deviation_not_a_confidence_interval(workspace):
    """Three runs cannot support the sampling model a CI implies, so the artifact reports what it
    actually has."""
    _write_grid(workspace)
    report = _run()

    entry = report["per_condition"]["B"]["balanced_accuracy"]
    assert set(entry) == {"mean", "std", "per_seed"}
    assert len(entry["per_seed"]) == 3
    assert entry["std"] is not None


def test_per_class_recall_change_is_reported_against_A(workspace):
    """A balanced-accuracy gain coming entirely from nv would mean the synthetic data did nothing
    for the classes it was generated for, and only the per-class view shows that."""
    _write_grid(workspace)
    report = _run()

    deltas = report["per_class_recall_delta_vs_A"]
    assert set(deltas) == {"B", "C"}
    assert set(deltas["C"]) == set(LABELS)
    baseline = report["per_condition"]["A"]["per_class_recall_mean"]
    assert deltas["C"]["df"] == pytest.approx(
        report["per_condition"]["C"]["per_class_recall_mean"]["df"] - baseline["df"]
    )


# ==============================================================================================
# Incomplete grids
# ==============================================================================================


def test_a_missing_seed_is_named_with_the_command_that_finishes_it(workspace):
    """A condition with two of its three seeds is not quietly summarised as though the third had
    agreed with the other two."""
    _write_grid(workspace, conditions=("A", "B"))
    _write_run(workspace, "C", 42)
    _write_run(workspace, "C", 43)

    report = _run()

    assert report["complete"] is False
    assert report["runs_missing"] == ["C/seed44"]
    assert report["per_condition"]["C"]["complete"] is False
    assert any("--condition C --seed 44" in command for command in report["commands_for_missing_runs"])


def test_no_runs_at_all_names_the_training_command(workspace):
    with pytest.raises(SystemExit, match="ham10000_train_conditions"):
        _run()


# ==============================================================================================
# Fairness — only the data may differ
# ==============================================================================================


def test_a_seed_trained_under_a_different_step_budget_stops_the_table(workspace):
    """Equal optimizer steps, not equal epochs, is the crux: at fixed epochs the larger B dataset
    silently receives more gradient updates and "more data" becomes indistinguishable from "more
    training"."""
    _write_grid(workspace)
    _write_run(workspace, "B", 43, **{"budget.max_steps": 6000})

    with pytest.raises(SystemExit, match="FAIRNESS INVARIANT"):
        _run()


def test_a_condition_trained_at_a_different_resolution_stops_the_table(workspace):
    _write_grid(workspace)
    _write_run(workspace, "C", 42, **{"model.resolution": 256})

    with pytest.raises(SystemExit, match="model.resolution differs"):
        _run()


def test_a_condition_trained_on_a_different_real_split_stops_the_table(workspace):
    _write_grid(workspace)
    _write_run(workspace, "A", 42, **{"data.real_train_split": "gen_train"})

    with pytest.raises(SystemExit, match="real_train_split differs"):
        _run()


def test_a_second_rebalancing_mechanism_applied_to_one_condition_stops_the_table(workspace):
    """Class weighting on C but not on A would make C's advantage partly a loss-function change."""
    _write_grid(workspace)
    _write_run(workspace, "C", 44, class_weighting="inverse_frequency")

    with pytest.raises(SystemExit, match="class_weighting differs"):
        _run()


# ==============================================================================================
# The conditions must actually be different conditions
# ==============================================================================================


def test_a_condition_C_identical_to_B_is_refused(workspace):
    """Either the selector kept every candidate — in which case C is not a selection — or C is
    reading B's manifest. Both turn the experiment into a different one."""
    _write_grid(workspace, conditions=("A", "B"))
    for seed in SEEDS:
        _write_run(workspace, "C", seed, **{"data.synthetic_images": SYNTHETIC["B"]})

    with pytest.raises(SystemExit, match="same number of synthetic images"):
        _run()


def test_a_condition_B_with_no_synthetic_images_is_refused(workspace):
    _write_grid(workspace, conditions=("A", "C"))
    for seed in SEEDS:
        _write_run(workspace, "B", seed, **{"data.synthetic_images": 0})

    with pytest.raises(SystemExit, match="condition B was trained with no synthetic images"):
        _run()


def test_a_real_only_baseline_carrying_synthetic_images_is_refused(workspace):
    _write_grid(workspace, conditions=("B", "C"))
    for seed in SEEDS:
        _write_run(workspace, "A", seed, **{"data.synthetic_images": 500})

    with pytest.raises(SystemExit, match="condition A was trained with synthetic images"):
        _run()


# ==============================================================================================
# The protected split
# ==============================================================================================


def test_a_run_that_selected_on_the_protected_split_is_refused(workspace):
    """A Stage 5 number measured on data the model selected against is not recoverable afterwards."""
    _write_grid(workspace)
    _write_run(workspace, "C", 42, **{"data.selection_split": "final_eval_heldout"})

    with pytest.raises(SystemExit, match="PROTECTED SPLIT"):
        _run()


def test_a_manifest_that_does_not_assert_the_protected_split_was_untouched_is_refused(workspace):
    """Absence of a claim is not a claim. A manifest from an older run that never recorded the flag
    cannot be read as though it had recorded it as false."""
    _write_grid(workspace)
    _write_run(workspace, "A", 43, final_eval_heldout_read=None)

    with pytest.raises(SystemExit, match="does not assert"):
        _run()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
