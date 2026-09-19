#!/usr/bin/env python3
"""Stage 5 — the protected evaluation and the comparison that answers the thesis.

The evaluator's preconditions are tested against the real entry point on a laid-out PROJECT_ROOT.
The analysis is tested on constructed prediction tables, because that is what it reads and because
the properties under test are properties of the METHOD, not of any result: that resampling happens
over lesions rather than images, that a condition's metric is the mean of its seeds rather than an
ensemble of them, that exactly one comparison is confirmatory. Where a test needs a known answer it
builds predictions with that answer deliberately — a condition made perfect, or two conditions made
identical — and says so.
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
from test_ham10000_stage4_aggregate import (  # noqa: E402
    LABELS,
    NAMESPACE,
    SEEDS,
    _write_grid,
    _write_run,
)

RUN_ID = "ham-stage5-fixture-a"
N_LESIONS = 40
IMAGES_PER_LESION = 2
N_IMAGES = N_LESIONS * IMAGES_PER_LESION


def _truth() -> np.ndarray:
    return np.array([index % len(LABELS) for index in range(N_IMAGES)])


def _reference() -> pd.DataFrame:
    truth = _truth()
    return pd.DataFrame({
        "image_id": [f"ISIC_{index:04d}" for index in range(N_IMAGES)],
        "lesion_id": [f"HAM_{index // IMAGES_PER_LESION:04d}" for index in range(N_IMAGES)],
        "true_class_index": truth,
    })


def _prediction_frame(condition: str, seed: int, *, accuracy: float = 0.6, perfect: bool = False) -> pd.DataFrame:
    """Probabilities with a DELIBERATE accuracy, so a test that needs a known answer has one."""
    reference = _reference()
    truth = reference["true_class_index"].to_numpy()
    rng = np.random.default_rng(abs(hash((condition, seed, perfect))) % 2**31)
    probabilities = rng.dirichlet(np.ones(len(LABELS)) * 0.5, N_IMAGES)
    correct = np.ones(N_IMAGES, dtype=bool) if perfect else rng.random(N_IMAGES) < accuracy
    for row in range(N_IMAGES):
        target = truth[row] if correct[row] else (truth[row] + 1) % len(LABELS)
        probabilities[row] = probabilities[row] * 0.1
        probabilities[row, target] += 0.9
        probabilities[row] /= probabilities[row].sum()

    frame = reference.copy()
    frame["true_dx"] = [LABELS[index] for index in truth]
    frame["condition"] = condition
    frame["seed"] = seed
    for index, label in enumerate(LABELS):
        frame[f"prob_{label}"] = probabilities[:, index]
    return frame


def _write_predictions(run_dir: Path, accuracies=None, **kwargs) -> Path:
    accuracies = accuracies or {"A": 0.55, "B": 0.60, "C": 0.70}
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for condition, accuracy in accuracies.items():
        for seed in SEEDS:
            frame = _prediction_frame(condition, seed, accuracy=accuracy, **kwargs)
            frame.to_parquet(predictions_dir / f"{condition}_seed{seed}.parquet", index=False)
    return predictions_dir


# ==============================================================================================
# The analysis
# ==============================================================================================


def _compare(run_dir: Path, **kwargs):
    from scripts.eval.ham10000_compare_conditions import run

    return run(run_dir, n_resamples=kwargs.pop("n_resamples", 150), **kwargs)


@pytest.fixture
def analysis_dir():
    with fixture_workspace("stage5-analysis") as root:
        _write_predictions(root)
        yield root


def test_exactly_one_comparison_is_confirmatory_and_it_is_C_versus_B(analysis_dir):
    """C vs A only shows that synthetic data helps. C vs B is the one that isolates SELECTING it,
    which is the contribution, so it is the only claim the family-wise correction protects."""
    result = _compare(analysis_dir)

    assert list(result["confirmatory"]) == ["C_vs_B:balanced_accuracy"]
    assert result["confirmatory"]["C_vs_B:balanced_accuracy"]["family"] == "confirmatory"
    assert all(entry["family"] == "exploratory" for entry in result["exploratory"].values())
    # Even the confirmatory PAIR's other metrics are exploratory: only the pre-registered metric on
    # the pre-registered pair carries the confirmatory claim.
    assert "C_vs_B:macro_f1" in result["exploratory"]
    assert "C_vs_A:balanced_accuracy" in result["exploratory"]


def test_every_p_value_arrives_with_an_effect_size_and_an_interval(analysis_dir):
    """Significance alone is not a result."""
    result = _compare(analysis_dir)

    for entry in list(result["confirmatory"].values()) + list(result["exploratory"].values()):
        assert {"observed_difference", "ci_lower", "ci_upper", "p_value"} <= set(entry)
        assert entry["ci_lower"] <= entry["observed_difference"] <= entry["ci_upper"]


def test_the_confirmatory_and_exploratory_families_are_corrected_differently(analysis_dir):
    """FDR controls the expected false-discovery proportion, not the family-wise error rate; the two
    are not interchangeable and the artifact must not blur them."""
    result = _compare(analysis_dir)

    confirmatory = result["confirmatory"]["C_vs_B:balanced_accuracy"]
    assert confirmatory["n_tests"] == 1  # Holm over a family of one
    exploratory = result["exploratory"]["C_vs_A:balanced_accuracy"]
    assert exploratory["n_tests"] == len(result["exploratory"])


def test_resampling_is_over_lesions_not_images(analysis_dir):
    """Several images of one lesion are not independent observations. Resampling images would make
    every interval narrower than the evidence supports."""
    result = _compare(analysis_dir)

    assert result["resampling_unit"] == "lesion_id"
    assert result["n_lesions"] == N_LESIONS
    assert result["n_images"] == N_IMAGES
    assert result["per_condition"]["A"]["balanced_accuracy"]["n_patients"] == N_LESIONS


def test_a_conditions_metric_is_the_mean_of_its_seeds_not_an_ensemble_of_them():
    """Averaging the seeds' PROBABILITIES would build an ensemble — a different and stronger model
    than any condition actually trained — and would quietly improve every condition."""
    from scripts.eval.ham10000_compare_conditions import condition_metric, metric_functions

    reference = _reference()
    truth = reference["true_class_index"].to_numpy()
    # One seed is perfect, two are deliberately wrong on every image. The mean of the three seeds'
    # accuracies is 1/3; an ensemble of them would not be.
    frames = [_prediction_frame("C", 42, perfect=True)]
    for seed in (43, 44):
        frame = _prediction_frame("C", seed, accuracy=0.0)
        frames.append(frame)

    evaluate = condition_metric(frames, truth, metric_functions()["accuracy"])
    assert evaluate(np.arange(len(truth))) == pytest.approx(1 / 3, abs=0.02)


def test_two_identical_conditions_show_no_difference(analysis_dir):
    """The null case the machinery has to get right before any positive result from it means
    anything: same predictions under two names must give a difference of zero."""
    from scripts.eval.ham10000_compare_conditions import load_predictions

    predictions_dir = analysis_dir / "predictions"
    for seed in SEEDS:
        frame = pd.read_parquet(predictions_dir / f"B_seed{seed}.parquet")
        frame["condition"] = "C"
        frame.to_parquet(predictions_dir / f"C_seed{seed}.parquet", index=False)

    result = _compare(analysis_dir)
    difference = result["confirmatory"]["C_vs_B:balanced_accuracy"]
    assert difference["observed_difference"] == pytest.approx(0.0, abs=1e-12)
    assert difference["rejected"] is False


def test_per_class_recall_is_reported_with_its_own_interval(analysis_dir):
    """A headline gain that came entirely from nv would say nothing about the rare classes the
    synthetic data was generated for."""
    result = _compare(analysis_dir)

    recalls = result["per_condition"]["C"]["per_class_recall"]
    assert set(recalls) == set(LABELS)
    for entry in recalls.values():
        assert entry["ci_lower"] <= entry["point_estimate"] <= entry["ci_upper"]
    assert any(name.startswith("C_vs_B:recall_") for name in result["exploratory"])


def test_a_missing_condition_stops_the_comparison(analysis_dir):
    for seed in SEEDS:
        (analysis_dir / "predictions" / f"B_seed{seed}.parquet").unlink()
    with pytest.raises(SystemExit, match=r"missing for condition\(s\) \['B'\]"):
        _compare(analysis_dir)


def test_conditions_evaluated_on_different_images_are_refused(analysis_dir):
    frame = pd.read_parquet(analysis_dir / "predictions" / "C_seed42.parquet")
    frame.loc[0, "image_id"] = "ISIC_not_in_the_others"
    frame.to_parquet(analysis_dir / "predictions" / "C_seed42.parquet", index=False)

    with pytest.raises(SystemExit, match="different image set"):
        _compare(analysis_dir)


def test_no_predictions_names_the_evaluator(analysis_dir):
    import shutil

    shutil.rmtree(analysis_dir / "predictions")
    with pytest.raises(SystemExit, match="ham10000_stage5_evaluate"):
        _compare(analysis_dir)


# ==============================================================================================
# The evaluator's preconditions
# ==============================================================================================


def _write_stage3_selection(root: Path, namespace: str = NAMESPACE) -> None:
    from scripts.utils.manifest import write_json

    path = root / "outputs/ham10000/stage3" / NAMESPACE / "asism_selection_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {"namespace": namespace, "threshold_policy": "network", "n_selected": 900})


@pytest.fixture
def workspace(monkeypatch):
    with fixture_workspace("stage5-eval") as root:
        monkeypatch.setenv("PROJECT_ROOT", str(root))
        overlay = root / "overlay.yaml"
        overlay.write_text(f"ham_stage4:\n  split_namespace: {NAMESPACE}\n", encoding="utf-8")
        monkeypatch.setenv("THESIS_CONFIG_OVERLAY", str(overlay))
        _write_grid(root)
        _write_stage3_selection(root)
        yield root


def _evaluate(root: Path, run_id: str = RUN_ID, **kwargs):
    from scripts.eval.ham10000_stage5_evaluate import run

    return run(NAMESPACE, run_id, **kwargs)


def test_the_protected_split_cannot_be_read_without_a_declared_run_id(workspace):
    """The run id is what makes "we looked once" checkable: it names the output directory, so a
    second evaluation is a separate result rather than an overwrite of the first."""
    with pytest.raises(SystemExit, match="explicit --final-eval-run-id"):
        _evaluate(workspace, run_id="")


def test_an_incomplete_stage4_grid_does_not_get_to_spend_the_one_look(workspace):
    (workspace / "outputs/ham10000/stage4" / NAMESPACE / "C/seed44/run_manifest.json").unlink()
    with pytest.raises(SystemExit, match="complete A/B/C grid or nothing"):
        _evaluate(workspace)


def test_a_confounded_comparison_is_refused_before_the_split_is_opened(workspace):
    """Being measured on held-out data does not make an unfair comparison sound."""
    _write_run(workspace, "B", 43, **{"budget.max_steps": 9000})
    with pytest.raises(SystemExit, match="FAIRNESS INVARIANT"):
        _evaluate(workspace)


def test_a_stage4_run_that_touched_the_protected_split_is_refused(workspace):
    _write_run(workspace, "A", 42, **{"data.selection_split": "final_eval_heldout"})
    with pytest.raises(SystemExit, match="PROTECTED SPLIT"):
        _evaluate(workspace)


def test_a_missing_selection_manifest_names_the_command_that_produces_it(workspace):
    (workspace / "outputs/ham10000/stage3" / NAMESPACE / "asism_selection_manifest.json").unlink()
    with pytest.raises(SystemExit, match="ham10000_05_adaptive_thresholds"):
        _evaluate(workspace)


def test_a_selection_from_another_namespace_would_evaluate_the_wrong_experiment(workspace):
    _write_stage3_selection(workspace, namespace="some-other-namespace")
    with pytest.raises(SystemExit, match="another experiment's selection"):
        _evaluate(workspace)


def test_a_protected_split_without_lesion_ids_is_refused(workspace):
    """Image-level resampling would treat several images of one lesion as independent evidence."""
    splits_dir = workspace / "data/ham10000/processed/splits" / NAMESPACE
    splits_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_id": ["ISIC_0000"], "dx": ["nv"]}).to_csv(
        splits_dir / "final_eval_heldout.csv", index=False
    )
    with pytest.raises(SystemExit, match="no lesion_id column"):
        _evaluate(workspace)


def test_an_outcome_read_of_the_protected_split_outside_stage5_is_refused():
    """The shared gate, exercised the way every other stage would hit it."""
    from scripts.utils.splits import FinalEvalAccessViolation, assert_final_eval_access_allowed

    with pytest.raises(FinalEvalAccessViolation):
        assert_final_eval_access_allowed("load_images", final_eval_run_id=None, caller="test")
    # A non-outcome purpose (checking the file exists) passes without a run id.
    assert_final_eval_access_allowed("existence_check", caller="test")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
