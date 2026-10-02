"""Seed-robustness analysis (protocol §5) and the ASISM v2 protocols and grid input checks."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from scripts.eval import ham10000_seed_robustness as rob
from scripts.followup import ham10000_asism_v2_stage4_grid as grid
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS
from scripts.utils.ham10000_conditions import get_protocol

K = len(CLASSIFIER_TARGET_LABELS)


def _frames(truth, accuracy, seeds, rng):
    frames = []
    for seed in seeds:
        predicted = np.where(rng.uniform(size=len(truth)) < accuracy, truth, rng.integers(0, K, len(truth)))
        probs = np.full((len(truth), K), 0.01)
        probs[np.arange(len(truth)), predicted] = 0.9
        frame = pd.DataFrame(probs, columns=[f"prob_{c}" for c in CLASSIFIER_TARGET_LABELS])
        frames.append(frame.assign(seed=seed))
    return frames


def _reference(n=140, seed=0):
    rng = np.random.default_rng(seed)
    truth = np.repeat(np.arange(K), n // K)
    return pd.DataFrame({"true_class_index": truth, "lesion_id": [f"L{i // 2}" for i in range(len(truth))]}), rng


def _plain_ba(truth, predicted):
    return float(np.mean([(predicted[truth == c] == c).mean() for c in np.unique(truth)]))


def test_weighted_ba_with_unit_weights_is_balanced_accuracy():
    rng = np.random.default_rng(1)
    truth = rng.integers(0, K, 300)
    predicted = np.where(rng.uniform(size=300) < 0.6, truth, rng.integers(0, K, 300))
    assert rob.balanced_accuracy_weighted(truth, predicted, np.ones(300)) == pytest.approx(_plain_ba(truth, predicted))


def test_weights_equal_expanded_rows():
    rng = np.random.default_rng(2)
    truth = rng.integers(0, K, 50)
    predicted = rng.integers(0, K, 50)
    weights = rng.integers(0, 4, 50)
    expanded_t, expanded_p = np.repeat(truth, weights), np.repeat(predicted, weights)
    assert rob.balanced_accuracy_weighted(truth, predicted, weights.astype(float)) == pytest.approx(
        _plain_ba(expanded_t, expanded_p))


def test_welch_matches_scipy():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0.6, 0.05, 20), rng.normal(0.57, 0.07, 20)
    out = rob.welch(a, b)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert out["p_value"] == pytest.approx(ref.pvalue)
    assert out["difference"] == pytest.approx(a.mean() - b.mean())
    assert out["ci95"][0] < out["difference"] < out["ci95"][1]


def test_bootstrap_is_deterministic_and_zero_for_identical_conditions():
    reference, rng = _reference()
    truth = reference["true_class_index"].to_numpy()
    lesions = reference["lesion_id"].to_numpy()
    left = _frames(truth, 0.6, [42, 43, 44], rng)
    one = rob.seed_lesion_bootstrap(left, left, truth, lesions, n_resamples=200)
    two = rob.seed_lesion_bootstrap(left, left, truth, lesions, n_resamples=200)
    assert one == two
    assert one["observed_difference"] == pytest.approx(0.0)
    assert one["ci95"][0] <= 0.0 <= one["ci95"][1]


def test_robustness_detects_a_clear_difference():
    reference, rng = _reference(n=700)
    truth = reference["true_class_index"].to_numpy()
    by_condition = {"C": _frames(truth, 0.8, range(42, 52), rng), "D": _frames(truth, 0.4, range(42, 52), rng)}
    out = rob.robustness(by_condition, reference, [("C", "D")], n_resamples=200)
    comparison = out["comparisons"]["C_vs_D"]
    assert comparison["welch"]["p_value"] < 1e-6
    assert comparison["seed_lesion_bootstrap"]["ci95"][0] > 0
    assert out["per_seed"]["C"]["seeds"] == list(range(42, 52))


# ---- protocols and grid -----------------------------------------------------------------------

def test_asism_v2_protocols():
    v2 = get_protocol("asism_v2")
    assert v2.conditions == ("A", "B", "C", "D")
    assert v2.confirmatory == (("C", "D"),)
    assert ("C", "D") in v2.equal_counts_required
    assert v2.selection_manifest.covers == ("C", "D")
    none = get_protocol("asism_v2_none")
    assert none.conditions == ("A", "B") and none.confirmatory == ()
    assert none.selection_manifest.covers == ()


def test_grid_refuses_a_file_the_selection_did_not_hash(tmp_path, monkeypatch):
    protocol = get_protocol("asism_v2")
    path = tmp_path / "c_selected.csv"
    path.write_text("image_id,image_path,dx\nx,/x.png,mel\n")
    monkeypatch.setattr(grid.stage4, "resolve_synthetic_manifest", lambda config, condition, seed: str(path))
    good = {"c_selected.csv": grid.sha256_file(path)}
    assert grid.verify_inputs(protocol, None, 42, "C", good) == good["c_selected.csv"]
    with pytest.raises(grid.GridError):
        grid.verify_inputs(protocol, None, 42, "C", {"c_selected.csv": "0" * 64})
    assert grid.verify_inputs(protocol, None, 42, "B", {}) is None


def test_asism_v2_stage4_recipe_is_the_v1_recipe_e4_checked():
    """E4 checked its recipe against ham10000_stage4.yaml (D1); the final grid trains with the
    asism_v2 config. Everything but the data paths and the seed list must be the same."""
    from scripts.utils.config import load_named_config

    v1 = load_named_config("ham10000_stage4.yaml", "ham_stage4")
    v2 = load_named_config(get_protocol("asism_v2").stage4_config, "ham_stage4")
    for key in ("model", "training", "loss", "classes", "real_train_split", "selection_split",
                "selection_metric", "split_namespace", "dataset"):
        assert v1[key] == v2[key], key
    assert list(v2.seeds) == list(range(42, 62))
    assert list(v2.seeds)[:3] == list(v1.seeds)


def test_classifier_val_comparison_end_to_end(tmp_path, monkeypatch):
    """Fake A/B/C/D Stage 4 runs (D with its own manifest per seed) -> gather, the frozen compare
    and the seed-robustness analysis, all on the monitoring split."""
    from omegaconf import OmegaConf

    from scripts.followup import ham10000_asism_v2_compare as cmp
    from scripts.utils.manifest import write_json

    seeds = [42, 43, 44]
    reference, rng = _reference(n=140)
    truth = reference["true_class_index"].to_numpy()
    accuracy = {"A": 0.5, "B": 0.6, "C": 0.7, "D": 0.6}
    synthetic = {"A": 0, "B": 3168, "C": 300, "D": 300}
    results = tmp_path / "results"
    for condition in "ABCD":
        for frame in _frames(truth, accuracy[condition], seeds, rng):
            seed = int(frame["seed"].iloc[0])
            run_dir = results / "ns" / condition / f"seed{seed}"
            run_dir.mkdir(parents=True)
            frame.assign(image_id=[f"i{k}" for k in range(len(truth))], lesion_id=reference["lesion_id"],
                         true_class_index=truth, condition=condition).to_parquet(
                run_dir / "selection_predictions.parquet")
            if not synthetic[condition]:
                manifest = None
            elif condition == "D":
                manifest = f"/m/d_selected_seed{seed}.csv"
            else:
                manifest = f"/m/{condition}.csv"
            write_json(run_dir / "run_manifest.json", {
                "final_eval_heldout_read": False, "loss": "cross_entropy", "class_weighting": "none",
                "class_weight_basis": None, "class_weights": None,
                "budget": {"max_steps": 3000, "batch_size": 32, "learning_rate": 1e-4, "weight_decay": 1e-4},
                "model": {"architecture": "densenet121", "pretrained_source": "imagenet", "resolution": 512,
                          "dropout_p": 0.2},
                "data": {"real_train_split": "classifier_train", "selection_split": "classifier_val",
                         "synthetic_images": synthetic[condition], "synthetic_manifest": manifest,
                         "real_train_images": 1641}})
            write_json(run_dir / "selection_metrics.json", {"balanced_accuracy": 0.0})
    fake = OmegaConf.create({"seeds": seeds, "paths": {"results_dir": str(results)}})
    monkeypatch.setattr(cmp, "load_named_config", lambda *a, **k: fake)
    out = cmp.run("asism_v2", "classifier_val", "ns")
    assert out["label"].startswith("MONITORING") and out["split"] == "classifier_val"
    assert list(out["confirmatory"]) == ["C_vs_D:balanced_accuracy"]
    assert out["seed_robustness"]["per_seed"]["D"]["seeds"] == seeds
    assert set(out["seed_robustness"]["comparisons"]) >= {"C_vs_D", "B_vs_A"}
    written = results / "ns" / "_classifier_val_comparison"
    assert (written / "asism_v2_comparison_classifier_val.json").is_file()
    assert len(list((written / "predictions").glob("*.parquet"))) == 12
