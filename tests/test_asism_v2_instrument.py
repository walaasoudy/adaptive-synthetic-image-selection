import numpy as np
import pytest

from scripts.asism_v2.instrument import frozen_linear_probe_utility


def test_probe_uses_only_training_distribution_and_reports_all_classes():
    rng = np.random.default_rng(4)
    train_y = np.repeat([0, 1, 2], 30)
    outcome_y = np.repeat([0, 1, 2], 12)
    train_x = np.eye(3)[train_y] + rng.normal(0, .1, (90, 3))
    outcome_x = np.eye(3)[outcome_y] + rng.normal(0, .1, (36, 3))
    result = frozen_linear_probe_utility(train_x, train_y, outcome_x, outcome_y, (0, 1, 2))
    assert result["macro_auroc_ovr"] > .95
    assert set(result["per_class_auroc_ovr"]) == {"0", "1", "2"}
    altered = outcome_x + 100
    # Changes to the outcome distribution must not alter training preprocessing;
    # here only the scored values change, and the function still runs without
    # fitting a scaler to outcome rows.
    assert frozen_linear_probe_utility(train_x, train_y, altered, outcome_y,
                                       (0, 1, 2))["recipe"] == result["recipe"]


def test_probe_rejects_missing_class_and_nonfinite_embeddings():
    x = np.eye(3)
    with pytest.raises(ValueError, match="Missing class"):
        frozen_linear_probe_utility(x[:2], np.array([0, 1]), x, np.array([0, 1, 2]), (0, 1, 2))
    with pytest.raises(ValueError, match="matrix"):
        frozen_linear_probe_utility(np.full((3, 3), np.nan), np.array([0, 1, 2]),
                                    x, np.array([0, 1, 2]), (0, 1, 2))
