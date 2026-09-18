#!/usr/bin/env python3
"""MC Dropout for the HAM10000 classifier: added without changing what the existing callers get.

The failure this file guards against is a silent one. If dropout is not actually re-enabled, or if
a single pass is accepted, MC Dropout still returns a well-formed standard deviation — it is just
identically zero, and the Stage 3 uncertainty signal becomes a constant that no downstream check
would flag as missing.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS  # noqa: E402
from scripts.utils.ham10000_classifier import (  # noqa: E402
    predict_probabilities,
    predict_probabilities_mc_dropout,
)
from fixture_workspace import fixture_workspace

N_CLASSES = len(CLASSIFIER_TARGET_LABELS)
RESOLUTION = 64


def _records(workspace: Path, n: int = 4) -> list[dict]:
    rng = np.random.default_rng(0)
    records = []
    for index in range(n):
        path = workspace / f"img_{index}.jpg"
        Image.fromarray(rng.integers(0, 255, (RESOLUTION, RESOLUTION, 3), dtype=np.uint8)).save(path)
        records.append({"image_id": path.stem, "image_path": str(path), "class_index": index % N_CLASSES})
    return records


def _model(dropout_p: float = 0.5):
    from scripts.utils.classifier import build_model

    return build_model(N_CLASSES, dropout_p, "random", seed=42)


# --------------------------------------------------------------------------------------------
# The existing API is unchanged
# --------------------------------------------------------------------------------------------

def test_default_prediction_still_returns_a_bare_array_of_softmax_rows():
    with fixture_workspace("ham-mc-default") as workspace:
        records = _records(workspace)
        probabilities = predict_probabilities(_model(), records, RESOLUTION, device="cpu")

    assert isinstance(probabilities, np.ndarray), "Stage 4 and the auxiliary classifier index this directly"
    assert probabilities.shape == (len(records), N_CLASSES)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_default_prediction_is_deterministic():
    """Dropout must be OFF on this path: two calls on one model must agree exactly."""
    with fixture_workspace("ham-mc-deterministic") as workspace:
        records = _records(workspace)
        model = _model()
        first = predict_probabilities(model, records, RESOLUTION, device="cpu")
        second = predict_probabilities(model, records, RESOLUTION, device="cpu")

    assert np.array_equal(first, second)


def test_a_model_left_in_mc_mode_still_predicts_deterministically_afterwards():
    """MC Dropout hands the model back in eval mode, so a later Stage 4 call is not stochastic."""
    with fixture_workspace("ham-mc-restore") as workspace:
        records = _records(workspace)
        model = _model()
        predict_probabilities_mc_dropout(model, records, RESOLUTION, passes=3, device="cpu")
        first = predict_probabilities(model, records, RESOLUTION, device="cpu")
        second = predict_probabilities(model, records, RESOLUTION, device="cpu")

    assert np.array_equal(first, second)


# --------------------------------------------------------------------------------------------
# The MC path
# --------------------------------------------------------------------------------------------

def test_mc_dropout_returns_mean_and_std_of_the_right_shape():
    with fixture_workspace("ham-mc-shape") as workspace:
        records = _records(workspace)
        mean, std = predict_probabilities_mc_dropout(_model(), records, RESOLUTION, passes=5, device="cpu")

    assert mean.shape == std.shape == (len(records), N_CLASSES)
    assert np.allclose(mean.sum(axis=1), 1.0), "the mean of softmax rows is still a distribution"
    assert np.all(std >= 0.0)


def test_mc_dropout_actually_varies_between_passes():
    """A non-zero spread somewhere is the whole point; all-zero means dropout never switched on."""
    with fixture_workspace("ham-mc-varies") as workspace:
        records = _records(workspace)
        _, std = predict_probabilities_mc_dropout(_model(0.5), records, RESOLUTION, passes=8, device="cpu")

    assert std.max() > 0.0


def test_batchnorm_stays_in_eval_while_dropout_is_active():
    """model.train() would switch BatchNorm to batch statistics and change the predictions
    themselves, so the spread would no longer be epistemic uncertainty."""
    import torch

    from scripts.utils.classifier import enable_mc_dropout

    model = _model()
    model.eval()
    activated = enable_mc_dropout(model)

    assert activated > 0, "DenseNet121 built with dropout_p must expose dropout modules"
    norm_layers = [m for m in model.modules() if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d))]
    assert norm_layers, "the fixture model should contain BatchNorm, or this test proves nothing"
    assert all(not layer.training for layer in norm_layers)
    assert all(layer.training for layer in model.modules() if isinstance(layer, torch.nn.Dropout))


# --------------------------------------------------------------------------------------------
# It refuses rather than degenerating
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("passes", [0, 1, -3])
def test_refuses_fewer_than_two_passes(passes):
    with fixture_workspace("ham-mc-passes") as workspace:
        records = _records(workspace)
        with pytest.raises(ValueError, match="identically zero"):
            predict_probabilities_mc_dropout(_model(), records, RESOLUTION, passes=passes, device="cpu")


def test_refuses_a_model_with_no_dropout_modules():
    """Without this guard every pass is identical and the uncertainty signal is a silent constant."""
    import torch

    model = _model()
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Dropout):
                setattr(module, child_name, torch.nn.Identity())
    assert not [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]

    with fixture_workspace("ham-mc-nodropout") as workspace:
        records = _records(workspace)
        with pytest.raises(ValueError, match="no dropout modules"):
            predict_probabilities_mc_dropout(model, records, RESOLUTION, passes=5, device="cpu")
