"""Deterministic frozen-embedding utility candidate for the proposed G0 gate.

Embeddings must come from one pinned encoder and be hash-bound by the caller.
This is a candidate instrument, not accepted supervision. All preprocessing
and logistic fitting use only the real+subset training records.
"""
from __future__ import annotations

import numpy as np

from scripts.utils.metrics import auroc


def frozen_linear_probe_utility(train_x: np.ndarray, train_y: np.ndarray,
                                outcome_x: np.ndarray, outcome_y: np.ndarray,
                                classes: tuple[int, ...]) -> dict:
    train_x = np.asarray(train_x, dtype=np.float64)
    outcome_x = np.asarray(outcome_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=int)
    outcome_y = np.asarray(outcome_y, dtype=int)
    if (train_x.ndim != 2 or outcome_x.ndim != 2 or train_x.shape[1] != outcome_x.shape[1]
            or len(train_x) != len(train_y) or len(outcome_x) != len(outcome_y)
            or not np.isfinite(train_x).all() or not np.isfinite(outcome_x).all()):
        raise ValueError("Invalid frozen-embedding matrix")
    if (set(train_y) != set(classes) or set(outcome_y) != set(classes)
            or len(set(classes)) < 2):
        raise ValueError("Missing class in train or outcome split")
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (train_x - mean) / scale
    x = np.column_stack([np.ones(len(train_x)), standardized])
    y = np.column_stack([(train_y == label).astype(float) for label in classes])
    penalty = np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    outcome = np.column_stack([np.ones(len(outcome_x)), (outcome_x - mean) / scale])
    logits = outcome @ coefficients
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    per_class = {str(label): auroc(probabilities[:, index], (outcome_y == label).astype(int))
                 for index, label in enumerate(classes)}
    return {"macro_auroc_ovr": float(np.mean(list(per_class.values()))),
            "per_class_auroc_ovr": per_class,
            "recipe": {"encoder": "pinned DINOv2 from ham10000_stage3.yaml",
                       "scaler": "per-feature train mean and population SD",
                       "classifier": "one-hot ridge least squares, lambda=1, unpenalized intercept, softmax"}}
