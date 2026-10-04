"""Multi-class metrics for HAM10000. Pure NumPy — no torch, runnable and testable off the pod.

WHY A SEPARATE MODULE FROM scripts/utils/metrics.py
  That module reports a MULTI-LABEL problem: per-label AUROC at independent thresholds, with a mask
  for uncertain cells. Those numbers are still meaningful here (one-vs-rest AUROC is the standard
  way to report imbalanced multi-class), but three things it cannot express are exactly the things
  an imbalanced 7-class problem is judged on:

    * Accuracy computed from the ARGMAX prediction — the single decision the model actually makes.
    * Balanced accuracy — the mean of per-class recalls. On HAM10000, where `nv` is ~67% of the
      data, plain accuracy is dominated by one class: a model that predicts `nv` for everything
      scores ~0.67 and has learned nothing. Balanced accuracy scores that model 1/7 ≈ 0.14.
    * A confusion matrix — which classes are being mistaken for which. With mutually exclusive
      classes this is the primary diagnostic artifact, and it has no multi-label analogue.

  scripts/utils/metrics.py is left untouched; the CheXpert path keeps reporting what it reported.
"""

from __future__ import annotations

import numpy as np

from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS
from scripts.utils.metrics import auroc, average_precision


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """(n_classes, n_classes) integer counts, rows = true class, columns = predicted class."""
    n_classes = n_classes if n_classes is not None else len(CLASSIFIER_TARGET_LABELS)
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_index, predicted_index in zip(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)):
        matrix[true_index, predicted_index] += 1
    return matrix


def per_class_precision_recall_f1(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Per-class precision, recall and F1 derived from a confusion matrix.

    TWO DIFFERENT KINDS OF "no value", kept distinct on purpose:

      class ABSENT from the truth (no true instances at all)
          -> NaN, and excluded from every macro average. Nothing was evaluated, so there is no
             score; averaging a 0.0 here would punish a model for a class the split never
             contained.

      class PRESENT but NEVER PREDICTED
          -> precision 0.0, recall 0.0, F1 0.0, and INCLUDED in the macro average. This is a real,
             total failure on a real class. Scoring it NaN and dropping it would compute macro-F1
             only over the classes the model bothered to predict — which flatters a degenerate
             majority-class predictor in exactly the way macro-F1 exists to prevent. On HAM10000,
             where a model can reach ~67% accuracy by always answering `nv`, that distinction is
             the whole point of reporting these metrics at all.
    """
    matrix = np.asarray(matrix)
    true_positive = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    actual = matrix.sum(axis=1).astype(np.float64)

    n_classes = matrix.shape[0]
    precision = np.full(n_classes, np.nan, dtype=np.float64)
    recall = np.full(n_classes, np.nan, dtype=np.float64)
    f1 = np.full(n_classes, np.nan, dtype=np.float64)

    for index in range(n_classes):
        if actual[index] == 0 and predicted[index] == 0:
            continue  # class absent from this evaluation entirely -> stays NaN

        precision[index] = true_positive[index] / predicted[index] if predicted[index] > 0 else 0.0
        recall[index] = true_positive[index] / actual[index] if actual[index] > 0 else np.nan

        if np.isfinite(recall[index]):
            denominator = precision[index] + recall[index]
            f1[index] = (2 * precision[index] * recall[index] / denominator) if denominator > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def balanced_accuracy(matrix: np.ndarray) -> float:
    """Mean of the per-class recalls, over classes that are actually present.

    The headline metric for this dataset: unlike plain accuracy it cannot be inflated by getting the
    one dominant class right.
    """
    recall = per_class_precision_recall_f1(matrix)["recall"]
    usable = recall[np.isfinite(recall)]
    return float(usable.mean()) if usable.size else float("nan")


def macro_f1(matrix: np.ndarray) -> float:
    """Unweighted mean F1 across classes present in the truth — every class counts the same,
    regardless of how rare it is."""
    f1 = per_class_precision_recall_f1(matrix)["f1"]
    usable = f1[np.isfinite(f1)]
    return float(usable.mean()) if usable.size else float("nan")


def one_vs_rest_auroc(probabilities: np.ndarray, y_true: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """Per-class one-vs-rest AUROC from the softmax probability matrix.

    NaN for any class with no positives or no negatives present — an undefined AUROC, never a silent
    0.5, so it can be excluded from a macro-average rather than dragging it toward chance.
    """
    n_classes = n_classes if n_classes is not None else probabilities.shape[1]
    y_true = np.asarray(y_true, dtype=int)
    return np.array(
        [auroc(probabilities[:, index], (y_true == index).astype(np.int64)) for index in range(n_classes)],
        dtype=np.float64,
    )


def one_vs_rest_average_precision(probabilities: np.ndarray, y_true: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """Per-class one-vs-rest average precision (PR-AUC) — the more informative curve for the rare
    classes, where a large negative pool makes AUROC look flattering."""
    n_classes = n_classes if n_classes is not None else probabilities.shape[1]
    y_true = np.asarray(y_true, dtype=int)
    return np.array(
        [average_precision(probabilities[:, index], (y_true == index).astype(np.int64)) for index in range(n_classes)],
        dtype=np.float64,
    )


# Contract §12, decided by Walaa on 2026-10-04 before any v2 result exists: top-label ECE with 15
# equal-width bins.
ECE_BINS = 15


def macro_average_precision(probabilities: np.ndarray, y_true: np.ndarray, n_classes: int | None = None) -> float:
    """Mean of the one-vs-rest average precisions; a class with no positives is left out, not scored 0."""
    values = one_vs_rest_average_precision(probabilities, y_true, n_classes)
    usable = values[np.isfinite(values)]
    return float(usable.mean()) if usable.size else float("nan")


def multiclass_brier_score(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    """Mean over images of the squared distance between the probability vector and the one-hot
    truth, summed over classes. 0 is perfect, 2 is certain and wrong; lower is better."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)
    if len(y_true) == 0:
        return float("nan")
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(((probabilities - one_hot) ** 2).sum(axis=1).mean())


def top_label_ece(probabilities: np.ndarray, y_true: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """Expected calibration error of the decision the model makes: confidence is the highest
    probability, correctness is whether its class is the true one. Equal-width bins on [0, 1],
    each (lower, upper], the first one closed at 0; lower is better."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)
    if len(y_true) == 0:
        return float("nan")
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lower) & (confidence <= upper) if lower > 0 else (confidence >= lower) & (confidence <= upper)
        if in_bin.any():
            error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def full_metric_suite(probabilities: np.ndarray, y_true: np.ndarray, labels: list[str] | None = None) -> dict:
    """Every HAM10000 metric for one condition, with per-class support attached.

    `probabilities` is the (n_images, n_classes) softmax matrix; `y_true` the (n_images,) class
    indices. The predicted class is the argmax — the decision the model actually makes — so the
    confusion matrix and the accuracy metrics describe real behaviour rather than a thresholding
    convention chosen after the fact.
    """
    labels = labels if labels is not None else CLASSIFIER_TARGET_LABELS
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = probabilities.argmax(axis=1)

    matrix = confusion_matrix(y_true, y_pred, len(labels))
    per_class = per_class_precision_recall_f1(matrix)
    auroc_values = one_vs_rest_auroc(probabilities, y_true, len(labels))
    ap_values = one_vs_rest_average_precision(probabilities, y_true, len(labels))
    support = matrix.sum(axis=1)

    usable_auroc = auroc_values[np.isfinite(auroc_values)]

    return {
        "accuracy": float((y_pred == y_true).mean()) if len(y_true) else float("nan"),
        "balanced_accuracy": balanced_accuracy(matrix),
        "macro_f1": macro_f1(matrix),
        "macro_auroc_ovr": float(usable_auroc.mean()) if usable_auroc.size else float("nan"),
        "confusion_matrix": matrix.tolist(),
        "labels": list(labels),
        "per_class": {
            label: {
                "precision": float(per_class["precision"][index]),
                "recall": float(per_class["recall"][index]),
                "f1": float(per_class["f1"][index]),
                "auroc_ovr": float(auroc_values[index]),
                "average_precision": float(ap_values[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "n_images": int(len(y_true)),
    }


def format_confusion_matrix(matrix: np.ndarray, labels: list[str] | None = None) -> str:
    """Human-readable confusion matrix for a terminal or a log."""
    labels = labels if labels is not None else CLASSIFIER_TARGET_LABELS
    matrix = np.asarray(matrix)
    header = "true\\pred" + "".join(f"{label:>7s}" for label in labels)
    lines = [header]
    for index, label in enumerate(labels):
        lines.append(f"{label:>9s}" + "".join(f"{int(value):7d}" for value in matrix[index]))
    return "\n".join(lines)


__all__ = [
    "confusion_matrix",
    "per_class_precision_recall_f1",
    "balanced_accuracy",
    "macro_f1",
    "one_vs_rest_auroc",
    "one_vs_rest_average_precision",
    "full_metric_suite",
    "format_confusion_matrix",
]
