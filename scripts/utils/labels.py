"""Frozen CheXpert label policy, shared by every Stage 2-5 script
(docs/stages2_to_5_plan.md §5 and §6).

Three label sets, deliberately not interchangeable:

  CLASSIFIER_TARGET_LABELS (14)  - the classifier's output space. Everything is predicted and
                                   reported.
  PRIMARY_ENDPOINT_LABELS  (11)  - the disease labels the PRIMARY endpoint (macro-AUROC) is
                                   computed over. Excludes "No Finding" (an absence-of-disease
                                   meta-label, not a pathology) and "Support Devices" (a device
                                   label, and the highest-prevalence/easiest column in CheXpert —
                                   including it would flatter the macro-average without measuring
                                   diagnostic performance). Also excludes "Pleural Other"
                                   (docs/stages2_to_5_plan.md §5.2 revision note): on the full
                                   production cohort only ~100 patients dataset-wide carry a
                                   confident negative label for it, so no split-fraction choice can
                                   give every decision-bearing split the frozen ≥50-negative-patient
                                   support rule (§1.3) — unlike "Lung Lesion"/"Atelectasis", which
                                   were fixed by revising configs/splits.yaml fractions instead. All
                                   three remain secondary outcomes.
  GENERATION_TARGET_LABELS (12)  - PRIMARY_ENDPOINT_LABELS plus INSUFFICIENT_SUPPORT_LABELS
                                   (docs/stages2_to_5_plan.md §3 revision note). Real-data support
                                   scarcity is a property of what can be reliably EVALUATED on the
                                   real cohort; it is not a reason to also stop generating or
                                   training on synthetic examples of the condition. Stage 2 recipe
                                   eligibility (co-occurrence mining, quotas, captions) keys off
                                   this list, not PRIMARY_ENDPOINT_LABELS. ASISM's agreement
                                   scoring (§4.5) and Stage 4 condition D's marginal matching (§7)
                                   deliberately stay scoped to PRIMARY_ENDPOINT_LABELS only.

Uncertainty policy (plan §6): raw labels are preserved; -1 and blank are MASKED out of loss and
metrics, never silently mapped to 0 or 1. Three quantities are always kept separate:
`raw_label`, `training_target`, and `loss_mask`/`eval_mask`.

`caption_builder.PATHOLOGY_COLUMNS` is the same 14 columns in the same canonical order; it is
imported here rather than re-typed so the two modules cannot drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.utils.caption_builder import (
    DEVICE_COLUMN,
    NO_FINDING_COLUMN,
    PATHOLOGY_COLUMNS,
)

# The classifier's full output space: all 14 CheXpert observations, canonical order.
CLASSIFIER_TARGET_LABELS: list[str] = list(PATHOLOGY_COLUMNS)

# Excluded from the primary endpoint due to insufficient patient-level negative support under the
# frozen §1.3 support rule (measured on the full production cohort: see module docstring). Reported
# as a secondary outcome with its support counts (plan §5.3), same treatment as No Finding /
# Support Devices.
INSUFFICIENT_SUPPORT_LABELS: list[str] = ["Pleural Other"]

# The 11 disease labels the primary endpoint is computed over (plan §5.2).
PRIMARY_ENDPOINT_LABELS: list[str] = [
    column
    for column in CLASSIFIER_TARGET_LABELS
    if column not in (NO_FINDING_COLUMN, DEVICE_COLUMN, *INSUFFICIENT_SUPPORT_LABELS)
]

# Reported, but never part of the primary macro-average.
SECONDARY_LABELS: list[str] = [NO_FINDING_COLUMN, DEVICE_COLUMN, *INSUFFICIENT_SUPPORT_LABELS]

# Every disease label worth intentionally synthesizing in Stage 2, whether or not it currently has
# enough REAL-data patient support to be scored as a primary endpoint (module docstring).
GENERATION_TARGET_LABELS: list[str] = [*PRIMARY_ENDPOINT_LABELS, *INSUFFICIENT_SUPPORT_LABELS]

# Sentinel used in integer label arrays for "uncertain or not mentioned" (i.e. masked).
MASKED = -1


def normalize_label(value) -> int | None:
    """Normalize one raw CheXpert cell to exactly one of {1, 0, -1, None}.

    None means blank/not-mentioned. -1 means the radiologist recorded uncertainty. Both are masked
    downstream, but they are distinct facts and are kept distinct in `raw_label`.
    """
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid CheXpert label value {value!r}; expected -1, 0, 1, or blank") from exc
    if numeric != numeric:  # NaN
        return None
    if numeric not in (-1.0, 0.0, 1.0):
        raise ValueError(f"invalid CheXpert label value {value!r}; expected -1, 0, 1, or blank")
    return int(numeric)


def validate_label_domain(frame: pd.DataFrame, labels: list[str], identifier_column: str = "Path") -> None:
    """Validate labels with enough row context to diagnose a bad source cohort."""
    for label in labels:
        if label not in frame.columns:
            raise ValueError(f"missing required CheXpert label column {label!r}")
        for index, value in frame[label].items():
            try:
                normalize_label(value)
            except ValueError as exc:
                identifier = frame.at[index, identifier_column] if identifier_column in frame else index
                raise ValueError(f"label={label!r}, value={value!r}, row/image={identifier!r}: {exc}") from exc


def build_label_arrays(
    frame: pd.DataFrame,
    labels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (raw_label, training_target, mask) as three aligned (n_rows, n_labels) arrays.

    raw_label       int8, values in {1, 0, -1, -2}; -2 encodes blank/not-mentioned so the raw
                    array stays integer while remaining distinguishable from an explicit -1.
    training_target float32, values in {0.0, 1.0}; positions where mask is False are 0.0 and MUST
                    NOT be read (they are meaningless placeholders, not negatives).
    mask            bool, True only where the label is confidently 0 or 1.

    The mask is what enforces plan §6: -1 and blank contribute to neither loss nor metrics, and are
    never converted into a 0 or a 1 that the model would then be trained to reproduce.
    """
    labels = labels if labels is not None else CLASSIFIER_TARGET_LABELS
    n_rows, n_labels = len(frame), len(labels)

    raw = np.full((n_rows, n_labels), -2, dtype=np.int8)
    target = np.zeros((n_rows, n_labels), dtype=np.float32)
    mask = np.zeros((n_rows, n_labels), dtype=bool)

    for column_index, label in enumerate(labels):
        if label not in frame.columns:
            # Column entirely absent: leave as blank/masked rather than inventing negatives.
            continue
        values = [normalize_label(value) for value in frame[label].to_numpy()]
        for row_index, value in enumerate(values):
            if value is None:
                continue
            raw[row_index, column_index] = value
            if value in (0, 1):
                target[row_index, column_index] = float(value)
                mask[row_index, column_index] = True

    return raw, target, mask


def patient_level_support(
    frame: pd.DataFrame,
    labels: list[str] | None = None,
    patient_column: str = "patient_id",
) -> pd.DataFrame:
    """Per-label positive/negative/masked PATIENT counts (plan §1.3 — patient counts, not image
    counts, determine primary support eligibility).

    A patient counts as positive for a label if ANY of their studies carries a confident 1, and as
    negative if they are not positive and any study carries a confident 0. Patients whose every
    study is uncertain/blank for that label count as masked-only and support neither side.

    Returns one row per label with columns:
    label, positive_patients, negative_patients, masked_only_patients, positive_images,
    negative_images, total_patients.
    """
    labels = labels if labels is not None else PRIMARY_ENDPOINT_LABELS
    if patient_column not in frame.columns:
        raise KeyError(
            f"patient_level_support requires a {patient_column!r} column; got {list(frame.columns)[:8]}..."
        )

    records = []
    total_patients = frame[patient_column].nunique()

    for label in labels:
        if label not in frame.columns:
            records.append(
                {
                    "label": label,
                    "positive_patients": 0,
                    "negative_patients": 0,
                    "masked_only_patients": total_patients,
                    "positive_images": 0,
                    "negative_images": 0,
                    "total_patients": total_patients,
                }
            )
            continue

        normalized = frame[label].map(normalize_label)
        is_positive = normalized == 1
        is_negative = normalized == 0

        positive_patients = set(frame.loc[is_positive, patient_column].unique())
        negative_candidates = set(frame.loc[is_negative, patient_column].unique())
        # "Any positive" wins: a patient with one positive study is a positive patient.
        negative_patients = negative_candidates - positive_patients
        masked_only = total_patients - len(positive_patients) - len(negative_patients)

        records.append(
            {
                "label": label,
                "positive_patients": len(positive_patients),
                "negative_patients": len(negative_patients),
                "masked_only_patients": masked_only,
                "positive_images": int(is_positive.sum()),
                "negative_images": int(is_negative.sum()),
                "total_patients": total_patients,
            }
        )

    return pd.DataFrame.from_records(records)


def check_support_rule(
    support: pd.DataFrame,
    min_positive_patients: int,
    min_negative_patients: int,
) -> tuple[bool, list[dict]]:
    """Apply the frozen §1.3 support rule to a `patient_level_support` frame.

    Returns (passed, failures), where each failure records the label and both observed counts so
    the caller can emit a complete report rather than a first-failure message.
    """
    failures = []
    for row in support.to_dict("records"):
        positive_ok = row["positive_patients"] >= min_positive_patients
        negative_ok = row["negative_patients"] >= min_negative_patients
        if not (positive_ok and negative_ok):
            failures.append(
                {
                    "label": row["label"],
                    "positive_patients": row["positive_patients"],
                    "negative_patients": row["negative_patients"],
                    "min_positive_patients": min_positive_patients,
                    "min_negative_patients": min_negative_patients,
                    "positive_ok": positive_ok,
                    "negative_ok": negative_ok,
                }
            )
    return (len(failures) == 0), failures


def intended_vector_to_labels(intended: dict[str, int]) -> list[str]:
    """The primary disease labels a Stage 2 recipe intends to be positive.

    A `No Finding` recipe is the all-zero intended vector over the 11 primary labels (plan §3), so
    this correctly returns an empty list for it — agreement (§4.5) then scores it as "all 11
    primary predicted probabilities should be low."
    """
    return [label for label in PRIMARY_ENDPOINT_LABELS if int(intended.get(label, 0)) == 1]
