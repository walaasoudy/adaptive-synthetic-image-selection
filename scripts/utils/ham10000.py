"""HAM10000 domain layer: the 7-class label policy, one-hot encoding, and dermoscopy captions.

This is the dermoscopy counterpart of scripts/utils/labels.py + scripts/utils/caption_builder.py,
and it is deliberately ONE module rather than two: HAM10000's label policy is small enough that
splitting it would create two files that must be kept in lockstep for no benefit.

SINGLE-LABEL MULTI-CLASS, AND WHY THAT IS NOT CheXpert'S SHAPE
-------------------------------------------------------------
CheXpert carries 14 INDEPENDENT observations per image, each {1, 0, -1, blank}: one radiograph can
be positive for several findings at once, and "uncertain" is a real recorded third state. The
pipeline it was built for therefore uses independent per-label sigmoids and a masked
binary-cross-entropy loss.

HAM10000 is a different problem: exactly ONE diagnosis out of 7, mutually exclusive, never
uncertain. The classifier for this dataset is therefore SOFTMAX + CROSS-ENTROPY, not sigmoid + BCE.
That is a methodological choice, not a performance one: independent sigmoids can assign
P(mel)=0.8 and P(nv)=0.7 to the same image, a state the ground truth can never occupy, and
calibration and threshold selection both inherit that incoherence. Softmax makes the seven
probabilities sum to 1 by construction, which is what the data actually says.

Two target encodings are exported, because downstream stages genuinely need different shapes:

  class_index_targets()  -> (n_images,) int64 of class indices.  THIS is what CrossEntropyLoss
                            takes, and what the classifier trains against.
  build_label_arrays()   -> (n_images, n_labels) one-hot + all-True mask.  Used by the EVALUATION
                            and ASISM layers, which are written against a per-label probability
                            matrix: one-vs-rest AUROC, per-class precision/recall, and ASISM's
                            agreement signal all need "probability of each class" rather than "the
                            argmax class". Softmax outputs slot into that shape unchanged.

The two are always derived from the same `dx` column and can never disagree; `one_hot_from_indices`
converts between them, and a test asserts the round-trip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Generic, dataset-independent caption helpers. Imported rather than re-typed so the deterministic
# paraphrase selection stays bit-identical to Stage 1's and cannot drift between the two datasets.
from scripts.utils.caption_builder import bucket_age, select_variant_index, sex_phrase

# ----------------------------------------------------------------------------------------------
# Label policy
# ----------------------------------------------------------------------------------------------

# Canonical order, most prevalent first. Must match configs/dataset_ham10000.yaml
# schema.diagnosis_classes. Once any artifact has been written this order is frozen: it is the
# column order of every target array, probability matrix and parquet on disk.
DIAGNOSIS_CLASSES: list[str] = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]

# The classifier's output space. Unlike CheXpert there is no meta-label ("No Finding") and no device
# label to exclude, so all three label sets coincide — every class is predicted, scored as a primary
# endpoint, and worth generating synthetically.
CLASSIFIER_TARGET_LABELS: list[str] = list(DIAGNOSIS_CLASSES)
PRIMARY_ENDPOINT_LABELS: list[str] = list(DIAGNOSIS_CLASSES)
GENERATION_TARGET_LABELS: list[str] = list(DIAGNOSIS_CLASSES)

# Full names, for reports and figure labels. Never used for conditioning — captions use
# DIAGNOSIS_PHRASE below, which is written to read naturally inside a sentence.
DIAGNOSIS_FULL_NAME: dict[str, str] = {
    "nv": "melanocytic nevus",
    "mel": "melanoma",
    "bkl": "benign keratosis-like lesion",
    "bcc": "basal cell carcinoma",
    "akiec": "actinic keratosis / intraepithelial carcinoma",
    "vasc": "vascular lesion",
    "df": "dermatofibroma",
}

# The classes whose real-data scarcity is the thesis's motivation for synthetic augmentation.
# Informational: nothing gates on this list, it exists so reports can mark the rare arm explicitly
# instead of leaving a reader to infer it from counts.
RARE_CLASSES: list[str] = ["bcc", "akiec", "vasc", "df"]


def normalize_diagnosis(value) -> str:
    """Normalize one raw `dx` cell to a known class code, or raise.

    Unlike CheXpert there is no blank/uncertain state to tolerate: a row whose diagnosis is missing
    or unrecognised is a broken row, not an unlabelled one, and silently dropping it would quietly
    change the cohort. Fail loudly instead.
    """
    if value is None:
        raise ValueError("missing HAM10000 dx value; expected one of " + ", ".join(DIAGNOSIS_CLASSES))
    text = str(value).strip().lower()
    if text not in DIAGNOSIS_CLASSES:
        raise ValueError(
            f"invalid HAM10000 dx value {value!r}; expected one of {', '.join(DIAGNOSIS_CLASSES)}"
        )
    return text


def validate_metadata(frame: pd.DataFrame, diagnosis_column: str = "dx", group_column: str = "lesion_id") -> None:
    """Validate the label and grouping columns with enough row context to diagnose a bad cohort."""
    for column in (diagnosis_column, group_column):
        if column not in frame.columns:
            raise ValueError(f"missing required HAM10000 column {column!r}; got {list(frame.columns)}")

    missing_group = frame[group_column].isna().sum()
    if missing_group:
        raise ValueError(
            f"{missing_group} row(s) have no {group_column!r}. Every image must be assignable to a "
            "lesion, or the split cannot guarantee that two photographs of one lesion stay on the "
            "same side."
        )

    for index, value in frame[diagnosis_column].items():
        try:
            normalize_diagnosis(value)
        except ValueError as exc:
            raise ValueError(f"row={index}: {exc}") from exc


def class_index_targets(
    frame: pd.DataFrame,
    labels: list[str] | None = None,
    diagnosis_column: str = "dx",
) -> np.ndarray:
    """Return (n_rows,) int64 class indices — the target CrossEntropyLoss takes.

    This is the classifier's training target. Index i corresponds to labels[i], so the ordering of
    DIAGNOSIS_CLASSES is load-bearing: it fixes which output neuron means which diagnosis, and must
    not be reordered once any checkpoint or probability artifact exists.
    """
    labels = labels if labels is not None else CLASSIFIER_TARGET_LABELS
    index_of = {label: position for position, label in enumerate(labels)}
    return np.array(
        [index_of[normalize_diagnosis(value)] for value in frame[diagnosis_column].to_numpy()],
        dtype=np.int64,
    )


def one_hot_from_indices(indices: np.ndarray, n_labels: int | None = None) -> np.ndarray:
    """(n_rows,) class indices -> (n_rows, n_labels) float32 one-hot.

    The single conversion point between the two encodings, so they cannot drift apart.
    """
    n_labels = n_labels if n_labels is not None else len(CLASSIFIER_TARGET_LABELS)
    indices = np.asarray(indices, dtype=np.int64)
    one_hot = np.zeros((len(indices), n_labels), dtype=np.float32)
    one_hot[np.arange(len(indices)), indices] = 1.0
    return one_hot


def build_label_arrays(
    frame: pd.DataFrame,
    labels: list[str] | None = None,
    diagnosis_column: str = "dx",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (raw_label, training_target, mask) as three aligned (n_rows, n_labels) arrays.

    The EVALUATION-side encoding. Same tuple shape as scripts/utils/labels.build_label_arrays, so
    the metric layer and ASISM's agreement signal — both written against a per-label probability
    matrix — need no special-casing:

      raw_label       int8 in {0, 1}; the one-hot row itself (no -1/-2 states exist here)
      training_target float32 one-hot
      mask            bool, all True — HAM10000 records one confident diagnosis per image, so no
                      cell is ever excluded from metrics

    NOT the classifier's training target: that is class_index_targets(), because the classifier
    uses CrossEntropy. Both are derived from the same `dx` column, so they cannot disagree.
    """
    labels = labels if labels is not None else CLASSIFIER_TARGET_LABELS
    index_of = {label: position for position, label in enumerate(labels)}

    n_rows, n_labels = len(frame), len(labels)
    raw = np.zeros((n_rows, n_labels), dtype=np.int8)
    target = np.zeros((n_rows, n_labels), dtype=np.float32)
    mask = np.ones((n_rows, n_labels), dtype=bool)

    for row_index, value in enumerate(frame[diagnosis_column].to_numpy()):
        column_index = index_of[normalize_diagnosis(value)]
        raw[row_index, column_index] = 1
        target[row_index, column_index] = 1.0

    return raw, target, mask


def one_hot_vector(diagnosis: str, labels: list[str] | None = None) -> dict[str, int]:
    """One diagnosis as the {label: 0/1} dict Stage 2 recipes and ASISM's agreement signal expect.

    This is the HAM10000 analogue of a CheXpert `intended_label_vector`: exactly one key is 1.
    """
    labels = labels if labels is not None else CLASSIFIER_TARGET_LABELS
    code = normalize_diagnosis(diagnosis)
    return {label: int(label == code) for label in labels}


def intended_vector_to_labels(intended: dict[str, int]) -> list[str]:
    """The classes a recipe intends to be positive — one element for a well-formed HAM10000 vector.

    Kept as a list (rather than a single string) so ASISM's agreement scoring, which is written
    against a list of intended positives, needs no special-casing for this dataset.
    """
    return [label for label in PRIMARY_ENDPOINT_LABELS if int(intended.get(label, 0)) == 1]


# ----------------------------------------------------------------------------------------------
# Lesion-level support (the analogue of CheXpert's patient-level support)
# ----------------------------------------------------------------------------------------------

def group_level_support(
    frame: pd.DataFrame,
    labels: list[str] | None = None,
    group_column: str = "lesion_id",
    diagnosis_column: str = "dx",
) -> pd.DataFrame:
    """Per-class positive/negative LESION counts — lesions, not images, decide eligibility.

    A lesion is positive for a class when any of its images carries that diagnosis (in practice all
    of them do: HAM10000's diagnosis is a property of the lesion, not of the individual photograph).
    Because the classes are mutually exclusive, a lesion that is positive for one class is a
    negative for all six others, so negatives are plentiful for every class and only the positive
    counts ever bind.
    """
    labels = labels if labels is not None else PRIMARY_ENDPOINT_LABELS
    if group_column not in frame.columns:
        raise KeyError(f"group_level_support requires a {group_column!r} column")

    normalized = frame[diagnosis_column].map(normalize_diagnosis)
    total_groups = frame[group_column].nunique()

    records = []
    for label in labels:
        positive_groups = set(frame.loc[normalized == label, group_column].unique())
        negative_groups = set(frame[group_column].unique()) - positive_groups
        records.append(
            {
                "label": label,
                "positive_groups": len(positive_groups),
                "negative_groups": len(negative_groups),
                "positive_images": int((normalized == label).sum()),
                "negative_images": int((normalized != label).sum()),
                "total_groups": total_groups,
            }
        )
    return pd.DataFrame.from_records(records)


def check_support_rule(
    support: pd.DataFrame,
    min_positive_groups: int,
    min_negative_groups: int,
) -> tuple[bool, list[dict]]:
    """Apply the frozen support rule to a `group_level_support` frame.

    Returns (passed, failures) and reports EVERY failing class, not just the first, so one run
    produces a complete picture of which classes a split cannot support.
    """
    failures = []
    for row in support.to_dict("records"):
        positive_ok = row["positive_groups"] >= min_positive_groups
        negative_ok = row["negative_groups"] >= min_negative_groups
        if not (positive_ok and negative_ok):
            failures.append(
                {
                    "label": row["label"],
                    "positive_groups": row["positive_groups"],
                    "negative_groups": row["negative_groups"],
                    "min_positive_groups": min_positive_groups,
                    "min_negative_groups": min_negative_groups,
                    "positive_ok": positive_ok,
                    "negative_ok": negative_ok,
                }
            )
    return (len(failures) == 0), failures


# ----------------------------------------------------------------------------------------------
# Dermoscopy captions
# ----------------------------------------------------------------------------------------------

# Phrasing chosen to sit naturally inside the templates below ("... an image of {phrase} on the
# ...") and to use the clinical term the literature uses, since that is what SDXL's text encoder has
# the best chance of having seen.
DIAGNOSIS_PHRASE: dict[str, str] = {
    "nv": "a melanocytic nevus",
    "mel": "a melanoma",
    "bkl": "a benign keratosis-like lesion",
    "bcc": "a basal cell carcinoma",
    "akiec": "an actinic keratosis",
    "vasc": "a vascular lesion",
    "df": "a dermatofibroma",
}

# Four paraphrase variants, matching the pipeline's configured num_paraphrase_variants. All four use
# exactly the {diagnosis}, {site_clause}, {age_bucket}, {sex} placeholders.
TEMPLATES = [
    "A dermoscopic image of {diagnosis}{site_clause} in a {age_bucket} {sex} patient.",
    "Dermoscopy of {diagnosis}{site_clause}, {age_bucket} {sex} patient.",
    "A {age_bucket} {sex} patient with {diagnosis}{site_clause}, dermoscopic view.",
    "Dermoscopic image, {age_bucket} {sex} patient, showing {diagnosis}{site_clause}.",
]

TEMPLATE_VERSION = "ham-v1"


def site_phrase(localization) -> str:
    """The body-site clause, or an empty string when the site is unrecorded.

    HAM10000 records localization as free-ish text with an explicit "unknown" value. An unknown site
    produces NO clause at all rather than the word "unknown": conditioning the generator on the
    literal token "unknown" would teach it that "unknown" is a body site.
    """
    if localization is None:
        return ""
    text = str(localization).strip().lower()
    if not text or text in {"unknown", "nan", "none"}:
        return ""
    return f" on the {text}"


def build_caption(row: dict, variant_index: int = 0, age_bucket_width_years: int = 10) -> str:
    """Build one caption from a HAM10000 metadata row.

    `row` must expose the columns named in configs/dataset_ham10000.yaml `schema`: dx, age, sex and
    localization. Age and localization may be missing — both degrade to a neutral phrasing rather
    than emitting a placeholder token into the prompt.
    """
    template = TEMPLATES[variant_index % len(TEMPLATES)]
    return template.format(
        diagnosis=DIAGNOSIS_PHRASE[normalize_diagnosis(row.get("dx"))],
        site_clause=site_phrase(row.get("localization")),
        age_bucket=bucket_age(row.get("age"), age_bucket_width_years),
        sex=sex_phrase(row.get("sex")),
    )


def build_caption_variants(
    row: dict,
    image_id: str,
    num_variants: int = 4,
    age_bucket_width_years: int = 10,
) -> list[str]:
    """All paraphrase variants for one image, in deterministic order.

    Stage 1 picks among these per epoch via caption_builder.select_variant_index, the same
    hash-based (never process-RNG) selection the CheXpert path uses, so caption assignment is
    reproducible across runs and machines.
    """
    return [
        build_caption(row, variant_index=index, age_bucket_width_years=age_bucket_width_years)
        for index in range(num_variants)
    ]


__all__ = [
    "DIAGNOSIS_CLASSES",
    "CLASSIFIER_TARGET_LABELS",
    "PRIMARY_ENDPOINT_LABELS",
    "GENERATION_TARGET_LABELS",
    "DIAGNOSIS_FULL_NAME",
    "DIAGNOSIS_PHRASE",
    "RARE_CLASSES",
    "TEMPLATE_VERSION",
    "normalize_diagnosis",
    "validate_metadata",
    "class_index_targets",
    "one_hot_from_indices",
    "build_label_arrays",
    "one_hot_vector",
    "intended_vector_to_labels",
    "group_level_support",
    "check_support_rule",
    "site_phrase",
    "build_caption",
    "build_caption_variants",
    "select_variant_index",
]
