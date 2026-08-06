"""Structured label-to-text caption builder for CheXpert chest X-rays (docs/stage1_plan.md §7).

This module is the single source of truth for turning a CheXpert label row into a text prompt.
It is imported verbatim by both the Stage 1 training data pipeline (scripts/data/04_generate_captions.py)
and, later, Stage 2's generation pipeline — any drift between train-time and generation-time
phrasing directly degrades conditioning fidelity, so nothing caption-related should be
reimplemented elsewhere.

Design decisions encoded here (see docs/stage1_plan.md §7 for full rationale):
- Uncertain (-1) labels are OMITTED from the finding clause, never asserted present or absent.
- "No Finding" == 1 is authoritative over any spuriously co-positive pathology columns.
- "Support Devices" is a device, not a pathology, and gets its own clause.
- Age is decade-bucketed to reduce caption vocabulary sparsity.
- A small set of paraphrase templates is selected deterministically (hash of image id + epoch),
  not by an unseeded RNG, so caption generation is reproducible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

# Canonical order for serializing multi-label findings. Must match configs/dataset_config.yaml's
# `schema.pathology_columns` — kept as a plain constant here (rather than loaded from YAML) so this
# module has no config-parsing dependency and can be imported standalone by Stage 2.
PATHOLOGY_COLUMNS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

DEVICE_COLUMN = "Support Devices"
NO_FINDING_COLUMN = "No Finding"

# Human-readable phrasing for each pathology column (lowercase, fit for "Findings: X, Y, and Z.")
FINDING_TEXT = {
    "Enlarged Cardiomediastinum": "an enlarged cardiomediastinum",
    "Cardiomegaly": "cardiomegaly",
    "Lung Opacity": "lung opacity",
    "Lung Lesion": "a lung lesion",
    "Edema": "pulmonary edema",
    "Consolidation": "consolidation",
    "Pneumonia": "pneumonia",
    "Atelectasis": "atelectasis",
    "Pneumothorax": "pneumothorax",
    "Pleural Effusion": "pleural effusion",
    "Pleural Other": "other pleural abnormality",
    "Fracture": "a fracture",
}

NO_FINDING_PHRASE = "no acute cardiopulmonary abnormality"
DEVICE_PHRASE = "Support devices are present."

# Paraphrase templates (plan §7: 3-5 variants, config default 4). All must use exactly the
# {view}, {age_bucket}, {sex}, {finding_clause} placeholders; {device_clause} is appended
# separately by build_caption(), never interpolated inside these.
TEMPLATES = [
    "A {view} chest X-ray of a {age_bucket} {sex} patient. Findings: {finding_clause}.",
    "Chest radiograph, {view} view, {age_bucket} {sex} patient. Findings: {finding_clause}.",
    "A {view} view chest X-ray from a {age_bucket} {sex} patient showing {finding_clause}.",
    "{age_bucket} {sex} patient, {view} chest X-ray. Findings: {finding_clause}.",
]


@dataclass(frozen=True)
class CaptionConfig:
    age_bucket_width_years: int = 10
    uncertain_label_policy: str = "omit"  # only "omit" is implemented (plan §7 decision)
    no_finding_overrides_positives: bool = True
    num_paraphrase_variants: int = 4
    template_version: str = "v1"


def bucket_age(age, bucket_width_years: int = 10) -> str:
    """Decade-bucket a numeric age into a phrase like '40-49-year-old'."""
    try:
        age_int = int(float(age))
    except (TypeError, ValueError):
        return "adult"
    if age_int <= 0:
        return "adult"
    low = (age_int // bucket_width_years) * bucket_width_years
    high = low + bucket_width_years - 1
    return f"{low}-{high}-year-old"


def sex_phrase(sex_value) -> str:
    if sex_value is None:
        return "patient"
    s = str(sex_value).strip().lower()
    if s.startswith("m"):
        return "male"
    if s.startswith("f"):
        return "female"
    return "patient"


def view_phrase(frontal_lateral_value, ap_pa_value=None) -> str:
    fl = str(frontal_lateral_value).strip().lower() if frontal_lateral_value is not None else "frontal"
    if fl.startswith("lat"):
        return "lateral"
    return "frontal"


def serialize_natural_list(items: list[str]) -> str:
    """Join items with natural 'X', 'X and Y', or 'X, Y, and Z' grammar (plan §7: more stable
    CLIP embeddings than a raw comma token-dump)."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _label_value(row: dict, column: str):
    """Normalize a raw CheXpert label cell to one of {-1, 0, 1, None}."""
    value = row.get(column)
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return int(f)


def build_finding_clause(row: dict, config: CaptionConfig = CaptionConfig()) -> str:
    """Build the 'Findings: ...' clause per the uncertainty and No-Finding policies in plan §7."""
    if config.uncertain_label_policy != "omit":
        raise NotImplementedError(
            f"uncertain_label_policy={config.uncertain_label_policy!r} not implemented; "
            "the plan's chosen policy is 'omit' — see docs/stage1_plan.md §7."
        )

    no_finding = _label_value(row, NO_FINDING_COLUMN)
    positive_pathologies = [
        col
        for col in PATHOLOGY_COLUMNS
        if col not in (NO_FINDING_COLUMN, DEVICE_COLUMN) and _label_value(row, col) == 1
    ]

    if config.no_finding_overrides_positives and no_finding == 1:
        # A positive "No Finding" is treated as authoritative over any spuriously co-positive
        # pathology columns (known NLP-label-extraction artifact) — see plan §7.
        return NO_FINDING_PHRASE

    if not positive_pathologies:
        if no_finding == 1:
            return NO_FINDING_PHRASE
        # No positive findings and No Finding not asserted (e.g. only uncertain/negative labels):
        # still emit the normal-chest phrasing rather than an empty clause.
        return NO_FINDING_PHRASE

    phrases = [FINDING_TEXT[col] for col in positive_pathologies]
    return serialize_natural_list(phrases)


def build_device_clause(row: dict) -> str:
    if _label_value(row, DEVICE_COLUMN) == 1:
        return DEVICE_PHRASE
    return ""


def select_variant_index(image_id: str, epoch: int, num_variants: int) -> int:
    """Deterministic (not process-RNG-dependent) paraphrase variant selection, so caption
    generation is reproducible across runs given the same image id and epoch (plan §7)."""
    key = f"{image_id}:{epoch}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest, 16) % num_variants


def build_caption(
    row: dict,
    config: CaptionConfig = CaptionConfig(),
    variant_index: int = 0,
) -> str:
    """Build a single caption string from a CheXpert label row.

    `row` is expected to expose (at minimum) the columns listed in
    configs/dataset_config.yaml `schema`: Sex, Age, Frontal/Lateral, AP/PA, and the 14
    pathology columns in PATHOLOGY_COLUMNS, with ternary values in {-1, 0, 1, NaN/None}.
    """
    template = TEMPLATES[variant_index % len(TEMPLATES)]
    view = view_phrase(row.get("Frontal/Lateral"), row.get("AP/PA"))
    age_bucket = bucket_age(row.get("Age"), config.age_bucket_width_years)
    sex = sex_phrase(row.get("Sex"))
    finding_clause = build_finding_clause(row, config)
    device_clause = build_device_clause(row)

    caption = template.format(
        view=view,
        age_bucket=age_bucket,
        sex=sex,
        finding_clause=finding_clause,
    )
    if device_clause:
        caption = f"{caption} {device_clause}"
    return caption


def build_caption_variants(row: dict, config: CaptionConfig = CaptionConfig()) -> list[str]:
    """Return all configured paraphrase variants for a row (useful for prompt-cache precomputation,
    plan §8, where every variant an image might use during training is embedded once up front)."""
    n = min(config.num_paraphrase_variants, len(TEMPLATES))
    return [build_caption(row, config, variant_index=i) for i in range(n)]
