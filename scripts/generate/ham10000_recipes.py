"""Stage 2 label recipes for HAM10000: per-class quotas, no label combinations.

WHY THE CheXpert RECIPE SAMPLER DOES NOT TRANSFER
  scripts/generate/01_sample_label_recipes.py mines CO-OCCURRENCE: which SETS of findings appear
  together in enough real patients to be worth generating (cardiomegaly + pleural effusion, and so
  on). Its knobs — `min_support_patients`, `single_label_proportion`, `multi_label_proportion`,
  `max_positive_labels` — all describe how many findings to put in one image.

  HAM10000's classes are mutually exclusive. There are no combinations to mine: the only valid
  recipes are the seven single classes, and every one of those knobs is meaningless here. Running
  the CheXpert sampler would either find no combinations at all or, worse, invent multi-diagnosis
  recipes that cannot exist and ask the generator to synthesise them.

  What remains is the part that actually matters for this thesis: HOW MANY images to request per
  class. That is a quota problem driven by real scarcity, and it is all this module does.
"""

from __future__ import annotations

import numpy as np

from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS, DIAGNOSIS_PHRASE, normalize_diagnosis


def class_counts(frame, diagnosis_column: str = "dx") -> dict[str, int]:
    """Real image count per class in a split."""
    normalized = frame[diagnosis_column].map(normalize_diagnosis)
    return {label: int((normalized == label).sum()) for label in CLASSIFIER_TARGET_LABELS}


def rarity_quotas(
    counts: dict[str, int],
    base_quota: int = 400,
    rarity_exponent: float = 0.5,
    min_per_class: int = 100,
    max_per_class: int = 1500,
) -> dict[str, int]:
    """How many synthetic candidates to generate per class, oversampling the rare ones.

        target = base_quota * (median_count / class_count) ** rarity_exponent,  clipped

    `rarity_exponent = 0.5` deliberately UNDER-compensates: full inverse-frequency weighting
    (exponent 1.0) would ask for roughly 58x more dermatofibroma than naevus images, far past the
    point where a LoRA trained on ~115 real examples can produce anything but near-copies of them.
    Square-root scaling asks for more of what is scarce without pretending the generator can
    manufacture diversity it never saw. Set the exponent to 0 to disable oversampling entirely.

    These are CANDIDATE counts, not selected counts: ASISM decides afterwards which of them are
    worth training on, which is the entire point of the thesis.
    """
    present = [count for count in counts.values() if count > 0]
    if not present:
        raise ValueError("no class has any real images; cannot derive quotas")
    median = float(np.median(present))

    quotas = {}
    for label in CLASSIFIER_TARGET_LABELS:
        count = counts.get(label, 0)
        if count <= 0:
            quotas[label] = int(max_per_class)
            continue
        scaled = base_quota * (median / count) ** rarity_exponent
        quotas[label] = int(np.clip(round(scaled), min_per_class, max_per_class))
    return quotas


def build_recipes(quotas: dict[str, int], seed: int = 42) -> list[dict]:
    """One recipe per requested image. Each carries exactly one diagnosis — never a combination.

    `intended_label_vector` is the one-hot dict ASISM's agreement signal consumes, so a recipe is
    shaped identically to a CheXpert one and nothing downstream needs a special case; it simply
    always has exactly one positive.
    """
    rng = np.random.default_rng(seed)
    recipes = []
    for label in CLASSIFIER_TARGET_LABELS:
        for index in range(int(quotas.get(label, 0))):
            recipes.append(
                {
                    "recipe_id": f"{label}_{index:05d}",
                    "diagnosis": label,
                    "intended_label_vector": {name: int(name == label) for name in CLASSIFIER_TARGET_LABELS},
                    "prompt": f"A dermoscopic image of {DIAGNOSIS_PHRASE[label]}.",
                    "seed": int(rng.integers(0, 2**31 - 1)),
                }
            )
    return recipes


__all__ = ["class_counts", "rarity_quotas", "build_recipes"]
