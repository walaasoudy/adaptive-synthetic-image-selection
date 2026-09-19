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


GENERATION_SOURCE_SPLIT = "gen_train"


def quotas_from_gen_train(split_dir, **quota_kwargs) -> tuple[dict[str, int], dict[str, int]]:
    """(quotas, real class counts) derived from `<split_dir>/gen_train.csv` and NOTHING else.

    Quotas are a generation decision, so they may only see the split the generator is trained on.
    Deriving them from the full metadata — as the first smoke test did — lets validation and
    final-eval class frequencies shape what gets generated. The numerical effect happens to be
    small for HAM10000 (quotas depend only on class ratios), but the rule is not negotiable on the
    size of the leak. Reads by explicit filename, so no other split file is ever opened.

    Fails closed on a class with zero gen_train images: rarity_quotas would otherwise assign it
    max_per_class, i.e. ask the generator for many images of a class it never saw.
    """
    import pandas as pd
    from pathlib import Path

    path = Path(split_dir) / f"{GENERATION_SOURCE_SPLIT}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"generation quotas require the frozen {GENERATION_SOURCE_SPLIT} split: {path}")
    counts = class_counts(pd.read_csv(path))
    empty = [label for label, count in counts.items() if count == 0]
    if empty:
        raise ValueError(f"{GENERATION_SOURCE_SPLIT} has no real images for {empty}; refusing to assign generation quotas")
    return rarity_quotas(counts, **quota_kwargs), counts


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


def derive_image_seed(base_seed: int, recipe_id: str) -> int:
    """Deterministic per-image seed from (base seed, recipe id) — the CheXpert Stage 2 rule, so a
    resumed or re-run generation reproduces exactly the image an uninterrupted run would have."""
    import hashlib

    return int(hashlib.sha256(f"{int(base_seed)}:{recipe_id}".encode("utf-8")).hexdigest()[:8], 16)


RECIPE_COLUMNS = [
    "recipe_id", "dx", "class_index", "intended_label_vector", "prompt", "caption_variant_index",
    "context_image_id", "age", "sex", "localization", "seed", "source_split",
]


def build_contextual_recipes(gen_train, quotas: dict[str, int], recipe_cfg, base_seed: int, captions_cfg):
    """Stage 2 recipes whose prompts use the SAME caption template the LoRA was trained on.

    Each recipe is one diagnosis. Its non-diagnostic context (age, sex, body site) is copied from a
    real gen_train image OF THE SAME CLASS, drawn with a class-seeded RNG, so the prompt distribution
    matches what the generator saw during training instead of a bare "an image of X" it never saw.
    The paraphrase variant is chosen by the same hash rule as training. Only gen_train rows are ever
    passed in; the caller proves that (quotas_from_gen_train + id isolation).
    """
    import json

    import pandas as pd

    from scripts.utils.ham10000 import build_caption, select_variant_index

    frame = gen_train.copy()
    frame["dx_norm"] = frame["dx"].map(normalize_diagnosis)
    rows = []
    for class_position, label in enumerate(CLASSIFIER_TARGET_LABELS):
        pool = frame[frame["dx_norm"] == label].sort_values("image_id").reset_index(drop=True)
        count = int(quotas.get(label, 0))
        if count and pool.empty:
            raise ValueError(f"no gen_train images of {label!r} to draw recipe context from")
        rng = np.random.default_rng([int(recipe_cfg.context_seed), class_position])
        picks = rng.integers(0, len(pool), size=count) if count else []
        for index, pick in enumerate(picks):
            context = pool.iloc[int(pick)].to_dict()
            recipe_id = f"syn_{label}_{index:05d}"
            variant = select_variant_index(recipe_id, 0, int(captions_cfg.num_paraphrase_variants))
            rows.append(
                {
                    "recipe_id": recipe_id,
                    "dx": label,
                    "class_index": class_position,
                    "intended_label_vector": json.dumps({name: int(name == label) for name in CLASSIFIER_TARGET_LABELS}),
                    "prompt": build_caption({**context, "dx": label}, variant, int(captions_cfg.age_bucket_width_years)),
                    "caption_variant_index": variant,
                    "context_image_id": str(context["image_id"]),
                    "age": context.get("age"),
                    "sex": context.get("sex"),
                    "localization": context.get("localization"),
                    "seed": derive_image_seed(base_seed, recipe_id),
                    "source_split": GENERATION_SOURCE_SPLIT,
                }
            )
    return pd.DataFrame(rows, columns=RECIPE_COLUMNS)


__all__ = [
    "class_counts", "rarity_quotas", "quotas_from_gen_train", "build_recipes", "GENERATION_SOURCE_SPLIT",
    "derive_image_seed", "build_contextual_recipes", "RECIPE_COLUMNS",
]
