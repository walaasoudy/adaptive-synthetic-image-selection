#!/usr/bin/env python3
"""HAM10000 data layer: label encodings, dermoscopy captions, lesion-level split and leakage audit.

The split tests are the important ones. A leakage guard that is only documented is not a guard, so
the audit is exercised against a DELIBERATELY leaked split and asserted to fail — passing on clean
input proves nothing on its own.
"""

from __future__ import annotations

import importlib.util
import sys

sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.utils.ham10000 import (  # noqa: E402
    CLASSIFIER_TARGET_LABELS,
    DIAGNOSIS_CLASSES,
    build_caption,
    build_caption_variants,
    build_label_arrays,
    check_support_rule,
    class_index_targets,
    group_level_support,
    normalize_diagnosis,
    one_hot_from_indices,
    one_hot_vector,
    site_phrase,
    validate_metadata,
)


def _load_split_builder():
    spec = importlib.util.spec_from_file_location(
        "ham_build_splits", REPO / "scripts" / "data" / "ham10000" / "01_build_splits.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata(n_lesions: int = 40, images_per_lesion: int = 2) -> pd.DataFrame:
    """Synthetic HAM10000-shaped metadata: every lesion photographed more than once, which is the
    exact condition an image-level split would leak on."""
    rows = []
    for lesion in range(n_lesions):
        diagnosis = DIAGNOSIS_CLASSES[lesion % len(DIAGNOSIS_CLASSES)]
        for image in range(images_per_lesion):
            rows.append(
                {
                    "lesion_id": f"HAM_{lesion:05d}",
                    "image_id": f"ISIC_{lesion:05d}_{image}",
                    "dx": diagnosis,
                    "dx_type": "histo",
                    "age": 40 + (lesion % 5) * 10,
                    "sex": "male" if lesion % 2 else "female",
                    "localization": "back" if lesion % 3 else "unknown",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- label encodings

def test_class_index_targets_match_the_canonical_class_order():
    frame = pd.DataFrame({"dx": ["nv", "df", "mel"]})
    indices = class_index_targets(frame)
    assert list(indices) == [
        CLASSIFIER_TARGET_LABELS.index("nv"),
        CLASSIFIER_TARGET_LABELS.index("df"),
        CLASSIFIER_TARGET_LABELS.index("mel"),
    ]
    assert indices.dtype == np.int64, "CrossEntropyLoss requires int64 class indices"


def test_the_two_target_encodings_cannot_disagree():
    """class_index_targets (CrossEntropy) and build_label_arrays (evaluation/ASISM) are derived
    from the same column; a round-trip through one_hot_from_indices must reproduce the other."""
    frame = pd.DataFrame({"dx": ["nv", "mel", "df", "vasc", "bkl", "bcc", "akiec"]})
    _, one_hot_target, _ = build_label_arrays(frame)
    from_indices = one_hot_from_indices(class_index_targets(frame))
    assert np.array_equal(one_hot_target, from_indices)


def test_one_hot_is_exactly_one_positive_per_image():
    """Mutual exclusivity is a property of the data, and the encoding must preserve it."""
    frame = pd.DataFrame({"dx": list(DIAGNOSIS_CLASSES)})
    _, target, mask = build_label_arrays(frame)
    assert target.sum(axis=1).tolist() == [1.0] * len(DIAGNOSIS_CLASSES)
    assert mask.all(), "HAM10000 has no uncertain state, so nothing may be masked out"


def test_unknown_diagnosis_is_rejected_not_silently_dropped():
    for bad in ("melanoma", "", None, "MEL2"):
        try:
            normalize_diagnosis(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid dx {bad!r}")
    assert normalize_diagnosis(" MEL ") == "mel", "case/whitespace normalisation is intended"


def test_validate_metadata_refuses_rows_without_a_lesion_id():
    frame = _metadata(n_lesions=3)
    frame.loc[0, "lesion_id"] = None
    try:
        validate_metadata(frame)
    except ValueError as exc:
        assert "lesion" in str(exc)
    else:
        raise AssertionError("a row with no lesion_id was accepted")


def test_one_hot_vector_has_a_single_positive_class():
    vector = one_hot_vector("mel")
    assert vector["mel"] == 1
    assert sum(vector.values()) == 1


# ---------------------------------------------------------------- captions

def test_caption_names_the_diagnosis_and_reads_as_one_sentence():
    caption = build_caption({"dx": "mel", "age": 55, "sex": "male", "localization": "back"})
    assert "melanoma" in caption
    assert "on the back" in caption
    assert "50-59-year-old" in caption
    assert caption.count(".") == 1, "one caption is one sentence"


def test_unknown_site_is_omitted_rather_than_captioned_as_unknown():
    """Conditioning on the literal token 'unknown' would teach the generator that 'unknown' is a
    body site."""
    assert site_phrase("unknown") == ""
    assert site_phrase(None) == ""
    caption = build_caption({"dx": "nv", "age": 30, "sex": "female", "localization": "unknown"})
    assert "unknown" not in caption.lower()


def test_variants_are_paraphrases_of_the_same_fact_not_different_facts():
    row = {"dx": "df", "age": 61, "sex": "female", "localization": "lower extremity"}
    variants = build_caption_variants(row, image_id="ISIC_0000001", num_variants=4)
    assert len(set(variants)) == 4, "the four variants must be distinct phrasings"
    for caption in variants:
        assert "dermatofibroma" in caption
        assert "lower extremity" in caption


def test_caption_uses_the_right_indefinite_article_before_the_age_bucket():
    """Regression: 'a 80-89-year-old' / 'a adult'. The article follows how the phrase is READ, and
    the generator is conditioned on this text, so it has to be right before Stage 1 trains."""
    from scripts.utils.ham10000 import TEMPLATES, build_caption, indefinite_article

    assert indefinite_article("80-89-year-old") == "an" and indefinite_article("adult") == "an"
    assert indefinite_article("40-49-year-old") == "a" and indefinite_article("10-19-year-old") == "a"

    for variant in range(len(TEMPLATES)):
        old = build_caption({"dx": "mel", "age": 85, "sex": "male", "localization": "back"}, variant)
        young = build_caption({"dx": "mel", "age": 45, "sex": "male", "localization": "back"}, variant)
        unknown_age = build_caption({"dx": "mel", "age": None, "sex": "male", "localization": "back"}, variant)
        for caption in (old, young, unknown_age):
            assert " a 8" not in caption and " a adult" not in caption and not caption.startswith("A adult")
            assert not caption.startswith("A 8"), caption
        # Variants 1 and 3 carry no article before the bucket at all; where one IS used it must agree.
        if "-year-old" in old and ("in a" in old or old[:2] in ("A ", "An")):
            assert ("an 80-89-year-old" in old) or old.startswith("An 80-89-year-old"), old
            assert ("a 40-49-year-old" in young) or young.startswith("A 40-49-year-old"), young
            assert ("an adult" in unknown_age) or unknown_age.startswith("An adult"), unknown_age


def test_caption_template_version_is_bumped_and_matches_the_config():
    """A caption-text change must change the version, or Stage 1/2 would happily reuse captions
    written with the old phrasing."""
    from omegaconf import OmegaConf

    from scripts.utils.ham10000 import TEMPLATE_VERSION

    assert TEMPLATE_VERSION == "ham-v2"
    config = OmegaConf.load(REPO / "configs" / "ham10000_stage1.yaml")
    assert str(config.captions.template_version) == TEMPLATE_VERSION


def test_every_class_has_a_caption_phrase():
    for label in DIAGNOSIS_CLASSES:
        caption = build_caption({"dx": label, "age": 50, "sex": "male", "localization": "back"})
        assert caption and caption[0].isupper()


# ---------------------------------------------------------------- lesion-level support

def test_support_counts_lesions_not_images():
    """One lesion photographed three times is ONE positive lesion, not three."""
    frame = pd.DataFrame(
        {
            "lesion_id": ["L1", "L1", "L1", "L2"],
            "dx": ["mel", "mel", "mel", "nv"],
        }
    )
    support = group_level_support(frame).set_index("label")
    assert support.loc["mel", "positive_groups"] == 1
    assert support.loc["mel", "positive_images"] == 3
    # Mutually exclusive classes: a lesion positive for mel is a negative for every other class.
    assert support.loc["nv", "negative_groups"] == 1


def test_support_rule_reports_every_failing_class():
    support = pd.DataFrame(
        [
            {"label": "df", "positive_groups": 2, "negative_groups": 500},
            {"label": "vasc", "positive_groups": 3, "negative_groups": 500},
            {"label": "nv", "positive_groups": 900, "negative_groups": 500},
        ]
    )
    passed, failures = check_support_rule(support, min_positive_groups=10, min_negative_groups=50)
    assert passed is False
    assert {failure["label"] for failure in failures} == {"df", "vasc"}


# ---------------------------------------------------------------- the split and its audit

def test_partition_is_exact_disjoint_and_deterministic():
    module = _load_split_builder()
    lesions = [f"HAM_{i:05d}" for i in range(500)]
    fractions = {
        "gen_train": 0.50, "gen_val": 0.06, "classifier_train": 0.16,
        "classifier_val": 0.09, "asism_tuning_heldout": 0.09, "final_eval_heldout": 0.10,
    }
    groups = module.partition_groups(lesions, fractions, seed=42)

    assert sum(len(ids) for ids in groups.values()) == len(lesions), "every lesion must be assigned"
    seen: set[str] = set()
    for ids in groups.values():
        assert not (seen & ids), "partitions must be disjoint"
        seen |= ids
    assert seen == set(lesions)
    assert module.partition_groups(lesions, fractions, seed=42) == groups, "same seed, same split"


def test_audit_passes_on_a_correctly_grouped_split():
    module = _load_split_builder()
    frame = _metadata(n_lesions=42, images_per_lesion=2)
    fractions = {
        "gen_train": 0.50, "gen_val": 0.06, "classifier_train": 0.16,
        "classifier_val": 0.09, "asism_tuning_heldout": 0.09, "final_eval_heldout": 0.10,
    }
    groups = module.partition_groups(sorted(frame["lesion_id"].unique()), fractions, seed=42)
    frames = {name: frame[frame["lesion_id"].isin(ids)].reset_index(drop=True) for name, ids in groups.items()}

    report = module.audit_splits(frames, "lesion_id", "image_id")
    assert report["status"] == "PASS"
    assert report["failed_pairs"] == []


def test_audit_FAILS_when_one_lesion_is_split_across_two_partitions():
    """The test that actually matters: an image-level leak must be caught, not merely warned about.

    This is exactly what a naive per-image split produces — two photographs of one lesion landing
    on opposite sides of the train/test boundary.
    """
    module = _load_split_builder()
    frame = _metadata(n_lesions=20, images_per_lesion=2)

    leaked_lesion = "HAM_00000"
    rows = frame[frame["lesion_id"] == leaked_lesion]
    assert len(rows) == 2, "fixture must give this lesion two images for the leak to be possible"

    frames = {name: frame.iloc[0:0] for name in module.SPLIT_NAMES}
    frames["gen_train"] = rows.iloc[[0]].reset_index(drop=True)      # image 1 of the lesion
    frames["final_eval_heldout"] = rows.iloc[[1]].reset_index(drop=True)  # image 2 of the SAME lesion

    report = module.audit_splits(frames, "lesion_id", "image_id")
    assert report["status"] == "FAIL"
    assert any("gen_train" in pair and "final_eval_heldout" in pair for pair in report["failed_pairs"])


def test_audit_records_that_visual_near_duplicates_are_out_of_scope():
    """The audit must not imply a guarantee it cannot make: near-duplicates filed under different
    lesion_ids are undetectable from metadata, and the report has to say so."""
    module = _load_split_builder()
    frame = _metadata(n_lesions=14, images_per_lesion=1)
    frames = {name: frame.iloc[i::6].reset_index(drop=True) for i, name in enumerate(module.SPLIT_NAMES)}
    report = module.audit_splits(frames, "lesion_id", "image_id")
    assert "near-duplicate" in report["not_covered_by_this_audit"].lower()


# ---------------------------------------------------------------- stratified split + feasibility gate

REAL_LESIONS_PER_CLASS = {"nv": 5403, "mel": 614, "bkl": 727, "bcc": 327, "akiec": 228, "vasc": 98, "df": 73}
DECISION_SPLITS = ["classifier_train", "classifier_val", "asism_tuning_heldout", "final_eval_heldout"]


def _config_fractions() -> dict[str, float]:
    from omegaconf import OmegaConf

    config = OmegaConf.load(REPO / "configs" / "splits_ham10000.yaml")
    return {name: float(value) for name, value in config.fractions.items()}


def test_allocation_is_exact_and_never_below_the_floor():
    module = _load_split_builder()
    fractions = _config_fractions()
    for count in (0, 1, 7, 73, 98, 5403):
        allocation = module.stratified_allocation(count, fractions)
        assert sum(allocation.values()) == count
        for name, value in allocation.items():
            assert value >= int(np.floor(count * fractions[name] + 1e-9))


def test_committed_fractions_pass_the_feasibility_gate_on_the_real_lesion_counts():
    """The fractions in splits_ham10000.yaml must satisfy min 10 lesions per class in every
    decision-bearing split, for the actual HAM10000 lesion counts."""
    module = _load_split_builder()
    report = module.feasibility_report(REAL_LESIONS_PER_CLASS, _config_fractions(), 10, 50, DECISION_SPLITS)
    assert report["feasible"], report["failures"]
    assert report["binding_class"] == "df"
    assert abs(report["minimum_fraction_per_decision_split"] - 10 / 73) < 1e-12
    for split in DECISION_SPLITS:
        assert min(report["allocation_lesions"][label][split] for label in REAL_LESIONS_PER_CLASS) >= 10


def test_feasibility_gate_FAILS_for_the_original_fractions():
    """Regression: gen_train 0.50 with 0.09 evaluation splits cannot hold 10 df lesions even when
    perfectly stratified (73 * 0.09 = 6.6). The gate must refuse, not the support check afterwards."""
    module = _load_split_builder()
    original = {
        "gen_train": 0.50, "gen_val": 0.06, "classifier_train": 0.16,
        "classifier_val": 0.09, "asism_tuning_heldout": 0.09, "final_eval_heldout": 0.10,
    }
    report = module.feasibility_report(REAL_LESIONS_PER_CLASS, original, 10, 50, DECISION_SPLITS)
    assert report["feasible"] is False
    failing = {(f["split"], f["label"]) for f in report["failures"] if "positive_groups" in f}
    assert ("classifier_val", "df") in failing and ("final_eval_heldout", "df") in failing


def test_the_support_floor_is_ten_not_lowered():
    from omegaconf import OmegaConf

    config = OmegaConf.load(REPO / "configs" / "splits_ham10000.yaml")
    assert int(config.support_rule.min_positive_groups) == 10
    assert "final_eval_heldout" in list(config.support_rule.decision_bearing_splits)
    assert str(config.stratify_by) == "dx"


def _stratified_metadata() -> pd.DataFrame:
    rows = []
    for label, lesions in {"nv": 400, "mel": 90, "bkl": 90, "bcc": 80, "akiec": 80, "vasc": 75, "df": 73}.items():
        for lesion in range(lesions):
            for image in range(1 + lesion % 3):  # 1-3 photographs per lesion
                rows.append({"lesion_id": f"{label}_{lesion:04d}", "image_id": f"{label}_{lesion:04d}_{image}", "dx": label})
    return pd.DataFrame(rows)


def test_stratified_partition_keeps_lesions_whole_is_disjoint_and_meets_support():
    module = _load_split_builder()
    frame = _stratified_metadata()
    lesion_to_class = module.lesion_classes(frame)
    assignment = module.partition_groups_stratified(lesion_to_class, _config_fractions(), seed=42)

    assigned = [lesion for lesions in assignment.values() for lesion in lesions]
    assert sorted(assigned) == sorted(lesion_to_class), "every lesion exactly once"
    frames = {name: frame[frame["lesion_id"].isin(ids)].reset_index(drop=True) for name, ids in assignment.items()}
    assert module.audit_splits(frames, "lesion_id", "image_id")["status"] == "PASS"
    assert sum(len(f) for f in frames.values()) == len(frame), "no image dropped"
    for split in DECISION_SPLITS:
        support = group_level_support(frames[split])
        passed, failures = check_support_rule(support, min_positive_groups=10, min_negative_groups=50)
        assert passed, (split, failures)


def test_stratified_partition_is_reproducible_and_seed_sensitive():
    module = _load_split_builder()
    lesion_to_class = module.lesion_classes(_stratified_metadata())
    fractions = _config_fractions()
    first = module.partition_groups_stratified(lesion_to_class, fractions, seed=42)
    shuffled = dict(reversed(list(lesion_to_class.items())))  # file order must not matter
    assert module.partition_groups_stratified(shuffled, fractions, seed=42) == first
    assert module.partition_groups_stratified(lesion_to_class, fractions, seed=7) != first


def test_a_lesion_with_two_diagnoses_is_refused_before_stratifying():
    module = _load_split_builder()
    frame = pd.DataFrame({"lesion_id": ["L1", "L1"], "image_id": ["a", "b"], "dx": ["nv", "mel"]})
    try:
        module.lesion_classes(frame)
    except SystemExit:
        return
    raise AssertionError("a lesion with conflicting diagnoses was stratified")


def test_audit_output_is_ascii_safe_for_windows_consoles():
    """Regression: the audit printed U+2229 and crashed on a cp1252 console before writing splits."""
    module = _load_split_builder()
    frame = _metadata(n_lesions=14, images_per_lesion=1)
    frames = {name: frame.iloc[i::6].reset_index(drop=True) for i, name in enumerate(module.SPLIT_NAMES)}
    report = module.audit_splits(frames, "lesion_id", "image_id")
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        module.print_audit(report)
    buffer.getvalue().encode("cp1252")  # raises UnicodeEncodeError on any non-cp1252 character
    source = (REPO / "scripts" / "data" / "ham10000" / "01_build_splits.py").read_text(encoding="utf-8")
    assert "∩" not in source


if __name__ == "__main__":
    import traceback

    tests = [(name, value) for name, value in sorted(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for name, function in tests:
        try:
            function()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    raise SystemExit(1 if failed else 0)
