#!/usr/bin/env python3
"""Stage 3b — the Go/No-Go gate that decides which signals may be weighted into the selector.

WHAT THESE FIXTURES ARE. The gate's input is score frames plus provenance sidecars, so that is what
the tests construct: small frames written with deliberate, known properties — a flagged near-
duplicate that kept its novelty credit, an entropy decomposition that does not add up, two signals
independent within class but separated across classes. They are contrasts built to exercise a
decision rule, NOT stand-ins for pipeline output: no test here fabricates a plausible-looking
artifact in order to make a signal pass. Every happy-path assertion is about a property the fixture
was explicitly constructed to have.

The gate is the one place where a signal can be refused, so what is under test is mainly refusal:
that an inadmissible explainability column cannot be included however good its numbers are, that a
provenance trail naming the protected split stops the signal rather than being reported, and that
redundancy is judged within class so that HAM10000's class imbalance cannot manufacture it.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fixture_workspace import fixture_workspace  # noqa: E402

NAMESPACE = "ham-gonogo-v1"
CLASSES = ("nv", "mel", "bkl")
PER_CLASS = 40
N = PER_CLASS * len(CLASSES)


def _image_ids() -> list[str]:
    return [f"SYN_{diagnosis}_{index:03d}" for diagnosis in CLASSES for index in range(PER_CLASS)]


def _diagnoses() -> list[str]:
    return [diagnosis for diagnosis in CLASSES for _ in range(PER_CLASS)]


def _spread(seed: int, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Distinct, well-spread values: enough unique levels to clear the stability floor."""
    return np.random.default_rng(seed).uniform(low, high, N)


# ----------------------------------------------------------------------------------------------
# Frames, each built to satisfy its own directionality contrast
# ----------------------------------------------------------------------------------------------


def _iqa_frame(seed: int = 1) -> pd.DataFrame:
    composite = _spread(seed)
    near_uniform = np.zeros(N, dtype=bool)
    near_uniform[:5] = True
    composite[:5] = composite[:5] * 0.01  # flagged images must score below unflagged ones
    return pd.DataFrame(
        {
            "image_id": _image_ids(),
            "iqa_composite": composite,
            "iqa_valid": True,
            "iqa_is_near_uniform": near_uniform,
        }
    )


def _similarity_frame(seed: int = 2, *, duplicate_keeps_novelty: bool = False) -> pd.DataFrame:
    knn_mean = _spread(seed, 0.2, 0.9)
    is_duplicate = np.zeros(N, dtype=bool)
    is_duplicate[:4] = True
    novelty = knn_mean * (~is_duplicate)
    if duplicate_keeps_novelty:
        novelty[0] = 0.7
    return pd.DataFrame(
        {
            "image_id": _image_ids(),
            "similarity_knn_mean": knn_mean,
            "novelty_is_near_duplicate": is_duplicate,
            "novelty_score": novelty,
        }
    )


def _uncertainty_frame(seed: int = 3, *, break_identity: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    predictive = rng.uniform(0.05, 0.95, N)
    expected = predictive * rng.uniform(0.2, 0.9, N)
    mutual = np.clip(predictive - expected, 0.0, None)
    if break_identity:
        mutual = mutual + 0.25
    return pd.DataFrame(
        {
            "image_id": _image_ids(),
            "uncertainty_predictive_entropy": predictive,
            "uncertainty_expected_entropy": expected,
            "uncertainty_mutual_information": mutual,
            "uncertainty_band": np.where(mutual <= 0.05, "low", np.where(mutual <= 0.20, "moderate", "extreme")),
            "uncertainty_n_passes": 20,
        }
    )


def _agreement_frame(seed: int = 4, *, invert: bool = False) -> pd.DataFrame:
    score = _spread(seed, 0.1, 0.9)
    is_match = np.arange(N) % 3 != 0
    score = np.where(is_match, score * 0.4 + 0.6, score * 0.4)  # matches score higher
    if invert:
        score = 1.0 - score
    return pd.DataFrame(
        {
            "image_id": _image_ids(),
            "agreement_score": score,
            "agreement_is_argmax_match": is_match,
            "agreement_intended_diagnosis": _diagnoses(),
        }
    )


def _explainability_frame(seed: int = 5, *, uncalibrated_class: str | None = None) -> pd.DataFrame:
    typicality = _spread(seed, 0.01, 0.99)
    calibrated = np.ones(N, dtype=bool)
    if uncalibrated_class is not None:
        mask = np.array([diagnosis == uncalibrated_class for diagnosis in _diagnoses()])
        typicality = np.where(mask, np.nan, typicality)
        calibrated = ~mask
    return pd.DataFrame(
        {
            "image_id": _image_ids(),
            "explainability_calibrated_typicality": typicality,
            "explainability_calibrated": calibrated,
            "explainability_content_box_exact": calibrated,
            "explainability_reference_split": "gen_train",
            "explainability_reference_scientific": True,
        }
    )


FRAME_BUILDER = {
    "iqa": _iqa_frame,
    "similarity": _similarity_frame,
    "uncertainty": _uncertainty_frame,
    "agreement": _agreement_frame,
    "explainability": _explainability_frame,
}


# ----------------------------------------------------------------------------------------------
# Workspace
# ----------------------------------------------------------------------------------------------


def _write_manifest(root: Path) -> Path:
    path = root / "outputs/ham10000/stage2" / NAMESPACE / "all_candidates.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_id": _image_ids(), "dx": _diagnoses(), "status": "accepted"}).to_csv(path, index=False)
    return path


def _write_artifact(root: Path, signal: str, frame: pd.DataFrame, provenance: dict | None = None) -> Path:
    from scripts.asism.signals import write_score_artifact

    scores_dir = root / "outputs/ham10000/stage3" / NAMESPACE / "signals"
    scores_dir.mkdir(parents=True, exist_ok=True)
    path = scores_dir / f"{signal}_scores.parquet"
    write_score_artifact(
        frame,
        path,
        signal,
        {
            "dataset": "ham10000",
            "split_namespace": NAMESPACE,
            "candidates_csv_sha256": "a" * 64,
            "git_commit_hash": "fixture",
            **(provenance or {}),
        },
    )
    return path


def _write_all(root: Path, **overrides) -> None:
    for signal, builder in FRAME_BUILDER.items():
        _write_artifact(root, signal, overrides.get(signal, builder()))


def _run(**kwargs):
    from scripts.asism.ham10000_02_gonogo import run

    return run(NAMESPACE, **kwargs)


def _config():
    from scripts.utils.config import load_named_config

    return load_named_config("ham10000_stage3.yaml", "ham_stage3")


@pytest.fixture
def workspace(monkeypatch):
    with fixture_workspace("gonogo") as root:
        monkeypatch.setenv("PROJECT_ROOT", str(root))
        _write_manifest(root)
        yield root


# ==============================================================================================
# The happy path
# ==============================================================================================


def test_five_well_formed_signals_all_pass_and_the_report_names_the_evidence(workspace):
    _write_all(workspace)
    report = _run()

    assert sorted(report["surviving_signals"]) == ["agreement", "explainability", "iqa", "similarity", "uncertainty"]
    assert report["excluded_signals"] == []
    assert report["asism_variant_status"] == "primary"
    assert "final_eval_heldout never read" in report["evidence"]
    assert Path(report["report_path"]).is_file()

    # The report is the write-up's evidence: every check's numbers survive into it, not just a verdict.
    checks = report["per_signal"]["iqa"]["checks"]
    assert set(checks) == {
        "technical_validity", "selection_eligibility", "numerical_stability", "missing_rate",
        "provenance", "directionality", "redundancy", "usefulness",
    }
    assert checks["directionality"]["method"] == "near_uniform_images_score_lower_than_clean"


def test_uncertainty_is_included_without_any_direction_being_asserted(workspace):
    """The signal has no a-priori good end: the gate checks the decomposition arithmetic and that the
    bands are populated, and the usefulness check declines to score it rather than inventing one."""
    _write_all(workspace)
    report = _run()
    checks = report["per_signal"]["uncertainty"]["checks"]

    assert report["per_signal"]["uncertainty"]["outcome"] == "include"
    assert checks["directionality"]["detail"]["max_identity_residual"] < 1e-9
    assert checks["usefulness"]["reason"] == "signal_has_no_directional_score"
    assert checks["usefulness"]["jaccard_overlap_of_selection"] is None


def test_an_artifact_that_was_never_produced_is_excluded_and_said_to_be_absent(workspace):
    _write_all(workspace)
    (workspace / "outputs/ham10000/stage3" / NAMESPACE / "signals" / "similarity_scores.parquet").unlink()
    report = _run()

    assert report["per_signal"]["similarity"]["outcome"] == "exclude"
    assert report["per_signal"]["similarity"]["reason"] == "artifact not produced"
    assert report["per_signal"]["similarity"]["artifact_retained_for_audit"] is False


# ==============================================================================================
# Hard refusals
# ==============================================================================================


def test_a_signal_missing_its_headline_column_is_excluded(workspace):
    _write_all(workspace, agreement=_agreement_frame().drop(columns=["agreement_score"]))
    report = _run()
    assert report["per_signal"]["agreement"]["outcome"] == "exclude"
    assert "technical_validity" in report["per_signal"]["agreement"]["reason"]


def test_a_near_constant_score_is_excluded_even_though_it_computes_cleanly(workspace):
    frame = _iqa_frame()
    frame["iqa_composite"] = 0.5
    _write_all(workspace, iqa=frame)
    report = _run()

    entry = report["per_signal"]["iqa"]
    assert entry["outcome"] == "exclude"
    assert entry["checks"]["numerical_stability"]["n_unique_values"] == 1


def test_RAW_explainability_cannot_be_admitted_however_good_its_numbers_are(workspace):
    """The single rule Stage 3 exists to enforce: only the typicality calibrated against the real
    gen_train reference is a selection feature. Rows that were never calibrated carry a number the
    guard refuses, and the gate must refuse them too rather than treating the column as present."""
    frame = _explainability_frame()
    frame["explainability_calibrated"] = False
    _write_all(workspace, explainability=frame)
    report = _run()

    entry = report["per_signal"]["explainability"]
    assert entry["outcome"] == "exclude"
    assert "selection_eligibility" in entry["reason"]
    assert "not calibrated" in entry["checks"]["selection_eligibility"]["reason"]


def test_explainability_calibrated_against_a_non_permitted_split_is_excluded(workspace):
    frame = _explainability_frame()
    frame["explainability_reference_split"] = "classifier_train"
    _write_all(workspace, explainability=frame)
    report = _run()

    assert report["per_signal"]["explainability"]["outcome"] == "exclude"
    assert "selection_eligibility" in report["per_signal"]["explainability"]["reason"]


def test_a_class_with_no_reference_pushes_explainability_past_the_missing_rate(workspace):
    """A whole class scoring NaN is legitimate and recorded upstream, but a signal that cannot rank
    one class in seven would select only inside the classes it covers."""
    _write_all(workspace, explainability=_explainability_frame(uncalibrated_class="mel"))
    report = _run()

    entry = report["per_signal"]["explainability"]
    assert entry["outcome"] == "exclude"
    assert entry["checks"]["missing_rate"]["missing_fraction"] == pytest.approx(1 / 3)


def test_a_signal_without_a_provenance_sidecar_is_excluded(workspace):
    _write_all(workspace)
    sidecar = workspace / "outputs/ham10000/stage3" / NAMESPACE / "signals" / "iqa_scores.provenance.json"
    sidecar.unlink()
    report = _run()

    assert report["per_signal"]["iqa"]["outcome"] == "exclude"
    assert report["per_signal"]["iqa"]["checks"]["provenance"]["reason"] == "missing_provenance_sidecar"


def test_a_signal_whose_provenance_names_the_protected_split_is_excluded_not_merely_reported(workspace):
    """A score computed against final_eval_heldout would invalidate Stage 5 whatever it looks like,
    and the only trace of it is the provenance trail — so the trail is what is searched."""
    _write_all(workspace)
    _write_artifact(
        workspace, "similarity", _similarity_frame(), {"reference_split": "final_eval_heldout"}
    )
    report = _run()

    entry = report["per_signal"]["similarity"]
    assert entry["outcome"] == "exclude"
    assert entry["checks"]["provenance"]["forbidden_split_mentions"] == ["reference_split"]


# ==============================================================================================
# Directionality — the score must move the way the methodology says it does
# ==============================================================================================


def test_agreement_is_excluded_when_mismatched_candidates_score_HIGHER(workspace):
    _write_all(workspace, agreement=_agreement_frame(invert=True))
    report = _run()

    entry = report["per_signal"]["agreement"]
    assert entry["outcome"] == "exclude"
    assert "contradicts" in entry["reason"]
    assert entry["checks"]["directionality"]["method"] == (
        "candidates_read_back_as_the_intended_class_score_higher"
    )


def test_a_flagged_near_duplicate_that_kept_its_novelty_credit_excludes_similarity(workspace):
    """Memorisation must not be able to win on novelty: if a copied image still scores, the signal
    is rewarding exactly what it was built to catch."""
    _write_all(workspace, similarity=_similarity_frame(duplicate_keeps_novelty=True))
    report = _run()

    entry = report["per_signal"]["similarity"]
    assert entry["outcome"] == "exclude"
    assert entry["checks"]["directionality"]["detail"]["max_novelty_among_flagged"] == pytest.approx(0.7)


def test_an_entropy_decomposition_that_does_not_add_up_excludes_uncertainty(workspace):
    """Mutual information IS predictive minus expected entropy. If the artifact disagrees, the passes
    were not a valid probability sample and every band derived from them is meaningless."""
    _write_all(workspace, uncertainty=_uncertainty_frame(break_identity=True))
    report = _run()

    entry = report["per_signal"]["uncertainty"]
    assert entry["outcome"] == "exclude"
    assert entry["checks"]["directionality"]["detail"]["max_identity_residual"] == pytest.approx(0.25)


def test_a_typicality_outside_the_unit_interval_excludes_explainability(workspace):
    frame = _explainability_frame()
    frame.loc[0, "explainability_calibrated_typicality"] = 1.4
    _write_all(workspace, explainability=frame)
    report = _run()
    assert report["per_signal"]["explainability"]["outcome"] == "exclude"


def test_a_directionality_contrast_absent_from_the_sample_does_not_fail_the_signal(workspace):
    """"Not contradicted" is not the same as "verified", and the gate says which one it got rather
    than punishing a pool that happens to contain no flagged image."""
    frame = _similarity_frame()
    frame["novelty_is_near_duplicate"] = False
    _write_all(workspace, similarity=frame)
    report = _run()

    checks = report["per_signal"]["similarity"]["checks"]["directionality"]
    assert checks["passed"] is True
    assert checks["method"] == "no_near_duplicates_in_sample"


# ==============================================================================================
# Redundancy and usefulness — soft findings, and measured within class
# ==============================================================================================


def test_a_duplicated_signal_is_kept_for_the_ablation_rather_than_thrown_away(workspace):
    """Redundancy is a reason not to WEIGHT a signal twice, not a reason to discard its evidence: the
    leave-one-signal-out analysis needs the artifact, and the write-up needs the finding."""
    agreement = _agreement_frame()
    iqa = _iqa_frame()
    # Same ordering, different scale: rank-redundant by construction.
    iqa["iqa_composite"] = agreement["agreement_score"] * 3.0 + 0.5
    iqa["iqa_is_near_uniform"] = False
    _write_all(workspace, iqa=iqa, agreement=agreement)
    report = _run()

    entry = report["per_signal"]["iqa"]
    assert entry["outcome"] == "ablation_only"
    assert entry["checks"]["redundancy"]["most_correlated_with"] == "agreement"
    assert entry["checks"]["redundancy"]["max_abs_correlation"] == pytest.approx(1.0)
    assert entry["artifact_retained_for_audit"] is True
    assert "iqa" in report["ablation_only_signals"]


def test_class_structure_alone_does_not_manufacture_redundancy(workspace):
    """The reason the check is within class. Two signals independent inside every class but both
    offset by class read as almost perfectly correlated when pooled — and HAM10000's pool is exactly
    that shape, since nv candidates differ systematically from mel ones on nearly every signal."""
    from scripts.asism.ham10000_02_gonogo import check_redundancy

    rng = np.random.default_rng(11)
    # All seven real classes, because the size of the pooled artefact grows with the number of
    # classes: the between-class ordering both signals share is what a pooled correlation picks up,
    # and with seven classes it swamps the independent within-class noise entirely.
    labels = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
    per_class = 30
    image_ids = [f"SYN_{name}_{index:03d}" for name in labels for index in range(per_class)]
    names = [name for name in labels for _ in range(per_class)]
    shift = np.array([float(labels.index(name)) for name in names])
    merged = pd.DataFrame(
        {
            "iqa_composite": shift + rng.normal(0, 0.2, len(names)),
            "agreement_score": shift + rng.normal(0, 0.2, len(names)),
        },
        index=image_ids,
    )
    diagnoses = pd.Series(names, index=image_ids)

    pooled = float(merged["iqa_composite"].corr(merged["agreement_score"], method="spearman"))
    threshold = float(_config().gonogo.redundancy_abs_correlation_max)
    assert pooled > threshold  # a pooled check would have called these redundant

    result = check_redundancy(merged, diagnoses, "iqa", _config())
    assert result["passed"] is True
    assert abs(result["max_abs_correlation"]) < 0.5
    assert set(result["per_class"]["agreement"]) == set(labels)


def _two_signals_by_class(signs: dict[str, int], per_class: int = 60, seed: int = 21):
    """iqa and agreement locked together within every class, with the given sign per class."""
    rng = np.random.default_rng(seed)
    image_ids, names, iqa, agreement = [], [], [], []
    for name, sign in signs.items():
        base = rng.normal(0, 1, per_class)
        image_ids += [f"SYN_{name}_{index:03d}" for index in range(per_class)]
        names += [name] * per_class
        iqa += list(base)
        agreement += list(sign * base + rng.normal(0, 0.05, per_class))
    merged = pd.DataFrame({"iqa_composite": iqa, "agreement_score": agreement}, index=image_ids)
    return merged, pd.Series(names, index=image_ids)


def test_opposite_signs_across_classes_do_not_cancel_redundancy(workspace):
    """The bug this guards: +0.98 in some classes and -0.98 in others averaged to near zero."""
    from scripts.asism.ham10000_02_gonogo import check_redundancy

    signs = {"nv": -1, "mel": +1, "bkl": +1, "bcc": -1, "akiec": +1, "vasc": -1, "df": +1}
    merged, diagnoses = _two_signals_by_class(signs)
    result = check_redundancy(merged, diagnoses, "iqa", _config())

    per_class = result["per_class"]["agreement"]
    assert all(abs(rho) > 0.95 for rho in per_class.values())
    assert all(np.sign(per_class[name]) == sign for name, sign in signs.items())
    assert abs(result["signed_correlations_diagnostic_only"]["agreement"]) < 0.2  # what the old rule saw
    assert result["correlations"]["agreement"] > 0.95
    assert result["max_abs_correlation"] > 0.95
    assert result["method"] == "class_size_weighted_within_class_abs_spearman"
    assert result["passed"] is False


def test_same_sign_classes_give_the_same_aggregate_as_the_signed_rule(workspace):
    from scripts.asism.ham10000_02_gonogo import check_redundancy

    for sign in (+1, -1):
        merged, diagnoses = _two_signals_by_class({name: sign for name in ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]})
        result = check_redundancy(merged, diagnoses, "iqa", _config())
        signed = result["signed_correlations_diagnostic_only"]["agreement"]
        assert result["correlations"]["agreement"] == pytest.approx(abs(signed))
        assert result["passed"] is False
    # and with no redundancy at all, both read low and the check passes
    rng = np.random.default_rng(5)
    labels = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
    ids = [f"SYN_{name}_{index:03d}" for name in labels for index in range(60)]
    merged = pd.DataFrame({"iqa_composite": rng.normal(size=len(ids)), "agreement_score": rng.normal(size=len(ids))}, index=ids)
    result = check_redundancy(merged, pd.Series([n for n in labels for _ in range(60)], index=ids), "iqa", _config())
    assert result["passed"] is True
    assert result["threshold"] == pytest.approx(0.90)


def test_a_signal_that_changes_nothing_about_the_selection_is_ablation_only():
    """Measured directly on the check: a score whose within-class ranking already matches the
    composite of the others keeps exactly the same candidates whether it is weighted or not."""
    from scripts.asism.ham10000_02_gonogo import check_usefulness

    rng = np.random.default_rng(12)
    base = rng.uniform(0, 1, N)
    merged = pd.DataFrame(
        {
            "iqa_composite": base,
            "agreement_score": base,
            "similarity_knn_mean": base,
        },
        index=_image_ids(),
    )
    diagnoses = pd.Series(_diagnoses(), index=_image_ids())

    result = check_usefulness(merged, diagnoses, "similarity", _config())
    assert result["passed"] is False
    assert result["jaccard_overlap_of_selection"] == pytest.approx(1.0)


def test_the_reference_selector_keeps_a_share_of_EVERY_class(workspace):
    """A pooled top-fraction would simply keep whichever class scores highest overall, which on a
    two-thirds-nv pool means measuring usefulness on nv alone."""
    from scripts.asism.ham10000_02_gonogo import check_usefulness

    rng = np.random.default_rng(13)
    # nv scores far above everything else on both signals.
    bonus = np.array([3.0 if diagnosis == "nv" else 0.0 for diagnosis in _diagnoses()])
    merged = pd.DataFrame(
        {"iqa_composite": bonus + rng.uniform(0, 1, N), "agreement_score": rng.uniform(0, 1, N)},
        index=_image_ids(),
    )
    diagnoses = pd.Series(_diagnoses(), index=_image_ids())

    result = check_usefulness(merged, diagnoses, "iqa", _config())
    # Half of each of the three classes, not half of the pool drawn from one class.
    assert result["n_selected"] == pytest.approx(N // 2, abs=len(CLASSES))
    assert result["selection_fraction_per_class"] == 0.5


# ==============================================================================================
# Pool-level gates
# ==============================================================================================


def test_artifacts_scored_on_DIFFERENT_candidate_pools_are_refused(workspace):
    """Cross-signal comparison across two pools — a full run against a --limit smoke run — produces
    correlations and overlaps that look ordinary and mean nothing."""
    _write_all(workspace)
    _write_artifact(workspace, "iqa", _iqa_frame(), {"candidates_csv_sha256": "b" * 64})

    with pytest.raises(SystemExit, match="different candidate manifests"):
        _run()


def test_an_alternate_scores_directory_requires_an_explicit_report_path(workspace):
    """An audit must never silently replace the canonical report used downstream."""
    _write_all(workspace)
    from scripts.asism.ham10000_02_gonogo import run

    alternate = workspace / "alternate-signals"
    alternate.mkdir()
    for source in (workspace / "outputs/ham10000/stage3" / NAMESPACE / "signals").iterdir():
        target = alternate / source.name
        target.write_bytes(source.read_bytes())

    with pytest.raises(SystemExit, match="Supply --out explicitly"):
        run(NAMESPACE, scores_dir=alternate)

    report_path = alternate / "audit_gonogo_report.json"
    report = run(NAMESPACE, scores_dir=alternate, out_path=report_path)
    assert report_path.is_file()
    assert Path(report["report_path"]) == report_path


def test_a_scored_image_absent_from_the_manifest_stops_the_gate(workspace):
    _write_all(workspace)
    frame = _iqa_frame()
    frame.loc[0, "image_id"] = "SYN_not_in_the_manifest"
    _write_artifact(workspace, "iqa", frame)

    with pytest.raises(SystemExit, match="absent from the candidate manifest"):
        _run()


def test_no_artifacts_at_all_names_the_command_that_produces_them(workspace):
    (workspace / "outputs/ham10000/stage3" / NAMESPACE / "signals").mkdir(parents=True, exist_ok=True)
    with pytest.raises(SystemExit, match="ham10000_01_compute_signals"):
        _run()


def test_a_missing_candidate_manifest_names_the_command_that_produces_it(workspace):
    _write_all(workspace)
    (workspace / "outputs/ham10000/stage2" / NAMESPACE / "all_candidates.csv").unlink()
    with pytest.raises(SystemExit, match="ham10000_generate_synthetic_images"):
        _run()


# ==============================================================================================
# The decision rule itself
# ==============================================================================================


def _checks(**overrides) -> dict:
    base = {
        name: {"passed": True}
        for name in (
            "technical_validity", "selection_eligibility", "numerical_stability", "missing_rate",
            "provenance", "directionality", "redundancy", "usefulness",
        )
    }
    base["redundancy"].update(most_correlated_with=None, max_abs_correlation=0.0)
    base.update(overrides)
    return base


def test_decide_maps_hard_failures_to_exclude_and_soft_ones_to_ablation_only():
    from scripts.asism.ham10000_02_gonogo import decide

    assert decide(_checks())[0] == "include"
    assert decide(_checks(provenance={"passed": False}))[0] == "exclude"
    assert decide(_checks(directionality={"passed": False, "method": "m"}))[0] == "exclude"
    assert decide(_checks(usefulness={"passed": False}))[0] == "ablation_only"
    assert decide(
        _checks(redundancy={"passed": False, "most_correlated_with": "iqa", "max_abs_correlation": 0.97})
    )[0] == "ablation_only"
    # A hard failure outranks a soft one: the reason names the hard check, not the redundancy.
    outcome, reason = decide(
        _checks(numerical_stability={"passed": False}, usefulness={"passed": False})
    )
    assert outcome == "exclude" and "numerical_stability" in reason


def test_an_unverified_directionality_result_is_not_treated_as_a_failure():
    """`passed: None` means the contrast could not be formed. Only an explicit False excludes."""
    from scripts.asism.ham10000_02_gonogo import decide

    assert decide(_checks(directionality={"passed": None, "method": ""}))[0] == "include"


def test_fewer_than_three_surviving_signals_downgrades_the_whole_variant(workspace):
    """The methodological consequence the gate exists to make visible: a selector built on one or two
    signals is not the method the thesis proposes, and must not be reported as though it were."""
    _write_all(
        workspace,
        iqa=_iqa_frame().drop(columns=["iqa_composite"]),
        agreement=_agreement_frame(invert=True),
        explainability=_explainability_frame(uncalibrated_class="mel"),
    )
    report = _run()

    assert len(report["surviving_signals"]) < 3
    assert report["asism_variant_status"].startswith("reduced_variant")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
