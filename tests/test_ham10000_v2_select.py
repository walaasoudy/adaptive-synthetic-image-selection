"""The Stage 3 v2 selection: 5b fill-to-N targets, 6a per-class floor, top-k, and the D2 control.

The scores are arbitrary; what is under test is that the pre-registered rules hold, that what the
selection cannot deliver is reported rather than hidden, and that nothing of ham-final-v1 is touched.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.followup import ham10000_v2_select as v2

LABELS = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]


def _cfg(**overrides):
    selection = {"fill_to_total": 300, "quality_floor_percentile": 25, "quality_floor_scope": "per_class",
                 "within_class_rule": "top_k", **overrides.pop("selection", {})}
    d2 = {"seed": 42, "source_pool": "safe", **overrides.pop("d2", {})}
    return SimpleNamespace(selection=SimpleNamespace(**selection), d2=SimpleNamespace(**d2))


def _scored(per_class, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for label, n in per_class.items():
        for i in range(n):
            rows.append({"image_id": f"{label}_{i:04d}", "dx": label, "score": float(rng.random())})
    return pd.DataFrame(rows)


REAL = {"nv": 4000, "mel": 190, "bkl": 190, "bcc": 90, "akiec": 50, "vasc": 20, "df": 25}


def test_targets_fill_each_class_to_N_and_nv_gets_nothing():
    targets = v2.fill_targets(REAL, 300, LABELS)
    assert targets == {"nv": 0, "mel": 110, "bkl": 110, "bcc": 210, "akiec": 250, "vasc": 280, "df": 275}


def test_a_non_positive_N_is_refused():
    with pytest.raises(v2.V2SelectionError):
        v2.fill_targets(REAL, 0, LABELS)


def test_the_floor_is_computed_within_each_class_not_pooled():
    # mel scores all low, df all high: a pooled p25 would remove most of mel and none of df.
    frame = pd.DataFrame({
        "image_id": [f"m{i}" for i in range(100)] + [f"d{i}" for i in range(100)],
        "dx": ["mel"] * 100 + ["df"] * 100,
        "score": list(np.linspace(0.0, 0.3, 100)) + list(np.linspace(0.7, 1.0, 100)),
    })
    above, floors = v2.per_class_floor(frame, 25)
    assert floors["mel"] < 0.3 < 0.7 < floors["df"]
    assert (above["dx"] == "mel").sum() == (above["dx"] == "df").sum() == 75


def test_top_k_takes_each_classes_best_up_to_its_target():
    frame = _scored({"mel": 50, "df": 50})
    chosen = v2.top_k_per_class(frame, {"mel": 10, "df": 0})
    assert (chosen["dx"] == "df").sum() == 0
    mel = frame[frame["dx"] == "mel"].sort_values("score", ascending=False)
    assert set(chosen["image_id"]) == set(mel["image_id"].head(10))


def test_a_short_class_takes_everything_and_the_shortfall_is_reported_without_moving_N():
    scored = _scored({"nv": 500, "mel": 400, "bkl": 400, "bcc": 400, "akiec": 400, "vasc": 40, "df": 400})
    result = v2.select_v2(scored, REAL, _cfg(), LABELS)
    assert result["targets"]["vasc"] == 280                       # N was not lowered for vasc
    assert result["selected_per_class"]["vasc"] == 30             # all 40 minus its own floor
    assert result["classes_short_of_target"] == {"vasc": {"selected": 30, "target": 280}}
    assert result["selected_per_class"]["nv"] == 0
    assert result["selected_per_class"]["mel"] == 110


def test_c2_never_selects_below_its_own_classes_floor():
    scored = _scored({label: 400 for label in LABELS})
    result = v2.select_v2(scored, REAL, _cfg(), LABELS)
    floors = result["floors"]
    c2 = result["c2"]
    assert (c2["score"].to_numpy() >= c2["dx"].map(floors).to_numpy()).all()


def test_d2_has_c2s_counts_per_class_and_draws_from_the_safe_pool_not_the_floored_one():
    scored = _scored({label: 400 for label in LABELS})
    result = v2.select_v2(scored, REAL, _cfg(), LABELS)
    d2 = result["d2"]
    assert {label: int((d2["dx"] == label).sum()) for label in LABELS} == result["selected_per_class"]
    assert set(d2["image_id"]) <= set(scored["image_id"])
    below = d2["score"].to_numpy() < d2["dx"].map(result["floors"]).to_numpy()
    assert below.any()                                            # the floor was not applied to D2


def test_d2_is_reproducible_and_independent_of_the_score_files_row_order():
    scored = _scored({label: 400 for label in LABELS})
    first = v2.select_v2(scored, REAL, _cfg(), LABELS)["d2"]
    shuffled = scored.sample(frac=1.0, random_state=7).reset_index(drop=True)
    second = v2.select_v2(shuffled, REAL, _cfg(), LABELS)["d2"]
    assert sorted(first["image_id"]) == sorted(second["image_id"])
    other = v2.select_v2(scored, REAL, _cfg(d2={"seed": 43}), LABELS)["d2"]
    assert sorted(first["image_id"]) != sorted(other["image_id"])


@pytest.mark.parametrize("override", [
    {"selection": {"quality_floor_scope": "pooled"}},
    {"selection": {"within_class_rule": "threshold_network"}},
    {"d2": {"source_pool": "floored"}},
])
def test_a_config_that_departs_from_the_preregistered_rules_is_refused(override):
    with pytest.raises(v2.V2SelectionError):
        v2.select_v2(_scored({"mel": 100}), REAL, _cfg(**override), LABELS)


def test_a_score_file_that_misses_a_safe_candidate_is_refused():
    pool = pd.DataFrame({"image_id": ["a", "b", "c"], "dx": ["mel"] * 3})
    scores = pd.DataFrame({"image_id": ["a", "b"], "s": [0.1, 0.2]})
    with pytest.raises(v2.V2SelectionError, match="no score"):
        v2.attach_scores(pool, scores, "s")


def test_duplicate_ids_or_a_disagreeing_dx_in_the_score_file_are_refused():
    pool = pd.DataFrame({"image_id": ["a", "b"], "dx": ["mel", "df"]})
    with pytest.raises(v2.V2SelectionError, match="more than once"):
        v2.attach_scores(pool, pd.DataFrame({"image_id": ["a", "a", "b"], "s": [1, 2, 3]}), "s")
    with pytest.raises(v2.V2SelectionError, match="different dx"):
        v2.attach_scores(pool, pd.DataFrame({"image_id": ["a", "b"], "dx": ["mel", "nv"], "s": [1, 2]}), "s")


def test_scored_candidates_that_are_no_longer_safe_are_left_out_and_counted():
    pool = pd.DataFrame({"image_id": ["a", "b"], "dx": ["mel", "mel"]})
    scores = pd.DataFrame({"image_id": ["a", "b", "unsafe"], "s": [0.1, 0.2, 0.9]})
    scored, report = v2.attach_scores(pool, scores, "s")
    assert set(scored["image_id"]) == {"a", "b"}
    assert report["score_rows_not_in_safe_pool"] == 1


# ----------------------------------------------------------------------------------------------
# The entry point, over a laid-out PROJECT_ROOT with the pool and split loaders stubbed
# ----------------------------------------------------------------------------------------------


@pytest.fixture
def project(tmp_path, monkeypatch):
    from scripts.asism import ham10000_05_adaptive_thresholds as thresholds
    from scripts.asism import ham10000_ranking as ranking

    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("THESIS_CONFIG_OVERLAY", raising=False)
    scored = _scored({label: 400 for label in LABELS})
    pool = scored[["image_id", "dx"]]
    monkeypatch.setattr(ranking, "load_candidate_pool", lambda *a, **k: (pool, ["similarity"], {"n": len(pool)}))
    monkeypatch.setattr(thresholds, "real_class_counts", lambda splits_dir, split: dict(REAL))

    stage2 = tmp_path / "outputs" / "ham10000" / "stage2" / "ns"
    stage2.mkdir(parents=True)
    pool.assign(image_path=[f"/img/{i}.png" for i in pool["image_id"]]).to_csv(stage2 / "all_candidates.csv", index=False)
    scores_path = tmp_path / "scores.csv"
    scored.rename(columns={"score": "ranking_score"}).to_csv(scores_path, index=False)

    v1_selection = tmp_path / "outputs" / "ham10000" / "stage3" / "asism_selected.csv"
    v1_selection.parent.mkdir(parents=True)
    v1_selection.write_text("image_id,image_path,dx,selection_score\nv1_only,/x.png,mel,0.9\n")
    return SimpleNamespace(root=tmp_path, scores=scores_path, v1_selection=v1_selection)


def test_run_writes_c2_d2_and_a_manifest_under_stage3_v2_and_leaves_v1_untouched(project):
    before = project.v1_selection.read_bytes()
    manifest = v2.run("ns", project.scores, "ranking_score")

    out = project.root / "outputs" / "ham10000" / "stage3_v2" / "ns"
    c2, d2 = pd.read_csv(out / v2.C2_NAME), pd.read_csv(out / v2.D2_NAME)
    assert list(c2.columns) == ["image_id", "image_path", "dx", "selection_score"]
    assert len(c2) == len(d2) == manifest["n_selected_c2"] == manifest["n_selected_d2"]
    assert manifest["target_counts"]["nv"] == 0 and manifest["selected_per_class"]["nv"] == 0
    assert manifest["score_source"]["column"] == "ranking_score" and len(manifest["score_source"]["sha256"]) == 64
    assert json.loads((out / v2.MANIFEST_NAME).read_text())["config"]["fill_to_total"] == 300
    assert project.v1_selection.read_bytes() == before


def test_run_refuses_to_write_into_the_v1_stage3_directory(project):
    before = project.v1_selection.read_bytes()
    with pytest.raises(v2.V2SelectionError, match="v1 Stage 3"):
        v2.run("ns", project.scores, "ranking_score", out_dir=project.v1_selection.parent)
    assert project.v1_selection.read_bytes() == before
