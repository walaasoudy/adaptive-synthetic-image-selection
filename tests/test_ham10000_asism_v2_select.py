"""ASISM v2 final selection: signal roles, the gate's composite, safety before ranking, C and D."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.followup import ham10000_asism_v2_select as sel
from scripts.utils.config import load_named_config

ADMITTED = ["similarity", "iqa", "uncertainty", "explainability"]
SIM, EXPL, IQA = "similarity_knn_mean", "explainability_calibrated_typicality", "iqa_composite"


def _frame(n_per_class=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for dx in ("df", "mel", "nv"):
        for i in range(n_per_class):
            rows.append({"image_id": f"{dx}_{i:03d}", "dx": dx, SIM: rng.uniform(0.3, 0.9),
                         EXPL: rng.uniform(0, 1), IQA: rng.uniform(0.75, 1.0),
                         "uncertainty_mutual_information": rng.uniform(0, 0.1)})
    return pd.DataFrame(rows).set_index("image_id")


def _gate_composite(group, columns):
    """The composite exactly as ham10000_02_gonogo.check_usefulness.selected builds it."""
    parts = []
    for column in columns:
        values = group[column].astype(float)
        spread = values.max() - values.min()
        parts.append((values - values.min()) / spread if spread > 0 else values * 0.0)
    composite = sum(parts) / max(len(parts), 1)
    return composite.fillna(composite.min() if composite.notna().any() else 0.0)


# ---- roles ------------------------------------------------------------------------------------

def test_scored_columns_are_similarity_and_explainability_only():
    assert sel.ranking_columns(ADMITTED, ["similarity", "explainability"]) == [SIM, EXPL]
    assert sel.ranking_columns(ADMITTED, ["explainability", "similarity"]) == [SIM, EXPL]


@pytest.mark.parametrize("scored", [
    ["similarity", "iqa", "explainability"],          # IQA is safety-only (contract §6)
    ["similarity", "explainability", "uncertainty"],  # no a-priori direction
    ["similarity"],
    [],
])
def test_any_other_scored_set_is_refused(scored):
    with pytest.raises(sel.SelectionError):
        sel.ranking_columns(ADMITTED, scored)


def test_a_scored_signal_must_be_admitted():
    with pytest.raises(sel.SelectionError):
        sel.ranking_columns(["similarity", "iqa", "uncertainty"], ["similarity", "explainability"])


def test_config_scores_the_protocol_set_and_keeps_the_safety_filters():
    cfg = load_named_config(sel.SELECTION_CONFIG, "ham_asism_v2_selection")
    assert sorted(cfg.ranking.scored_signals) == sorted(sel.SCORED_SIGNALS)
    assert str(cfg.ranking.method) == "equal_weight_within_class_minmax"
    assert bool(cfg.safety.reject_invalid_iqa) and bool(cfg.safety.reject_near_duplicates)
    assert sorted(cfg.admitted_signals) == sorted(ADMITTED)


# ---- the composite ----------------------------------------------------------------------------

def test_composite_equals_the_gate_composite():
    frame = _frame()
    for _, group in frame.groupby("dx"):
        pd.testing.assert_series_equal(sel.composite(group, [SIM, EXPL]), _gate_composite(group, [SIM, EXPL]))


def test_iqa_values_cannot_change_the_ranking():
    frame = _frame()
    columns = sel.ranking_columns(ADMITTED, list(sel.SCORED_SIGNALS))
    before = sel.rank_within_class(frame, columns)
    shuffled = frame.copy()
    shuffled[IQA] = np.random.default_rng(9).permutation(shuffled[IQA].to_numpy())
    shuffled["uncertainty_mutual_information"] = 1.0 - shuffled["uncertainty_mutual_information"]
    after = sel.rank_within_class(shuffled, columns)
    assert list(before.index) == list(after.index)


def test_normalisation_is_within_class():
    frame = _frame()
    columns = [SIM, EXPL]
    # Shifting one class's scores by a constant changes nothing about any class's order.
    shifted = frame.copy()
    shifted.loc[shifted["dx"] == "df", SIM] += 10.0
    a = sel.rank_within_class(frame, columns)["rank_in_class"]
    b = sel.rank_within_class(shifted, columns)["rank_in_class"]
    assert a.sort_index().equals(b.sort_index())


def test_ranks_are_by_composite_then_image_id():
    frame = pd.DataFrame({"dx": ["mel"] * 4, SIM: [0.5, 0.5, 0.9, 0.1], EXPL: [0.5, 0.5, 0.9, 0.1]},
                         index=pd.Index(["m_b", "m_a", "m_c", "m_d"], name="image_id"))
    ranked = sel.rank_within_class(frame, [SIM, EXPL])
    assert list(ranked.index) == ["m_c", "m_a", "m_b", "m_d"]
    assert list(ranked["rank_in_class"]) == [0, 1, 2, 3]


def test_missing_score_takes_the_class_minimum():
    frame = pd.DataFrame({"dx": ["mel"] * 3, SIM: [0.9, 0.2, 0.5], EXPL: [np.nan, 0.0, 1.0]},
                         index=pd.Index(["x", "y", "z"], name="image_id"))
    score = sel.composite(frame, [SIM, EXPL])
    assert score["x"] == score.min()


def test_constant_column_contributes_nothing():
    frame = pd.DataFrame({"dx": ["nv"] * 3, SIM: [0.4, 0.4, 0.4], EXPL: [0.1, 0.9, 0.5]},
                         index=pd.Index(["a", "b", "c"], name="image_id"))
    assert list(sel.rank_within_class(frame, [SIM, EXPL]).index) == ["b", "c", "a"]


# ---- C and D ----------------------------------------------------------------------------------

def test_c_takes_the_top_n_of_each_class_and_refuses_more_than_the_pool():
    ranked = sel.rank_within_class(_frame(), [SIM, EXPL])
    chosen = sel.select_c(ranked, {"df": 5, "mel": 3, "nv": 0})
    assert chosen["dx"].value_counts().to_dict() == {"df": 5, "mel": 3}
    assert (chosen["rank_in_class"] < 5).all()
    with pytest.raises(sel.SelectionError):
        sel.select_c(ranked, {"df": 41})


def test_d_is_reproducible_per_seed_and_independent_of_the_scores():
    frame = _frame()
    counts = {"df": 5, "mel": 3, "nv": 2}
    a = sel.draw_d(frame, counts, seed=42, base=20261002)
    b = sel.draw_d(frame.sample(frac=1.0, random_state=1), counts, seed=42, base=20261002)
    c = sel.draw_d(frame, counts, seed=43, base=20261002)
    assert sorted(a.index) == sorted(b.index)
    assert sorted(a.index) != sorted(c.index)
    assert a["dx"].value_counts().to_dict() == counts
    assert a.index.is_unique


def test_unsafe_candidates_never_reach_the_ranking(tmp_path, monkeypatch):
    frame = _frame(n_per_class=6)
    ids = list(frame.index)
    pool = pd.DataFrame({"image_id": ids, "image_path": [f"/x/{i}.png" for i in ids], "dx": frame["dx"].to_list()})
    pool.to_csv(tmp_path / "all_candidates.csv", index=False)
    scores = tmp_path / "scores"
    scores.mkdir()
    iqa = pd.DataFrame({"image_id": ids, IQA: frame[IQA].to_list(), "iqa_valid": [i != "df_000" for i in ids]})
    sim = pd.DataFrame({"image_id": ids, SIM: frame[SIM].to_list(), "novelty_is_near_duplicate": [i == "mel_001" for i in ids]})
    expl = pd.DataFrame({"image_id": ids, EXPL: frame[EXPL].to_list()})
    unc = pd.DataFrame({"image_id": ids, "uncertainty_mutual_information": frame["uncertainty_mutual_information"].to_list()})
    sha = sel.sha256_file(tmp_path / "all_candidates.csv")
    monkeypatch.setattr(sel, "CANDIDATES_SHA256", sha)
    for name, art in {"iqa": iqa, "similarity": sim, "explainability": expl, "uncertainty": unc}.items():
        art.to_parquet(scores / f"{name}_scores.parquet")
        (scores / f"{name}_scores.provenance.json").write_text(json.dumps({"candidates_csv_sha256": sha}))
    report = tmp_path / "gonogo_report.json"
    report.write_text(json.dumps({"surviving_signals": ADMITTED, "candidates_csv_sha256": sha}))
    safe, evidence = sel.load_pool(tmp_path / "all_candidates.csv", scores, report, ADMITTED, [SIM, EXPL])
    assert "df_000" not in safe.index and "mel_001" not in safe.index
    assert evidence["safety_removed"] == 2
    assert IQA not in safe.columns
    ranked = sel.rank_within_class(safe, [SIM, EXPL])
    assert set(ranked.index) == set(safe.index)


def test_v2_selection_has_no_quality_floor_or_v1_bounds():
    """Contract §10 removed the P25 floor, the real-count target and the 50/2000 bounds for v2."""
    cfg = load_named_config(sel.SELECTION_CONFIG, "ham_asism_v2_selection")
    text = open(sel.__file__, encoding="utf-8").read()
    for key in ("quality_floor", "target_synthetic_to_real_ratio", "min_accepted_per_class",
                "max_accepted_per_class", "fill_to_total"):
        assert key not in cfg and key not in str(cfg.get("selection", "")) and key not in text


# ---- run end to end, and the pool E4 measured ---------------------------------------------------

def _write_inputs(tmp_path, monkeypatch, n_per_class=40):
    frame = _frame(n_per_class=n_per_class)
    ids = list(frame.index)
    pool = pd.DataFrame({"image_id": ids, "image_path": [f"/x/{i}.png" for i in ids], "dx": frame["dx"].to_list()})
    pool.to_csv(tmp_path / "all_candidates.csv", index=False)
    scores = tmp_path / "scores"
    scores.mkdir()
    sha = sel.sha256_file(tmp_path / "all_candidates.csv")
    monkeypatch.setattr(sel, "CANDIDATES_SHA256", sha)
    artifacts = {
        "iqa": pd.DataFrame({"image_id": ids, IQA: frame[IQA].to_list(), "iqa_valid": [i != "df_000" for i in ids]}),
        "similarity": pd.DataFrame({"image_id": ids, SIM: frame[SIM].to_list(),
                                    "novelty_is_near_duplicate": [i == "mel_001" for i in ids]}),
        "explainability": pd.DataFrame({"image_id": ids, EXPL: frame[EXPL].to_list()}),
        "uncertainty": pd.DataFrame({"image_id": ids, "uncertainty_mutual_information":
                                     frame["uncertainty_mutual_information"].to_list()}),
    }
    for name, art in artifacts.items():
        art.to_parquet(scores / f"{name}_scores.parquet")
        (scores / f"{name}_scores.provenance.json").write_text(json.dumps({"candidates_csv_sha256": sha}))
    report = tmp_path / "gonogo_report.json"
    report.write_text(json.dumps({"surviving_signals": ADMITTED, "candidates_csv_sha256": sha}))
    safe_ids = [i for i in ids if i not in ("df_000", "mel_001")]
    counts = pd.Series([frame.loc[i, "dx"] for i in safe_ids]).value_counts().sort_index()
    return tmp_path / "all_candidates.csv", scores, report, safe_ids, {c: int(n) for c, n in counts.items()}


def _consequences(path, q, counts, ids_sha, verdict="COARSE"):
    per_class = sel.e4c.allocate(q, counts) if verdict == "COARSE" else None
    path.write_text(json.dumps({"verdict": verdict, "v1": {"q_star": q}, "v3_pool_counts": counts,
                                "v3_pool_ids_sha256": ids_sha, "count": {"per_class": per_class}}))
    return path


def test_run_writes_c_and_every_d_draw_with_their_hashes(tmp_path, monkeypatch):
    candidates, scores, report, safe_ids, counts = _write_inputs(tmp_path, monkeypatch)
    cons = _consequences(tmp_path / "e4_consequences.json", 30, counts, sel.e4.ids_sha256(safe_ids))
    m = sel.run("ns", candidates, scores, report, cons, tmp_path / "out")
    out = tmp_path / "out" / "ns"
    seeds = m["stage4_seeds"]
    assert len(seeds) == 20
    assert set(m["files_sha256"]) == {sel.C_NAME} | {sel.D_TEMPLATE.format(seed=s) for s in seeds}
    for name, sha in m["files_sha256"].items():
        assert sel.sha256_file(out / name) == sha
    c = pd.read_csv(out / sel.C_NAME)
    assert len(c) == m["n_selected_c"] == 30 and c["dx"].value_counts().to_dict() == m["per_class"]
    assert not set(c["image_id"]) & {"df_000", "mel_001"}
    for s in seeds:
        d = pd.read_csv(out / sel.D_TEMPLATE.format(seed=s))
        assert d["dx"].value_counts().to_dict() == m["per_class"] and d["image_id"].is_unique
        assert not set(d["image_id"]) & {"df_000", "mel_001"}
    assert m["safe_pool_ids_sha256"] == sel.e4.ids_sha256(safe_ids)
    assert m["ranking"]["safety_only"] == ["iqa"] and m["ranking"]["not_scored_no_direction"] == ["uncertainty"]


def test_run_refuses_a_pool_with_the_e4_counts_but_other_ids(tmp_path, monkeypatch):
    candidates, scores, report, safe_ids, counts = _write_inputs(tmp_path, monkeypatch)
    other = sel.e4.ids_sha256(safe_ids[:-1] + ["nv_999"])
    for ids_sha in (other, None):
        with pytest.raises(sel.SelectionError, match="not the pool E4 measured"):
            sel.run("ns", candidates, scores, report, _consequences(tmp_path / "c.json", 30, counts, ids_sha),
                    tmp_path / "o")


def test_run_under_no_writes_only_the_manifest_and_under_go_refuses(tmp_path, monkeypatch):
    candidates, scores, report, safe_ids, counts = _write_inputs(tmp_path, monkeypatch)
    sha = sel.e4.ids_sha256(safe_ids)
    m = sel.run("ns", candidates, scores, report, _consequences(tmp_path / "n.json", 0, counts, sha, "NO"),
                tmp_path / "o")
    assert m["n_selected_c"] == 0
    assert sorted(p.name for p in (tmp_path / "o" / "ns").iterdir()) == [sel.MANIFEST_NAME]
    with pytest.raises(sel.SelectionError, match="E4b"):
        sel.run("ns", candidates, scores, report, _consequences(tmp_path / "g.json", 3168, counts, sha, "GO"),
                tmp_path / "g")
