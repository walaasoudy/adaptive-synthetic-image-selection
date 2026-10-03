"""Condition C and its matched random D as Stage 4 reads them; the all / none outcomes. Toy data."""
import hashlib
import json

import pandas as pd
import pytest

from scripts.asism_v2 import selection_files
from scripts.asism_v2.pipeline import fit_ranker
from scripts.asism_v2.selection_files import (C_NAME, D_TEMPLATE, MANIFEST_NAME, build_selection,
                                              draw_matched_random, write_selection)
from scripts.asism_v2.stopping import progressive_select
from test_asism_v2_stopping import SEEDS, _toy

STAGE4_SEEDS = [42, 43, 44]


@pytest.fixture(scope="module")
def fitted_and_pool():
    frame, subsets, measurements, protocol, pool = _toy()
    fitted = fit_ranker(frame, subsets, measurements, SEEDS, protocol, max_epochs=300, bootstrap=20)
    return fitted, pool.assign(image_path=[f"/synthetic/{i}.png" for i in pool.image_id])


def test_c_is_the_stopping_rules_output_and_d_matches_its_counts(fitted_and_pool):
    fitted, pool = fitted_and_pool
    selection = build_selection(fitted, pool, STAGE4_SEEDS, n_candidates=len(pool))
    reference = progressive_select(fitted, pool)
    assert selection["outcome"] == "subset" and selection["counts"] == reference["counts"]
    assert selection["c_ids"] == list(reference["selected"]["image_id"])
    dx = pool.set_index("image_id").dx
    for seed in STAGE4_SEEDS:
        drawn = selection["d_ids"][seed]
        assert len(set(drawn)) == len(drawn)
        assert dx.loc[drawn].value_counts().to_dict() == {k: v for k, v in selection["counts"].items() if v}
    assert selection["d_ids"][42] != selection["d_ids"][43]
    assert selection["d_ids"][42] == draw_matched_random(pool, selection["counts"], 42)


def test_the_files_are_what_stage4_reads_and_are_written_once(fitted_and_pool, tmp_path):
    fitted, pool = fitted_and_pool
    selection = build_selection(fitted, pool, STAGE4_SEEDS, n_candidates=len(pool))
    manifest = write_selection(tmp_path, selection, pool, "ns", STAGE4_SEEDS, {"prereg_sha256": "p" * 64})
    c = pd.read_csv(tmp_path / C_NAME)
    assert list(c.columns) == ["image_id", "image_path", "dx"] and len(c) == manifest["n_selected_c"]
    # the hash ham10000_train_conditions.build_condition_records writes into each run manifest
    assert manifest["c_ids_sha256"] == hashlib.sha256("\n".join(sorted(c.image_id)).encode()).hexdigest()
    assert manifest["protocol"] == "asism_v2_learned" and manifest["selection_outcome"] == "subset"
    assert manifest["namespace"] == "ns" and manifest["stage4_seeds"] == STAGE4_SEEDS
    assert manifest["final_eval_heldout_read"] is False and manifest["classifier_val_read"] is False
    for seed in STAGE4_SEEDS:
        name = D_TEMPLATE.format(seed=seed)
        d = pd.read_csv(tmp_path / name)
        assert d.dx.value_counts().to_dict() == c.dx.value_counts().to_dict()
        assert manifest["files_sha256"][name] == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    assert json.loads((tmp_path / MANIFEST_NAME).read_text()) == manifest
    with pytest.raises(ValueError, match="made once"):
        write_selection(tmp_path, selection, pool, "ns", STAGE4_SEEDS, {})


@pytest.mark.parametrize("outcome", ["all", "none"])
def test_all_and_none_are_recorded_as_results_and_build_no_files(fitted_and_pool, tmp_path, monkeypatch, outcome):
    fitted, pool = fitted_and_pool
    real = progressive_select(fitted, pool)
    kept = pool if outcome == "all" else pool.iloc[:0]
    forced = {**real, "selected": kept[["image_id", "dx"]].assign(score_source="heldout"),
              "counts": {dx: int((kept.dx == dx).sum()) for dx in fitted.classes}, "trajectory": []}
    monkeypatch.setattr(selection_files, "progressive_select", lambda *_: forced)
    selection = build_selection(fitted, pool, STAGE4_SEEDS, n_candidates=len(pool))
    assert selection["outcome"] == outcome and selection["d_ids"] == {}
    manifest = write_selection(tmp_path, selection, pool, "ns", STAGE4_SEEDS, {})
    assert manifest["protocol"] == "asism_v2_learned_all_or_none" and manifest["files_sha256"] == {}
    assert manifest["n_selected_c"] == len(kept) and manifest["n_selected_d"] == 0
    assert [p.name for p in tmp_path.iterdir()] == [MANIFEST_NAME]


def test_every_safe_image_kept_is_not_all_when_the_safety_filter_removed_some(fitted_and_pool, monkeypatch):
    fitted, pool = fitted_and_pool
    forced = {**progressive_select(fitted, pool), "selected": pool[["image_id", "dx"]].assign(score_source="heldout"),
              "counts": {dx: int((pool.dx == dx).sum()) for dx in fitted.classes}}
    monkeypatch.setattr(selection_files, "progressive_select", lambda *_: forced)
    assert build_selection(fitted, pool, STAGE4_SEEDS, n_candidates=len(pool) + 5)["outcome"] == "subset"
