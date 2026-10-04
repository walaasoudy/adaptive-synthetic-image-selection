"""The per-class threshold report reads a selection and changes nothing in it. Toy data only."""
import hashlib
import json

import pandas as pd
import pytest

from scripts.asism_v2.pipeline import fit_ranker
from scripts.asism_v2.selection_files import MANIFEST_NAME, TRAJECTORY_NAME, build_selection, write_selection
from scripts.followup.ham10000_asism_v2_threshold_report import (COLUMNS, REPORT_CSV, REPORT_JSON,
                                                                  ThresholdReportError, build_report,
                                                                  write_report)
from test_asism_v2_stopping import SEEDS, _toy

STAGE4_SEEDS = [42, 43]


@pytest.fixture(scope="module")
def selection_and_pool():
    frame, subsets, measurements, protocol, pool = _toy()
    fitted = fit_ranker(frame, subsets, measurements, SEEDS, protocol, max_epochs=300, bootstrap=20)
    pool = pool.assign(image_path=[f"/synthetic/{i}.png" for i in pool.image_id])
    return build_selection(fitted, pool, STAGE4_SEEDS, n_candidates=len(pool) + 1), pool


def _written(selection_and_pool, directory):
    selection, pool = selection_and_pool
    manifest = write_selection(directory, selection, pool, "ns", STAGE4_SEEDS, {"safe_pool_ids_sha256": "s" * 64})
    return manifest, {str(k): int(v) for k, v in pool.dx.value_counts().items()}


def _hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())}


def test_the_report_states_the_cut_the_trajectory_holds(selection_and_pool, tmp_path):
    manifest, sizes = _written(selection_and_pool, tmp_path)
    report = build_report(tmp_path, sizes)
    trajectory = pd.read_csv(tmp_path / TRAJECTORY_NAME)
    assert [row["dx"] for row in report["per_class"]] == sorted(sizes)
    assert sum(row["selected"] for row in report["per_class"]) == manifest["n_selected_c"]
    for row in report["per_class"]:
        accepted = trajectory[trajectory.dx == row["dx"]]
        assert row["selected"] == manifest["per_class"][row["dx"]] == len(accepted)
        assert row["candidates"] == sizes[row["dx"]]
        assert row["stop_reason"] == manifest["stops"][row["dx"]]["reason"]
        if not len(accepted):
            assert row["score_threshold"] is None and row["lower_bound_at_threshold"] is None
            continue
        last = accepted.loc[accepted.rank_in_class.idxmax()]
        assert row["last_accepted_image_id"] == last.image_id
        assert row["score_threshold"] == pytest.approx(last.ranking_score)
        assert row["lower_bound_at_threshold"] == pytest.approx(last.own_lower_bound)
        assert row["min_score_selected"] == pytest.approx(accepted.ranking_score.min())
        assert row["fraction_selected"] == pytest.approx(len(accepted) / sizes[row["dx"]])
    assert report["final_eval_heldout_read"] is False and report["classifier_val_read"] is False


def test_the_selection_files_are_untouched_and_the_report_is_written_once(selection_and_pool, tmp_path):
    _, sizes = _written(selection_and_pool, tmp_path)
    before = _hashes(tmp_path)
    report = build_report(tmp_path, sizes)
    assert _hashes(tmp_path) == before
    write_report(tmp_path, report)
    after = _hashes(tmp_path)
    assert {name: after[name] for name in before} == before
    assert set(after) - set(before) == {REPORT_JSON, REPORT_CSV}
    assert list(pd.read_csv(tmp_path / REPORT_CSV).columns) == COLUMNS
    saved = json.loads((tmp_path / REPORT_JSON).read_text())
    assert saved["csv_sha256"] == after[REPORT_CSV] and saved["per_class"] == report["per_class"]
    assert saved["trajectory_sha256"] == before[TRAJECTORY_NAME]
    with pytest.raises(ThresholdReportError, match="written once"):
        write_report(tmp_path, report)


def test_a_changed_trajectory_or_another_pool_is_refused(selection_and_pool, tmp_path):
    _, sizes = _written(selection_and_pool, tmp_path)
    first = sorted(sizes)[0]
    with pytest.raises(ThresholdReportError, match="safe pool"):
        build_report(tmp_path, {**sizes, first: sizes[first] + 1})
    path = tmp_path / TRAJECTORY_NAME
    path.write_text(path.read_text() + " ")
    with pytest.raises(ThresholdReportError, match="not the file"):
        build_report(tmp_path, sizes)


def test_without_a_subset_selection_there_is_nothing_to_report(selection_and_pool, tmp_path):
    selection, pool = selection_and_pool
    with pytest.raises(ThresholdReportError, match="select phase"):
        build_report(tmp_path, {})
    nothing = {**selection, "outcome": "none", "c_ids": [], "d_ids": {}, "counts": {k: 0 for k in selection["counts"]}}
    write_selection(tmp_path, nothing, pool, "ns", STAGE4_SEEDS, {})
    assert json.loads((tmp_path / MANIFEST_NAME).read_text())["selection_outcome"] == "none"
    with pytest.raises(ThresholdReportError, match="no cut to report"):
        build_report(tmp_path, {str(k): int(v) for k, v in pool.dx.value_counts().items()})
