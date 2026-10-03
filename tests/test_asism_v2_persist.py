"""A saved ranker selects exactly as the fitted one did. Toy data only."""
import json

import pytest

from scripts.asism_v2.persist import load_fitted, save_fitted
from scripts.asism_v2.pipeline import fit_ranker
from scripts.asism_v2.stopping import progressive_select
from test_asism_v2_stopping import SEEDS, _toy

PROVENANCE = {"prereg_sha256": "p" * 64, "protocol_sha256": "q" * 64}


@pytest.fixture(scope="module")
def fitted_and_pool():
    frame, subsets, measurements, protocol, pool = _toy()
    return fit_ranker(frame, subsets, measurements, SEEDS, protocol, max_epochs=300, bootstrap=20), pool


def test_a_loaded_ranker_scores_and_selects_identically(fitted_and_pool, tmp_path):
    fitted, pool = fitted_and_pool
    path = tmp_path / "ranker.json"
    content = save_fitted(path, fitted, PROVENANCE)
    loaded, provenance = load_fitted(path, PROVENANCE)
    assert provenance == {**PROVENANCE, "content_sha256": content}
    assert loaded.classes == fitted.classes and loaded.history == fitted.history
    assert len(loaded.ensemble) == 20 and loaded.normalizer == fitted.normalizer
    assert loaded.score_frame(pool).equals(fitted.score_frame(pool))
    before, after = progressive_select(fitted, pool), progressive_select(loaded, pool)
    assert before["counts"] == after["counts"] and before["trajectory"] == after["trajectory"]
    assert before["stops"] == after["stops"]


def test_a_saved_ranker_is_never_replaced(fitted_and_pool, tmp_path):
    fitted, _ = fitted_and_pool
    save_fitted(tmp_path / "ranker.json", fitted, PROVENANCE)
    with pytest.raises(FileExistsError):
        save_fitted(tmp_path / "ranker.json", fitted, PROVENANCE)


def test_a_changed_or_stale_file_is_refused(fitted_and_pool, tmp_path):
    fitted, _ = fitted_and_pool
    path = tmp_path / "ranker.json"
    save_fitted(path, fitted, PROVENANCE)
    with pytest.raises(ValueError, match="stale ranker, prereg_sha256"):
        load_fitted(path, {"prereg_sha256": "x" * 64})
    payload = json.loads(path.read_text())
    payload["model"]["log_count"] = 1.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="content was changed"):
        load_fitted(path, PROVENANCE)
