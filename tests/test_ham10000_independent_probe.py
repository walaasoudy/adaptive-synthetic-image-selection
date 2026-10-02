#!/usr/bin/env python3
"""P1, the independent probe of Amendment 3: the probe learns separable classes and is deterministic,
its settings are the amendment's, the gate and the decision table, the synthetic pool never reaches
training, and end to end it reads synthetic mel placed on real nv as influential and synthetic mel
placed on real mel as not.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.asism import ham10000_independent_probe as probe  # noqa: E402
from scripts.asism import ham10000_v2_signal_diagnostics as diag  # noqa: E402
from scripts.utils.manifest import sha256_file, write_json  # noqa: E402

DIM = 24
N_REAL = 50   # per class
N_SYN = 40    # per class (>= the separability loader's 38)


def _centres(rng):
    return 4.0 * rng.standard_normal((len(probe.CLASSES), DIM))


def _cloud(rng, centre, n, spread=0.5):
    return centre + spread * rng.standard_normal((n, DIM))


def _gate(recalls):
    return probe.gate({name: (int(round(r * 1000)), 1000) for name, r in zip(probe.CLASSES, recalls)})


def _q2(mel_to_nv, nv_sink):
    def entry(rate):
        return {"k": int(rate * 100), "n": 100, "rate": rate, "ci95": [rate, rate]}
    influential = (mel_to_nv >= diag.CRITERIA["q2_max_mel_to_nv_share"]
                   or nv_sink >= diag.CRITERIA["q2_max_nv_share_of_mismatches"])
    return {"mel_predicted_nv": entry(mel_to_nv), "nv_share_of_mismatches": entry(nv_sink),
            "mismatches_predicted_as": {}, "influential": influential}


# ----------------------------------------------------------------------------------------------
# The probe and its settings
# ----------------------------------------------------------------------------------------------


def test_settings_are_the_amendments():
    assert probe.L2_PENALTY == 1e-4
    assert probe.N_FOLDS == 5 and probe.SEED == 42 and probe.N_BOOTSTRAP == 1000
    assert probe.LBFGS_TOLERANCE == 1e-9 and probe.LBFGS_MAX_ITERATIONS == 1000
    config = yaml.safe_load((REPO / "configs/ham10000_stage3.yaml").read_text(encoding="utf-8"))
    assert config["auxiliary_classifier_v3"]["weight_decay"] == probe.L2_PENALTY
    assert config["auxiliary_classifier_v3"]["class_weighting"] == "inverse_frequency"
    accept = config["auxiliary_classifier_v3"]["acceptance"]["min_balanced_accuracy_exclusive"]
    assert accept == probe.GATE_MIN_BALANCED_ACCURACY_EXCLUSIVE


def test_probe_learns_separable_classes_and_is_deterministic():
    rng = np.random.default_rng(0)
    centres = _centres(rng)
    x = probe.l2_normalise(np.concatenate([_cloud(rng, c, N_REAL) for c in centres]))
    y = np.repeat(np.arange(len(probe.CLASSES)), N_REAL)
    w1, b1 = probe.fit_probe(x, y, probe._class_weights(y))
    w2, b2 = probe.fit_probe(x, y, probe._class_weights(y))
    assert np.array_equal(w1, w2) and np.array_equal(b1, b2)
    assert (probe.predict(x, w1, b1) == y).mean() > 0.95


def test_class_weights_are_inverse_frequency_with_mean_one():
    y = np.array([0] * 6 + [1] * 3 + [2] * 1 + [3, 4, 5, 6])
    weights = probe._class_weights(y)
    assert weights.mean() == pytest.approx(1.0)
    assert weights[0] * 6 == pytest.approx(weights[1] * 3) == pytest.approx(weights[2])


# ----------------------------------------------------------------------------------------------
# Gate and decision table
# ----------------------------------------------------------------------------------------------


def test_gate_needs_balanced_accuracy_above_0478_and_no_zero_recall():
    assert _gate([0.6] * 7)["passed"]
    assert not _gate([0.478] * 7)["passed"]                     # exclusive
    failed = _gate([0.9] * 6 + [0.0])
    assert not failed["passed"] and failed["zero_recall_classes"] == [probe.CLASSES[-1]]


@pytest.mark.parametrize("mel_to_nv, nv_sink, verdict, one", [
    (0.10, 0.40, "classifier_specific", None),
    (0.85, 0.89, "class_fidelity_likelier", False),
    (0.10, 0.60, "class_fidelity_likelier", True),
    (0.40, 0.20, "class_fidelity_likelier", True),
])
def test_decision_table(mel_to_nv, nv_sink, verdict, one):
    decision = probe.decide_p1(_gate([0.6] * 7), _q2(mel_to_nv, nv_sink))
    assert decision["verdict"] == verdict
    if one is not None:
        assert decision["rests_on_one_condition"] is one


def test_a_failed_gate_is_not_informative_whatever_q2_says():
    for q2 in (_q2(0.1, 0.1), _q2(0.9, 0.9)):
        assert probe.decide_p1(_gate([0.3] * 7), q2)["verdict"] == "not_informative"


# ----------------------------------------------------------------------------------------------
# End to end on a fixture
# ----------------------------------------------------------------------------------------------


def _workspace(tmp_path: Path, mel_like: str, seed: int = 1) -> dict:
    """Real classes as separate clouds; synthetic classes on their real cloud, except synthetic mel,
    which sits on the real `mel_like` cloud."""
    rng = np.random.default_rng(seed)
    centres = dict(zip(probe.CLASSES, _centres(rng)))
    xs, ids, dx, source, meta = [], [], [], [], []
    for name in probe.CLASSES:
        for i in range(N_REAL):
            image_id = f"ISIC_{name}_{i:03d}"
            xs.append(_cloud(rng, centres[name], 1)[0]); ids.append(image_id); dx.append(name); source.append("real")
            meta.append({"lesion_id": f"HAM_{name}_{i // 2:03d}", "image_id": image_id, "dx": name})
    candidates = []
    for name in probe.CLASSES:
        target = mel_like if name == "mel" else name
        for i in range(N_SYN):
            image_id = f"syn_{name}_{i:05d}"
            xs.append(_cloud(rng, centres[target], 1)[0]); ids.append(image_id); dx.append(name)
            source.append("synthetic"); candidates.append({"image_id": image_id, "dx": name, "seed": i})
    emb = tmp_path / "emb"; emb.mkdir()
    np.savez(emb / "embeddings.npz", embeddings=np.array(xs, dtype=np.float32), image_id=np.array(ids),
             dx=np.array(dx), source=np.array(source))
    write_json(emb / "provenance.json", {"embeddings_sha256": sha256_file(emb / "embeddings.npz")})
    pd.DataFrame(meta).to_csv(tmp_path / "meta.csv", index=False)
    pd.DataFrame(candidates).to_csv(tmp_path / "cand.csv", index=False)
    return {"emb": emb, "meta": tmp_path / "meta.csv", "cand": tmp_path / "cand.csv",
            "n": len(candidates), "ids": [c["image_id"] for c in candidates]}


def _fake_v3(ids):
    """Stand-in for the v3 artifacts: every synthetic mel read as nv except the first five."""
    rows = []
    for k, image_id in enumerate(ids):
        name = image_id.split("_")[1]
        read_nv = name == "mel" and int(image_id.split("_")[2]) >= 5
        predicted = "nv" if read_nv else name
        rows.append({"image_id": image_id, "agreement_intended_diagnosis": name,
                     "agreement_predicted_diagnosis": predicted,
                     "agreement_is_argmax_match": predicted == name,
                     "agreement_intended_prob": 0.3 if read_nv else 0.8,
                     "agreement_best_rival_diagnosis": "nv" if read_nv else "bkl",
                     "agreement_best_rival_prob": 0.6 if read_nv else 0.1,
                     "agreement_margin": -0.3 if read_nv else 0.7})
    return pd.DataFrame(rows)


@pytest.fixture
def fake_v3(monkeypatch):
    """Replaces the v3 loader (its model-id and hash checks are tested with the diagnostic); the
    frame is also written where run() hashes it."""
    class Holder(dict):
        def __setitem__(self, key, frame):
            super().__setitem__(key, frame)
            frame.to_parquet(self.dir / "agreement_scores.parquet")
    holder = Holder()
    monkeypatch.setattr(probe.diag, "load_version", lambda directory, version, expected_n: holder["frame"])
    return holder


@pytest.mark.parametrize("mel_like, influential, verdict", [
    ("nv", True, "class_fidelity_likelier"),
    ("mel", False, "classifier_specific"),
])
def test_end_to_end(tmp_path, fake_v3, mel_like, influential, verdict):
    fake_v3.dir = tmp_path
    ws = _workspace(tmp_path, mel_like)
    fake_v3["frame"] = _fake_v3(ws["ids"])
    report = probe.run(ws["emb"], ws["meta"], tmp_path, ws["cand"], expected_n=ws["n"])
    assert report["gate"]["passed"]
    assert report["q2"]["influential"] is influential
    assert report["decision"]["verdict"] == verdict
    d1 = report["descriptive"]["d1"]
    assert d1["n"] == N_SYN - 5 and d1["mel_certainly_runner_up"]["rate"] == 1.0   # 0.3 > 1 - 0.6 - 0.3
    assert report["descriptive"]["d2"]["n_read_mel"] == 5
    assert "Decision" in probe.render_markdown(report)


def test_the_synthetic_pool_never_reaches_training(tmp_path, fake_v3):
    fake_v3.dir = tmp_path
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    a, b = _workspace(tmp_path / "a", "nv"), _workspace(tmp_path / "b", "mel")
    fake_v3["frame"] = _fake_v3(a["ids"])
    ra = probe.run(a["emb"], a["meta"], tmp_path, a["cand"], expected_n=a["n"])
    rb = probe.run(b["emb"], b["meta"], tmp_path, b["cand"], expected_n=b["n"])
    # Same seed, same real images; only synthetic mel moved. The real-side results are identical.
    assert ra["oof_recall_counts"] == rb["oof_recall_counts"]
    assert ra["settings"]["class_weights"] == rb["settings"]["class_weights"]


def test_refuses_metadata_that_disagrees_with_the_embeddings(tmp_path, fake_v3):
    fake_v3.dir = tmp_path
    ws = _workspace(tmp_path, "nv")
    meta = pd.read_csv(ws["meta"])
    meta.loc[0, "dx"] = "df" if meta.loc[0, "dx"] != "df" else "nv"
    meta.to_csv(ws["meta"], index=False)
    fake_v3["frame"] = _fake_v3(ws["ids"])
    with pytest.raises(SystemExit, match="metadata dx disagrees"):
        probe.run(ws["emb"], ws["meta"], tmp_path, ws["cand"], expected_n=ws["n"])


def test_refuses_a_candidate_list_that_differs_from_the_embeddings(tmp_path, fake_v3):
    fake_v3.dir = tmp_path
    ws = _workspace(tmp_path, "nv")
    pd.read_csv(ws["cand"]).iloc[1:].to_csv(ws["cand"], index=False)
    fake_v3["frame"] = _fake_v3(ws["ids"])
    with pytest.raises(SystemExit, match="do not hold the same images"):
        probe.run(ws["emb"], ws["meta"], tmp_path, ws["cand"], expected_n=ws["n"])
