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


# ----------------------------------------------------------------------------------------------
# P1b (Amendment 4)
# ----------------------------------------------------------------------------------------------

ENCODER = {"encoder": "vit_small_patch14_dinov2", "weights_repo_id": "timm/x",
           "weights_revision": "936966a8", "weights_sha256": "abc"}


def _p1b_workspace(tmp_path: Path, mel_like: str, encoder=None, limit=None, seed: int = 3) -> dict:
    """The separability file (synthetic as in _workspace) plus an embed-real file whose
    classifier_train and classifier_val clouds sit on the same class centres."""
    ws = _workspace(tmp_path, mel_like, seed=seed)
    write_json(ws["emb"] / "provenance.json",
               {"embeddings_sha256": sha256_file(ws["emb"] / "embeddings.npz"), **ENCODER})
    data = np.load(ws["emb"] / "embeddings.npz")
    real = data["source"] == "real"
    x, dx = data["embeddings"][real], data["dx"][real]
    rng = np.random.default_rng(seed + 1)
    noise = 0.5 * rng.standard_normal(x.shape)
    split = np.where(np.arange(len(x)) % 2 == 0, probe.P1B_TRAIN_SPLIT, probe.P1B_GATE_SPLIT)
    real_dir = tmp_path / "real"; real_dir.mkdir()
    np.savez(real_dir / "real_embeddings.npz", embeddings=(x + noise).astype(np.float32),
             image_id=np.array([f"R{i}" for i in range(len(x))]), dx=dx, split=split)
    write_json(real_dir / "provenance.json", {"embeddings_sha256": sha256_file(real_dir / "real_embeddings.npz"),
                                              "limit": limit, **(encoder or ENCODER)})
    return {**ws, "real": real_dir}


@pytest.mark.parametrize("mel_like, verdict", [("nv", "class_fidelity_likelier"), ("mel", "classifier_specific")])
def test_p1b_end_to_end(tmp_path, mel_like, verdict):
    ws = _p1b_workspace(tmp_path, mel_like)
    report = probe.run_p1b(ws["emb"], ws["real"], ws["cand"], expected_n=ws["n"])
    assert report["gate"]["passed"]
    assert report["decision"]["verdict"] == verdict
    assert report["reading_with_p1"] == probe.P1B_READING[verdict]
    assert report["inputs"]["n_train"] + report["inputs"]["n_gate"] == N_REAL * len(probe.CLASSES)
    text = probe.render_markdown(report)
    assert "P1b" in text and "Amendment 4" in text and "Descriptive" not in text


def test_p1b_reading_table_covers_every_verdict():
    assert set(probe.P1B_READING) == {"classifier_specific", "class_fidelity_likelier", "not_informative"}
    assert probe.P1B_TRAIN_SPLIT == "classifier_train" and probe.P1B_GATE_SPLIT == "classifier_val"


def test_p1b_refuses_embeddings_from_another_encoder(tmp_path):
    ws = _p1b_workspace(tmp_path, "nv", encoder={**ENCODER, "weights_revision": "main"})
    with pytest.raises(SystemExit, match=r"differ in \['weights_revision'\]"):
        probe.run_p1b(ws["emb"], ws["real"], ws["cand"], expected_n=ws["n"])


def test_p1b_refuses_a_smoke_run(tmp_path):
    ws = _p1b_workspace(tmp_path, "nv", limit=14)
    with pytest.raises(SystemExit, match="smoke run"):
        probe.run_p1b(ws["emb"], ws["real"], ws["cand"], expected_n=ws["n"])


def test_p1b_refuses_real_embeddings_that_do_not_match_their_hash(tmp_path):
    ws = _p1b_workspace(tmp_path, "nv")
    (ws["real"] / "real_embeddings.npz").write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="does not match its provenance hash"):
        probe.run_p1b(ws["emb"], ws["real"], ws["cand"], expected_n=ws["n"])


@pytest.fixture
def embed_pipeline(monkeypatch):
    from fixture_workspace import fixture_workspace
    from test_ham10000_cam_reference_build import NAMESPACE, _overlay, _workspace as _cam_workspace, _write_images

    from scripts.asism import ham10000_01_compute_signals as signals

    with fixture_workspace("probe-embed-real") as path:
        layout = _cam_workspace(path)
        for split in (probe.P1B_TRAIN_SPLIT, probe.P1B_GATE_SPLIT):
            rows = [{"image_id": f"ISIC_{split}_{name}_{i}", "lesion_id": f"HAM_{split}_{name}_{i}", "dx": name}
                    for name in probe.CLASSES for i in range(2)]
            pd.DataFrame(rows).to_csv(layout["splits_dir"] / f"{split}.csv", index=False)
            _write_images(layout["images_dir"] / split, [r["image_id"] for r in rows][:-1])  # one absent
        monkeypatch.setenv("PROJECT_ROOT", str(path))
        monkeypatch.setenv("THESIS_CONFIG_OVERLAY", str(_overlay(path)))
        weights = path / "weights.safetensors"
        weights.write_bytes(b"fake")
        seen = {}
        monkeypatch.setattr(signals, "load_encoder", lambda cfg, device: (None, None, weights))

        def fake_embed(encoder, transform, paths, batch_size, device):
            seen["paths"] = list(paths)
            return np.ones((len(paths), 4))
        monkeypatch.setattr(signals, "embed_images", fake_embed)
        yield {"root": path, "namespace": NAMESPACE, "seen": seen, **layout}


def test_embed_real_embeds_the_classifier_splits_into_its_own_directory(embed_pipeline):
    result = probe.embed_real(embed_pipeline["namespace"], device="cpu")
    out_dir = embed_pipeline["root"] / probe.PROBE_OUTPUT_SUBDIR / embed_pipeline["namespace"]
    assert Path(result["out_dir"]) == out_dir
    data = np.load(out_dir / "real_embeddings.npz")
    assert set(data["split"]) == {probe.P1B_TRAIN_SPLIT, probe.P1B_GATE_SPLIT}
    # records_from_split skips the absent image, as v3's training did, and the count says so
    assert result["splits"][probe.P1B_TRAIN_SPLIT] == {"in_split": 14, "embedded": 13}
    assert all(Path(p).parent.name in (probe.P1B_TRAIN_SPLIT, probe.P1B_GATE_SPLIT) for p in embed_pipeline["seen"]["paths"])
    assert result["embeddings_sha256"] == sha256_file(out_dir / "real_embeddings.npz")
    assert not (embed_pipeline["root"] / "outputs/ham10000/diagnostics/synthetic_separability").exists()


def test_embed_real_refuses_final_eval_images(embed_pipeline):
    splits = embed_pipeline["splits_dir"]
    leaked = pd.read_csv(splits / f"{probe.P1B_GATE_SPLIT}.csv")["image_id"].iloc[0]
    final_eval = pd.read_csv(splits / "final_eval_heldout.csv")
    final_eval.loc[len(final_eval)] = {"image_id": leaked, "lesion_id": "HAM_x", "dx": "nv"}
    final_eval.to_csv(splits / "final_eval_heldout.csv", index=False)
    with pytest.raises(SystemExit, match="final_eval_heldout"):
        probe.embed_real(embed_pipeline["namespace"], device="cpu")
