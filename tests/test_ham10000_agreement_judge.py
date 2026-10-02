#!/usr/bin/env python3
"""J1, the agreement judge of Amendment 6: the sink and restatement criteria, lesion disjointness of
its training and validation data, a deterministic judge id, an agreement artifact in the signal
schema, and end to end the acceptance follows the criteria (synthetic mel on real mel accepted,
synthetic mel on real nv not)."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.asism import ham10000_agreement_judge as judge  # noqa: E402
from scripts.utils.manifest import sha256_file, write_json  # noqa: E402

CLASSES = judge.CLASSES
DIM = 24
ENCODER = {"encoder": "vit_small_patch14_dinov2", "weights_repo_id": "timm/x",
           "weights_revision": "936966a8", "weights_sha256": "abc"}
N = {"gen": 40, "train": 20, "val": 20, "syn": 40}


# ----------------------------------------------------------------------------------------------
# V4, V5 and helpers
# ----------------------------------------------------------------------------------------------


def _frame(predicted: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for dx, preds in predicted.items():
        for k, p in enumerate(preds):
            rows.append({"image_id": f"{dx}_{k}", "dx": dx, "agreement_predicted_diagnosis": p,
                         "agreement_is_argmax_match": p == dx})
    return pd.DataFrame(rows)


def test_v4_flags_any_class_that_takes_half_the_mismatches():
    sink = judge.v4_sink(_frame({"mel": ["df"] * 6 + ["nv"] * 2, "bkl": ["df"] * 2 + ["bkl"] * 5}))
    assert sink["largest"] == "df" and sink["largest_share"] == pytest.approx(0.8) and not sink["passed"]
    spread = judge.v4_sink(_frame({"mel": ["df", "nv", "bkl", "bcc"], "bkl": ["akiec", "vasc", "mel"]}))
    assert spread["passed"] and spread["max_share_exclusive"] == 0.5


def test_v4_at_exactly_half_fails():
    assert not judge.v4_sink(_frame({"mel": ["df", "df", "nv", "bcc"]}))["passed"]


def _partner(ids, dx, values, column):
    return pd.DataFrame({"image_id": ids, column: values})


def test_v5_flags_a_within_class_restatement_with_opposite_signs():
    rng = np.random.default_rng(0)
    ids = [f"{c}_{k}" for c in CLASSES for k in range(30)]
    dx = [c for c in CLASSES for _ in range(30)]
    base = rng.normal(size=len(ids))
    sign = np.array([1 if CLASSES.index(c) % 2 else -1 for c in dx])
    agreement = pd.DataFrame({"image_id": ids, "dx": dx, "agreement_score": base})
    partners = {"similarity": _partner(ids, dx, sign * base, "similarity_knn_mean"),
                "uncertainty": _partner(ids, dx, rng.normal(size=len(ids)), "uncertainty_mutual_information"),
                "explainability": _partner(ids, dx, rng.normal(size=len(ids)), "explainability_calibrated_typicality")}
    v5 = judge.v5_not_a_restatement(agreement, partners)
    assert v5["pairs"]["agreement~similarity"]["redundant"]
    assert not v5["pairs"]["agreement~uncertainty"]["redundant"]
    assert not v5["passed"]
    assert (v5["min_abs_rho"], v5["min_classes"]) == (0.7, 4)


def test_softmax_rows_sum_to_one():
    p = judge.softmax(np.array([[1000.0, 0.0, -1000.0], [0.1, 0.2, 0.3]]))
    assert np.allclose(p.sum(axis=1), 1.0) and np.isfinite(p).all()


def test_judge_id_is_deterministic_and_input_sensitive():
    w, b = np.ones((3, 2)), np.zeros(2)
    a = judge.judge_id(w, b, {"x": "1"})
    assert a == judge.judge_id(w, b, {"x": "1"}) and a != judge.judge_id(w, b, {"x": "2"})
    assert a.startswith("ham10000-judge-j1:")


# ----------------------------------------------------------------------------------------------
# End to end on a fixture
# ----------------------------------------------------------------------------------------------


def _write_signal(folder: Path, signal: str, frame: pd.DataFrame, cand_sha: str):
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / f"{signal}_scores.parquet", index=False)
    write_json(folder / f"{signal}_scores.provenance.json",
               {"schema_version": 2, "signal": signal, "n_rows": len(frame), "candidates_csv_sha256": cand_sha,
                "git_commit_hash": "test", "parquet_sha256": sha256_file(folder / f"{signal}_scores.parquet")})


def _workspace(root: Path, mel_like: str, shared_lesion: bool = False, seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    centres = dict(zip(CLASSES, 0.35 * rng.standard_normal((len(CLASSES), DIM))))
    # Classes of increasing difficulty, so recall varies by class on real and synthetic alike and
    # Q1's rank correlation is defined.
    spread = {c: 0.15 + 0.12 * i for i, c in enumerate(CLASSES)}
    cloud = lambda c, n: centres[c] + spread[c] * rng.standard_normal((n, DIM))  # noqa: E731
    sep_x, sep_ids, sep_dx, sep_src, real_x, real_ids, real_dx, split, meta = [], [], [], [], [], [], [], [], []
    for c in CLASSES:
        for kind, n in (("gen", N["gen"]), ("train", N["train"]), ("val", N["val"])):
            x = cloud(c, n)
            ids = [f"ISIC_{kind}_{c}_{k}" for k in range(n)]
            meta += [{"image_id": i, "lesion_id": f"L_{i}", "dx": c} for i in ids]
            if kind == "gen":
                sep_x.append(x); sep_ids += ids; sep_dx += [c] * n; sep_src += ["real"] * n
            else:
                real_x.append(x); real_ids += ids; real_dx += [c] * n
                split += ["classifier_train" if kind == "train" else "classifier_val"] * n
    if shared_lesion:
        meta[-1]["lesion_id"] = meta[0]["lesion_id"]   # a classifier_val image shares a gen_train lesion
    cand = []
    for c in CLASSES:
        ids = [f"syn_{c}_{k:05d}" for k in range(N["syn"])]
        sep_x.append(cloud(mel_like if c == "mel" else c, N["syn"])); sep_ids += ids
        sep_dx += [c] * N["syn"]; sep_src += ["synthetic"] * N["syn"]
        cand += [{"image_id": i, "dx": c, "seed": k} for k, i in enumerate(ids)]
    sep, real = root / "sep", root / "real"
    sep.mkdir(parents=True); real.mkdir()
    np.savez(sep / "embeddings.npz", embeddings=np.concatenate(sep_x).astype(np.float32), image_id=np.array(sep_ids),
             dx=np.array(sep_dx), source=np.array(sep_src))
    write_json(sep / "provenance.json", {"embeddings_sha256": sha256_file(sep / "embeddings.npz"), **ENCODER})
    np.savez(real / "real_embeddings.npz", embeddings=np.concatenate(real_x).astype(np.float32),
             image_id=np.array(real_ids), dx=np.array(real_dx), split=np.array(split))
    write_json(real / "provenance.json", {"embeddings_sha256": sha256_file(real / "real_embeddings.npz"),
                                          "limit": None, **ENCODER})
    pd.DataFrame(meta).to_csv(root / "meta.csv", index=False)
    pd.DataFrame(cand).to_csv(root / "cand.csv", index=False)
    sha = sha256_file(root / "cand.csv")
    syn_ids = [r["image_id"] for r in cand]
    _write_signal(root / "sim", "similarity", pd.DataFrame({"image_id": syn_ids, "similarity_knn_mean": rng.random(len(syn_ids))}), sha)
    _write_signal(root / "sim", "iqa", pd.DataFrame({"image_id": syn_ids, "iqa_composite": rng.random(len(syn_ids))}), sha)
    v3 = pd.DataFrame({"image_id": syn_ids, "uncertainty_mutual_information": rng.random(len(syn_ids)),
                       "explainability_calibrated_typicality": rng.random(len(syn_ids))})
    _write_signal(root / "v3", "uncertainty", v3[["image_id", "uncertainty_mutual_information"]], sha)
    _write_signal(root / "v3", "explainability", v3[["image_id", "explainability_calibrated_typicality"]], sha)
    return {"sep": sep, "real": real, "meta": root / "meta.csv", "cand": root / "cand.csv", "sim": root / "sim",
            "v3": root / "v3", "out": root / "out", "n": len(cand), "v3_frame": v3}


@pytest.fixture
def stub_gate(monkeypatch):
    """The fixture's signal files are not real signal artifacts, so the gate's own checks are stubbed
    (they are tested in test_ham10000_gonogo.py); V6 reads whatever it returns for agreement."""
    holder = {"outcome": "include"}

    def evaluate(frames, diagnoses, scores_dir, config):
        assert set(frames) == {"similarity", "iqa", "uncertainty", "explainability", "agreement"}
        return {s: {"outcome": holder["outcome"] if s == "agreement" else "include", "reason": "stub", "checks": {}}
                for s in frames}
    monkeypatch.setattr(judge.gonogo, "evaluate", evaluate)
    return holder


@pytest.fixture
def v3_loader(monkeypatch):
    holder = {}
    monkeypatch.setattr(judge.diag, "load_version", lambda d, v, n: holder["frame"])
    return holder


@pytest.mark.parametrize("mel_like, accepted", [("mel", True), ("nv", False)])
def test_end_to_end_acceptance_follows_the_criteria(tmp_path, stub_gate, v3_loader, mel_like, accepted):
    ws = _workspace(tmp_path, mel_like)
    v3_loader["frame"] = ws["v3_frame"]
    report = judge.run(ws["sep"], ws["real"], ws["meta"], ws["cand"], ws["sim"], ws["v3"], ws["out"], expected_n=ws["n"])
    assert report["inputs"]["n_train"] == len(CLASSES) * (N["gen"] + N["train"])
    assert report["inputs"]["n_validation"] == len(CLASSES) * N["val"]
    assert report["criteria"]["V1_real_validity"]
    assert report["accepted"] is accepted
    if not accepted:
        assert not report["criteria"]["V3_q2_not_influential"]
        assert "left out of ASISM v2" in report["decision"]
    scores = pd.read_parquet(ws["out"] / "agreement_scores.parquet")
    assert len(scores) == ws["n"] and {"image_id", "agreement_score", "agreement_is_argmax_match"} <= set(scores.columns)
    prov = judge.read_json(ws["out"] / "agreement_scores.provenance.json")
    assert prov["judge_trained_on_splits"] == ["gen_train", "classifier_train"]
    assert prov["parquet_sha256"] == sha256_file(ws["out"] / "agreement_scores.parquet")
    assert np.array(report["confusion_validation"]).sum() == len(CLASSES) * N["val"]
    assert "J1" in judge.render_markdown(report)


def test_a_gate_exclusion_alone_rejects_the_judge(tmp_path, stub_gate, v3_loader):
    ws = _workspace(tmp_path, "mel")
    v3_loader["frame"] = ws["v3_frame"]
    stub_gate["outcome"] = "exclude"
    report = judge.run(ws["sep"], ws["real"], ws["meta"], ws["cand"], ws["sim"], ws["v3"], ws["out"], expected_n=ws["n"])
    assert not report["criteria"]["V6_gonogo_include"] and not report["accepted"]


def test_refuses_training_and_validation_that_share_a_lesion(tmp_path, stub_gate, v3_loader):
    ws = _workspace(tmp_path, "mel", shared_lesion=True)
    v3_loader["frame"] = ws["v3_frame"]
    with pytest.raises(SystemExit, match="share 1 lesion"):
        judge.run(ws["sep"], ws["real"], ws["meta"], ws["cand"], ws["sim"], ws["v3"], ws["out"], expected_n=ws["n"])


def test_the_synthetic_pool_and_validation_never_reach_training(tmp_path, stub_gate, v3_loader):
    """Moving synthetic mel, or the classifier_val images, leaves the trained judge unchanged."""
    a = _workspace(tmp_path / "a", "mel")
    b = _workspace(tmp_path / "b", "nv")
    for ws in (a, b):
        v3_loader["frame"] = ws["v3_frame"]
        ws["report"] = judge.run(ws["sep"], ws["real"], ws["meta"], ws["cand"], ws["sim"], ws["v3"], ws["out"], expected_n=ws["n"])
    assert a["report"]["settings"]["class_weights"] == b["report"]["settings"]["class_weights"]
    assert a["report"]["validation_recall_counts"] == b["report"]["validation_recall_counts"]
