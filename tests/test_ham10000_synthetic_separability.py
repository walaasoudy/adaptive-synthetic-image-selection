#!/usr/bin/env python3
"""The synthetic class-separability check: S1-S3 confirm on collapsed synthetic classes and do not
confirm on synthetic classes drawn like the real ones; the reading rule; the colour conversion; and
the embed step's wiring (encoder mocked), its output location and its final_eval refusal.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fixture_workspace import fixture_workspace  # noqa: E402
from scripts.asism import ham10000_synthetic_separability as sep  # noqa: E402
from scripts.utils.manifest import sha256_file, write_json  # noqa: E402
from test_ham10000_cam_reference_build import NAMESPACE  # noqa: E402

DIM = 32
N_REAL = 60
N_SYN = 60


def _clusters(rng, n, spread, centres) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([centres[i] + spread * rng.standard_normal((n, DIM)) for i in range(len(sep.CLASSES))])
    y = np.repeat(np.array(sep.CLASSES), n)
    return x, y


def _centres(rng) -> np.ndarray:
    return 3.0 * rng.standard_normal((len(sep.CLASSES), DIM))


def _collapsed(rng, centres, n) -> tuple[np.ndarray, np.ndarray]:
    """Every synthetic class sits next to the real nv centre, with little spread."""
    nv = centres[sep.CLASSES.index("nv")]
    offsets = 0.3 * rng.standard_normal((len(sep.CLASSES), DIM))
    x = np.concatenate([nv + offsets[i] + 0.4 * rng.standard_normal((n, DIM)) for i in range(len(sep.CLASSES))])
    return x, np.repeat(np.array(sep.CLASSES), n)


def _write_embeddings(root: Path, xr, yr, xs, ys) -> Path:
    x = np.concatenate([xs, xr]).astype(np.float32)
    y = np.concatenate([ys, yr])
    source = np.array(["synthetic"] * len(ys) + ["real"] * len(yr))
    ids = np.array([f"img_{i}" for i in range(len(y))])
    np.savez_compressed(root / "embeddings.npz", embeddings=x, image_id=ids, dx=y, source=source)
    write_json(root / "provenance.json", {"embeddings_sha256": sha256_file(root / "embeddings.npz")})
    pd.DataFrame({"image_id": ids, "dx": y, "source": source, "L": 50.0, "a": 10.0, "b": 5.0}).to_csv(
        root / "colour.csv", index=False)
    return root


@pytest.fixture
def root():
    with fixture_workspace("separability") as path:
        yield path


def test_the_pre_registered_constants():
    assert (sep.SEED, sep.N_PER_CLASS, sep.N_DRAWS, sep.N_FOLDS, sep.N_BOOTSTRAP) == (42, 38, 200, 5, 1000)
    assert sep.S3_CLASSES == ("mel", "bkl")
    text = (REPO / "docs/ham10000_synthetic_separability.md").read_text(encoding="utf-8")
    for phrase in ["**38 images per class**", "**200 draws**", "1,000 resamples", "percentile is **< 0**",
                   "**> 0 for both mel and bkl**", "Seed** 42"]:
        assert phrase in text, phrase


# ==============================================================================================
# Building blocks
# ==============================================================================================


def test_srgb_to_lab_reference_points():
    assert sep.srgb_to_lab(np.array([255, 255, 255])) == pytest.approx([100.0, 0.0, 0.0], abs=0.05)
    assert sep.srgb_to_lab(np.array([0, 0, 0])) == pytest.approx([0.0, 0.0, 0.0], abs=0.05)
    # sRGB red, a textbook value: L* 53.24, a* 80.09, b* 67.20
    assert sep.srgb_to_lab(np.array([255, 0, 0])) == pytest.approx([53.24, 80.09, 67.20], abs=0.1)


def test_central_crop_lab_reads_only_the_centre(root):
    from PIL import Image

    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[10:30, 10:30] = 255
    Image.fromarray(image).save(root / "centre.png")
    assert sep.central_crop_lab(root / "centre.png") == pytest.approx((100.0, 0.0, 0.0), abs=0.05)


def test_nearest_centroid_cv_on_separated_and_identical_classes():
    rng = np.random.default_rng(0)
    x, y = _clusters(rng, 20, 0.1, _centres(rng))
    ba, confusion = sep.nearest_centroid_cv(x, y, rng)
    assert ba == pytest.approx(1.0)
    assert confusion.sum() == len(y) and np.trace(confusion) == len(y)
    noise = rng.standard_normal((len(y), DIM))
    assert sep.nearest_centroid_cv(noise, y, rng)[0] < 0.5


def test_mean_pairwise_cosine_distance():
    assert sep.mean_pairwise_cosine_distance(np.array([[1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])) == pytest.approx(0.0)
    assert sep.mean_pairwise_cosine_distance(np.array([[1.0, 0.0], [0.0, 1.0]])) == pytest.approx(1.0)


def test_the_reading_rule():
    yes, no = {"confirms": True}, {"confirms": False}
    assert sep.reading(yes, yes, yes) == "synthetic_distribution"
    assert sep.reading(no, no, no) == "visual_sample_misleading"
    assert sep.reading(yes, no, yes) == "mixed"


# ==============================================================================================
# The questions end to end
# ==============================================================================================


def test_collapsed_synthetic_classes_confirm_all_three(root):
    rng = np.random.default_rng(1)
    centres = _centres(rng)
    xr, yr = _clusters(rng, N_REAL, 1.0, centres)
    xs, ys = _collapsed(rng, centres, N_SYN)
    report = sep.analyse(_write_embeddings(root, xr, yr, xs, ys), draws=20, bootstrap=200)
    assert report["s1"]["confirms"] and report["s2"]["confirms"] and report["s3"]["confirms"]
    assert report["s3"]["per_class"]["mel"]["synthetic_share"] > 0.9
    assert report["reading"] == "synthetic_distribution"
    text = sep.render_markdown(report)
    assert "**Reading: synthetic_distribution**" in text and "## Colour" in text


def test_synthetic_classes_drawn_like_the_real_ones_confirm_none(root):
    rng = np.random.default_rng(2)
    centres = _centres(rng)
    xr, yr = _clusters(rng, N_REAL, 1.0, centres)
    xs, ys = _clusters(rng, N_SYN, 1.0, centres)
    report = sep.analyse(_write_embeddings(root, xr, yr, xs, ys), draws=20, bootstrap=200)
    assert not report["s1"]["confirms"] and not report["s2"]["confirms"] and not report["s3"]["confirms"]
    assert report["reading"] == "visual_sample_misleading"


def test_s3_needs_both_mel_and_bkl(root):
    rng = np.random.default_rng(3)
    centres = _centres(rng)
    xr, yr = _clusters(rng, N_REAL, 1.0, centres)
    xs, ys = _clusters(rng, N_SYN, 1.0, centres)
    nv = centres[sep.CLASSES.index("nv")]
    xs[ys == "mel"] = nv + 0.5 * rng.standard_normal((N_SYN, DIM))   # only mel leans to nv
    s3 = sep.analyse(_write_embeddings(root, xr, yr, xs, ys), draws=5, bootstrap=200)["s3"]
    assert s3["per_class"]["mel"]["difference_interval"][0] > 0
    assert s3["per_class"]["bkl"]["difference_interval"][0] <= 0
    assert not s3["confirms"]


def test_s3_real_queries_never_build_a_centroid():
    rng = np.random.default_rng(4)
    centres = _centres(rng)
    xr, yr = _clusters(rng, 41, 1.0, centres)
    xs, ys = _clusters(rng, 40, 1.0, centres)
    s3 = sep.s3_nv_leaning(xr, yr, xs, ys, rng, bootstrap=10)
    assert s3["per_class"]["mel"]["n_real_queries"] == 41 - 41 // 2
    assert s3["per_class"]["mel"]["n_synthetic"] == 40


# ==============================================================================================
# Refusals
# ==============================================================================================


def test_a_class_below_the_draw_size_is_refused(root):
    rng = np.random.default_rng(5)
    centres = _centres(rng)
    xr, yr = _clusters(rng, 30, 1.0, centres)
    xs, ys = _clusters(rng, N_SYN, 1.0, centres)
    with pytest.raises(SystemExit, match="classes below 38"):
        sep.analyse(_write_embeddings(root, xr, yr, xs, ys), draws=2, bootstrap=2)


def test_embeddings_that_do_not_match_their_provenance_are_refused(root):
    rng = np.random.default_rng(6)
    centres = _centres(rng)
    _write_embeddings(root, *_clusters(rng, N_REAL, 1.0, centres), *_clusters(rng, N_SYN, 1.0, centres))
    write_json(root / "provenance.json", {"embeddings_sha256": "0" * 64})
    with pytest.raises(SystemExit, match="does not match its provenance hash"):
        sep.analyse(root, draws=2, bootstrap=2)


# ==============================================================================================
# The embed step (encoder mocked)
# ==============================================================================================


@pytest.fixture
def pipeline(monkeypatch):
    from test_ham10000_cam_reference_build import _overlay, _workspace  # noqa: F401
    from test_ham10000_compute_signals import _write_candidates

    import scripts.asism.ham10000_01_compute_signals as signals

    with fixture_workspace("separability-embed") as path:
        _workspace(path)
        _write_candidates(path)
        monkeypatch.setenv("PROJECT_ROOT", str(path))
        monkeypatch.setenv("THESIS_CONFIG_OVERLAY", str(_overlay(path)))
        weights = path / "weights.safetensors"
        weights.write_bytes(b"fake")
        monkeypatch.setattr(signals, "load_encoder", lambda cfg, device: (None, None, weights))
        monkeypatch.setattr(signals, "embed_images",
                            lambda encoder, transform, paths, batch_size, device: np.ones((len(paths), 4)))
        yield path


def test_embed_writes_under_the_diagnostics_root_only(pipeline):
    result = sep.embed(NAMESPACE, device="cpu")
    out_dir = pipeline / "outputs/ham10000/diagnostics/synthetic_separability" / NAMESPACE
    assert Path(result["out_dir"]) == out_dir
    data = np.load(out_dir / "embeddings.npz")
    assert set(data["source"]) == {"real", "synthetic"}
    assert len(data["embeddings"]) == result["n_real"] + result["n_synthetic"]
    colour = pd.read_csv(out_dir / "colour.csv")
    assert len(colour) == len(data["embeddings"])
    assert result["embeddings_sha256"] == sha256_file(out_dir / "embeddings.npz")
    assert result["weights_revision"] == "936966a8732c5442c9def5d126f2cc4ad4243dba"
    assert not (pipeline / "outputs/ham10000/stage3").exists()


def test_embed_refuses_real_images_in_final_eval(pipeline):
    splits = pipeline / "data/ham10000/processed/splits" / NAMESPACE
    leaked = pd.read_csv(splits / "gen_train.csv")["image_id"].iloc[0]
    final_eval = pd.read_csv(splits / "final_eval_heldout.csv")
    pd.concat([final_eval, pd.DataFrame([{"image_id": leaked, "lesion_id": "HAM_x", "dx": "nv"}])]).to_csv(
        splits / "final_eval_heldout.csv", index=False)
    with pytest.raises(SystemExit, match="final_eval_heldout"):
        sep.embed(NAMESPACE, device="cpu")
