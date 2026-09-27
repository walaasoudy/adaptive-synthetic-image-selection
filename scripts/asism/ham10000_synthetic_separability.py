#!/usr/bin/env python3
"""Are the synthetic HAM10000 classes as distinct as the real ones? (docs/ham10000_synthetic_separability.md)

An explanatory diagnostic, fixed before it runs. Two steps:

    embed    (GPU pod) DINOv2 embeddings, exactly as pinned for the similarity signal, of all Stage 2
             candidates and all real gen_train images, plus the mean L*a*b* of each image's central
             crop. Writes embeddings.npz, colour.csv and provenance.json to
             outputs/ham10000/diagnostics/synthetic_separability/<namespace>/.
    analyse  (CPU) S1 separability, S2 within-class variety and S3 nv-leaning of synthetic mel/bkl,
             each as synthetic minus real with a 95% interval, then the document's reading rule.
             Writes separability.json and separability.md next to the embeddings.

Every verdict is "does a 95% interval of the difference exclude 0"; there is no threshold to tune.
Nothing is trained, and no signal, threshold, ranking, selection or v1 artifact is touched. The real
images come from gen_train only; their ids are checked against final_eval_heldout.

Usage (after approval only):
    python scripts/asism/ham10000_synthetic_separability.py embed   --namespace ham-stratified-v1
    python scripts/asism/ham10000_synthetic_separability.py analyse --input-dir <dir with embeddings.npz>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS, normalize_diagnosis  # noqa: E402
from scripts.utils.manifest import get_git_commit_hash, read_json, sha256_file, write_json  # noqa: E402

CLASSES: list[str] = list(CLASSIFIER_TARGET_LABELS)
SEED = 42
N_PER_CLASS = 38          # the smallest real class in gen_train (df)
N_DRAWS = 200
N_FOLDS = 5
N_BOOTSTRAP = 1000
S3_CLASSES = ("mel", "bkl")
OUTPUT_SUBDIR = Path("outputs/ham10000/diagnostics/synthetic_separability")


# ==============================================================================================
# Colour
# ==============================================================================================


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 255] (..., 3) to CIE L*a*b* under D65."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    matrix = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
    xyz = linear @ matrix.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def central_crop_lab(path: Path) -> tuple[float, float, float]:
    """Mean L*a*b* of the central 50% (by side) of the image."""
    from PIL import Image

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    crop = rgb[h // 4: h - h // 4, w // 4: w - w // 4]
    lab = srgb_to_lab(crop.reshape(-1, 3)).mean(axis=0)
    return float(lab[0]), float(lab[1]), float(lab[2])


# ==============================================================================================
# Step 1 - embed (GPU pod)
# ==============================================================================================


def embed(namespace: str, device: str | None = None, limit: int | None = None) -> dict:
    from scripts.asism.ham10000_01_compute_signals import (
        REFERENCE_SPLIT,
        embed_images,
        load_candidates,
        load_encoder,
        load_final_eval_image_ids,
    )
    from scripts.utils.config import load_named_config

    stage1 = load_named_config("ham10000_stage1.yaml", "ham_stage1")
    stage2 = load_named_config("ham10000_stage2.yaml", "ham_stage2")
    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    splits_cfg = load_named_config("splits_ham10000.yaml", "ham_splits")
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    candidates = load_candidates(Path(stage2.paths.stage2_root), namespace, limit)
    split_dir = Path(splits_cfg.paths.splits_root) / namespace
    real = pd.read_csv(split_dir / f"{REFERENCE_SPLIT}.csv")
    real["dx"] = [normalize_diagnosis(value) for value in real["dx"]]
    real["image_id"] = real["image_id"].astype(str)
    if limit:
        real = real.groupby("dx", group_keys=False).head(max(1, int(limit) // len(CLASSES)))
    leaked = sorted(set(real["image_id"]) & load_final_eval_image_ids(split_dir))
    if leaked:
        raise SystemExit(f"{len(leaked)} real image(s) are in final_eval_heldout (e.g. {leaked[:3]}).")
    images_root = Path(stage1.paths.images_dir) / namespace / REFERENCE_SPLIT
    real_paths = [images_root / f"{image_id}.jpg" for image_id in real["image_id"]]
    missing = [str(path) for path in real_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} real image(s) missing (e.g. {missing[:2]}).")

    similarity_cfg = stage3.signals.similarity
    encoder, transform, weights_path = load_encoder(similarity_cfg, device)
    batch_size = int(similarity_cfg.batch_size)
    synthetic_paths = [Path(path) for path in candidates["image_path"]]
    print(f"embedding {len(synthetic_paths)} synthetic and {len(real_paths)} real images on {device}", flush=True)
    embeddings = np.concatenate([
        embed_images(encoder, transform, synthetic_paths, batch_size, device),
        embed_images(encoder, transform, real_paths, batch_size, device),
    ]).astype(np.float32)

    image_ids = [*candidates["image_id"].astype(str), *real["image_id"]]
    diagnoses = [*candidates["dx"], *real["dx"]]
    sources = ["synthetic"] * len(candidates) + ["real"] * len(real)
    out_dir = Path(stage3.paths.project_root) / OUTPUT_SUBDIR / namespace
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "embeddings.npz", embeddings=embeddings, image_id=np.array(image_ids),
                        dx=np.array(diagnoses), source=np.array(sources))
    colour = pd.DataFrame(
        [(i, d, s, *central_crop_lab(p)) for i, d, s, p in
         zip(image_ids, diagnoses, sources, [*synthetic_paths, *real_paths])],
        columns=["image_id", "dx", "source", "L", "a", "b"],
    )
    colour.to_csv(out_dir / "colour.csv", index=False)
    provenance = {
        "namespace": namespace,
        "encoder": str(similarity_cfg.encoder),
        "weights_repo_id": str(similarity_cfg.weights_repo_id),
        "weights_revision": str(similarity_cfg.weights_revision),
        "weights_sha256": sha256_file(weights_path),
        "candidates_csv_sha256": sha256_file(Path(stage2.paths.stage2_root) / namespace / "all_candidates.csv"),
        "real_split": REFERENCE_SPLIT,
        "n_synthetic": int(len(candidates)),
        "n_real": int(len(real)),
        "limit": limit,
        "embeddings_sha256": sha256_file(out_dir / "embeddings.npz"),
        "git_commit_hash": get_git_commit_hash(),
    }
    write_json(out_dir / "provenance.json", provenance)
    return {"out_dir": str(out_dir), **provenance}


# ==============================================================================================
# Step 2 - analyse (CPU)
# ==============================================================================================


def l2_normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def interval(values) -> list[float]:
    values = np.asarray(values, dtype=float)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _centroids(x: np.ndarray, y: np.ndarray, classes: list[str]) -> np.ndarray:
    return l2_normalise(np.stack([x[y == name].mean(axis=0) for name in classes]))


def nearest_centroid_cv(x: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                        classes: list[str] = CLASSES, folds: int = N_FOLDS) -> tuple[float, np.ndarray]:
    """Balanced accuracy and out-of-fold confusion (rows truth, columns prediction) of a cosine
    nearest-centroid classifier under stratified k-fold cross-validation."""
    x = l2_normalise(x)
    fold_of = np.empty(len(y), dtype=int)
    for name in classes:
        members = np.flatnonzero(y == name)
        fold_of[rng.permutation(members)] = np.arange(len(members)) % folds
    predicted = np.empty(len(y), dtype=object)
    for fold in range(folds):
        test = fold_of == fold
        centroids = _centroids(x[~test], y[~test], classes)
        predicted[test] = np.array(classes, dtype=object)[np.argmax(x[test] @ centroids.T, axis=1)]
    confusion = np.array([[int(np.sum((y == t) & (predicted == p))) for p in classes] for t in classes])
    recalls = np.diag(confusion) / confusion.sum(axis=1)
    return float(recalls.mean()), confusion


def _draw(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    index = np.concatenate([rng.choice(np.flatnonzero(y == name), size=n, replace=False) for name in CLASSES])
    return x[index], y[index]


def s1_separability(xr, yr, xs, ys, rng, n_per_class=N_PER_CLASS, draws=N_DRAWS) -> dict:
    real_ba, synthetic_ba, confusion = [], [], np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for _ in range(draws):
        ba, _ = nearest_centroid_cv(*_draw(xr, yr, rng, n_per_class), rng)
        real_ba.append(ba)
        ba, c = nearest_centroid_cv(*_draw(xs, ys, rng, n_per_class), rng)
        synthetic_ba.append(ba)
        confusion += c
    difference = np.array(synthetic_ba) - np.array(real_ba)
    return {
        "real_balanced_accuracy": {"mean": float(np.mean(real_ba)), "interval": interval(real_ba)},
        "synthetic_balanced_accuracy": {"mean": float(np.mean(synthetic_ba)), "interval": interval(synthetic_ba)},
        "difference_synthetic_minus_real": {"mean": float(difference.mean()), "interval": interval(difference)},
        "synthetic_confusion_pooled": {"classes": CLASSES, "matrix": confusion.tolist()},
        "confirms": bool(interval(difference)[1] < 0),
    }


def mean_pairwise_cosine_distance(x: np.ndarray) -> float:
    x = l2_normalise(x)
    n = len(x)
    similarity = x @ x.T
    return float(1.0 - (similarity.sum() - np.trace(similarity)) / (n * (n - 1)))


def s2_variety(xr, yr, xs, ys, rng, n_per_class=N_PER_CLASS, draws=N_DRAWS) -> dict:
    per_class = {name: [] for name in CLASSES}
    real_values = {name: [] for name in CLASSES}
    synthetic_values = {name: [] for name in CLASSES}
    for _ in range(draws):
        for name in CLASSES:
            r = mean_pairwise_cosine_distance(xr[rng.choice(np.flatnonzero(yr == name), n_per_class, replace=False)])
            s = mean_pairwise_cosine_distance(xs[rng.choice(np.flatnonzero(ys == name), n_per_class, replace=False)])
            real_values[name].append(r)
            synthetic_values[name].append(s)
            per_class[name].append(s - r)
    pooled = np.mean([per_class[name] for name in CLASSES], axis=0)
    return {
        "per_class": {name: {"real": float(np.mean(real_values[name])),
                             "synthetic": float(np.mean(synthetic_values[name])),
                             "difference_mean": float(np.mean(per_class[name])),
                             "difference_interval": interval(per_class[name])} for name in CLASSES},
        "pooled_difference": {"mean": float(pooled.mean()), "interval": interval(pooled)},
        "confirms": bool(interval(pooled)[1] < 0),
    }


def s3_nv_leaning(xr, yr, xs, ys, rng, bootstrap=N_BOOTSTRAP) -> dict:
    xr, xs = l2_normalise(xr), l2_normalise(xs)
    centroid_rows, query_rows = [], []
    for name in CLASSES:
        members = rng.permutation(np.flatnonzero(yr == name))
        half = len(members) // 2
        centroid_rows.append(members[:half])
        query_rows.append(members[half:])
    centroid_rows, query_rows = np.concatenate(centroid_rows), np.concatenate(query_rows)
    centroids = dict(zip(CLASSES, _centroids(xr[centroid_rows], yr[centroid_rows], CLASSES)))

    def leaning(x: np.ndarray, name: str) -> np.ndarray:
        return (x @ centroids["nv"]) > (x @ centroids[name])

    per_class = {}
    for name in [c for c in CLASSES if c != "nv"]:
        synthetic = leaning(xs[ys == name], name)
        real = leaning(xr[query_rows][yr[query_rows] == name], name)
        differences = [rng.choice(synthetic, len(synthetic)).mean() - rng.choice(real, len(real)).mean()
                       for _ in range(bootstrap)]
        per_class[name] = {
            "synthetic_share": float(synthetic.mean()), "n_synthetic": int(len(synthetic)),
            "real_share": float(real.mean()), "n_real_queries": int(len(real)),
            "difference": float(synthetic.mean() - real.mean()),
            "difference_interval": interval(differences),
            "decides": name in S3_CLASSES,
        }
    return {"per_class": per_class,
            "confirms": all(per_class[name]["difference_interval"][0] > 0 for name in S3_CLASSES)}


def colour_summary(colour: pd.DataFrame) -> dict:
    return {source: {name: {channel: float(group.loc[group["dx"] == name, channel].median())
                            for channel in ("L", "a", "b")} for name in CLASSES}
            for source, group in colour.groupby("source")}


def reading(s1: dict, s2: dict, s3: dict) -> str:
    confirmed = [s1["confirms"], s2["confirms"], s3["confirms"]]
    if all(confirmed):
        return "synthetic_distribution"
    if not any(confirmed):
        return "visual_sample_misleading"
    return "mixed"


def load_embeddings(input_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    provenance = read_json(input_dir / "provenance.json")
    if provenance.get("embeddings_sha256") != sha256_file(input_dir / "embeddings.npz"):
        raise SystemExit("embeddings.npz does not match its provenance hash.")
    data = np.load(input_dir / "embeddings.npz", allow_pickle=False)
    x, y, source = data["embeddings"], data["dx"].astype(str), data["source"].astype(str)
    for side in ("real", "synthetic"):
        counts = pd.Series(y[source == side]).value_counts()
        short = {name: int(counts.get(name, 0)) for name in CLASSES if counts.get(name, 0) < N_PER_CLASS}
        if short:
            raise SystemExit(f"{side}: classes below {N_PER_CLASS} images: {short}")
    return x, y, source


def analyse(input_dir: Path, draws: int = N_DRAWS, bootstrap: int = N_BOOTSTRAP) -> dict:
    x, y, source = load_embeddings(input_dir)
    xr, yr = x[source == "real"], y[source == "real"]
    xs, ys = x[source == "synthetic"], y[source == "synthetic"]
    rng = np.random.default_rng(SEED)
    s1 = s1_separability(xr, yr, xs, ys, rng, draws=draws)
    s2 = s2_variety(xr, yr, xs, ys, rng, draws=draws)
    s3 = s3_nv_leaning(xr, yr, xs, ys, rng, bootstrap=bootstrap)
    colour_path = input_dir / "colour.csv"
    return {
        "document": "docs/ham10000_synthetic_separability.md",
        "seed": SEED, "n_per_class": N_PER_CLASS, "draws": draws, "bootstrap": bootstrap,
        "n_real": int(len(yr)), "n_synthetic": int(len(ys)),
        "provenance": read_json(input_dir / "provenance.json"),
        "s1": s1, "s2": s2, "s3": s3,
        "colour_median_lab": colour_summary(pd.read_csv(colour_path)) if colour_path.is_file() else None,
        "reading": reading(s1, s2, s3),
    }


def _iv(values: list[float], digits: int = 3) -> str:
    return f"[{values[0]:.{digits}f}, {values[1]:.{digits}f}]"


def render_markdown(report: dict) -> str:
    s1, s2, s3 = report["s1"], report["s2"], report["s3"]
    lines = ["# HAM10000 synthetic class separability", "",
             f"Document: `{report['document']}`. Real {report['n_real']}, synthetic {report['n_synthetic']}; "
             f"{report['n_per_class']} per class per draw, {report['draws']} draws, seed {report['seed']}.", "",
             f"**Reading: {report['reading']}** (S1 {s1['confirms']}, S2 {s2['confirms']}, S3 {s3['confirms']})", "",
             "## S1: separability (nearest-centroid, 5-fold CV, balanced accuracy)", "",
             f"- real {s1['real_balanced_accuracy']['mean']:.3f} {_iv(s1['real_balanced_accuracy']['interval'])}",
             f"- synthetic {s1['synthetic_balanced_accuracy']['mean']:.3f} {_iv(s1['synthetic_balanced_accuracy']['interval'])}",
             f"- synthetic - real {s1['difference_synthetic_minus_real']['mean']:+.3f} "
             f"{_iv(s1['difference_synthetic_minus_real']['interval'])}; **confirms = {s1['confirms']}**", "",
             "Synthetic out-of-fold confusion, pooled over draws (rows truth):", "",
             "| | " + " | ".join(CLASSES) + " |", "|---" * (len(CLASSES) + 1) + "|"]
    for name, row in zip(CLASSES, s1["synthetic_confusion_pooled"]["matrix"]):
        lines.append(f"| {name} | " + " | ".join(str(v) for v in row) + " |")
    lines += ["", "## S2: within-class variety (mean pairwise cosine distance)", "",
              "| Class | real | synthetic | synthetic - real | interval |", "|---|---|---|---|---|"]
    for name, entry in s2["per_class"].items():
        lines.append(f"| {name} | {entry['real']:.4f} | {entry['synthetic']:.4f} | "
                     f"{entry['difference_mean']:+.4f} | {_iv(entry['difference_interval'], 4)} |")
    lines += [f"- pooled {s2['pooled_difference']['mean']:+.4f} {_iv(s2['pooled_difference']['interval'], 4)}; "
              f"**confirms = {s2['confirms']}**", "",
              "## S3: nv-leaning share (closer to the real nv centroid than to the own-class one)", "",
              "| Class | synthetic | real queries | difference | interval | decides |", "|---|---|---|---|---|---|"]
    for name, entry in s3["per_class"].items():
        lines.append(f"| {name} | {entry['synthetic_share']:.3f} (n={entry['n_synthetic']}) | "
                     f"{entry['real_share']:.3f} (n={entry['n_real_queries']}) | {entry['difference']:+.3f} | "
                     f"{_iv(entry['difference_interval'])} | {entry['decides']} |")
    lines.append(f"- **confirms = {s3['confirms']}**")
    if report.get("colour_median_lab"):
        lines += ["", "## Colour: median L*a*b* of the central crop (descriptive)", "",
                  "| Class | real L | real a | real b | synth L | synth a | synth b |", "|---|---|---|---|---|---|---|"]
        real, synthetic = report["colour_median_lab"].get("real", {}), report["colour_median_lab"].get("synthetic", {})
        for name in CLASSES:
            r, s = real.get(name, {}), synthetic.get(name, {})
            lines.append(f"| {name} | " + " | ".join(f"{d.get(k, float('nan')):.1f}" for d in (r, s) for k in "Lab") + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--namespace", required=True)
    e.add_argument("--device", default=None)
    e.add_argument("--limit", type=int, default=None, help="smoke runs only")
    a = sub.add_parser("analyse")
    a.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.step == "embed":
        import json

        print(json.dumps(embed(args.namespace, args.device, args.limit), indent=2))
        return 0
    report = analyse(args.input_dir)
    write_json(args.input_dir / "separability.json", report)
    text = render_markdown(report)
    (args.input_dir / "separability.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
