#!/usr/bin/env python3
"""FID and CLIP-score trend for a generated probe sample set (docs/stage1_plan.md §9).

Two FID variants are computed:
  - Standard Inception-based FID (torchmetrics) — field-standard, comparable to other work, but a
    poor fit for grayscale medical images (Inception features are ImageNet-domain).
  - Domain FID using torchxrayvision's DenseNet121 features — the more literature-appropriate
    signal for this project (docs/stage1_plan.md: "FID doesn't reflect diagnostic content").

Both are tracked ONLY as relative trends across checkpoints, not absolute quality bars (plan §9).
The real reference set is gen_val (held out from training but touched by Stage 1) —
classifier_heldout must never be read here, since it is reserved untouched for Stage 4/5.

Usage:
    python scripts/eval/compute_fid_clipscore.py --generated-dir outputs/stage1_samples/<checkpoint_name>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import load_stage1_config  # noqa: E402
from scripts.utils.manifest import read_json, write_json  # noqa: E402


def load_images_as_uint8_tensor(paths: list[Path], size: int) -> torch.Tensor:
    from torchvision.transforms import functional as TF

    tensors = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB").resize((size, size))
            tensors.append(TF.pil_to_tensor(im))
    return torch.stack(tensors)


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    from scipy import linalg

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_domain_fid(real_paths: list[Path], generated_paths: list[Path], device: torch.device) -> float:
    import torchxrayvision as xrv
    import torchvision.transforms as T

    model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device).eval()
    transform = T.Compose([T.Resize((224, 224))])

    def extract_features(paths: list[Path]) -> np.ndarray:
        feats = []
        with torch.no_grad():
            for p in paths:
                with Image.open(p) as im:
                    arr = np.array(im.convert("L"), dtype=np.float32)
                arr = xrv.datasets.normalize(arr, 255)
                tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
                tensor = transform(tensor)
                features = model.features(tensor)
                feats.append(features.mean(dim=[2, 3]).squeeze(0).cpu().numpy())
        return np.stack(feats)

    real_feats = extract_features(real_paths)
    gen_feats = extract_features(generated_paths)

    mu1, sigma1 = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = gen_feats.mean(axis=0), np.cov(gen_feats, rowvar=False)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


def compute_inception_fid(real_paths: list[Path], generated_paths: list[Path], device: torch.device) -> float:
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(normalize=False).to(device)
    fid.update(load_images_as_uint8_tensor(real_paths, 299).to(device), real=True)
    fid.update(load_images_as_uint8_tensor(generated_paths, 299).to(device), real=False)
    return float(fid.compute().item())


def compute_clip_score(generated_paths: list[Path], prompts: list[str], device: torch.device) -> float:
    from torchmetrics.multimodal.clip_score import CLIPScore

    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to(device)
    images = load_images_as_uint8_tensor(generated_paths, 224).to(device)
    score = metric(images, prompts)
    return float(score.item())


def real_reference_dir(cfg, namespace: str) -> Path:
    """gen_val as written by 03_preprocess_images.py: <images_dir>/<namespace>/gen_val."""
    return Path(cfg.paths.images_dir) / namespace / "gen_val"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=str, required=True, help="Directory with probe_manifest.json (from generate_probe_samples.py)")
    parser.add_argument("--num-real-reference", type=int, default=200)
    parser.add_argument("--namespace", default=None, help="Split namespace of the real gen_val images (default: split.namespace)")
    args = parser.parse_args()

    cfg = load_stage1_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    namespace = args.namespace or str(cfg.split.namespace)

    generated_dir = Path(args.generated_dir)
    manifest = read_json(generated_dir / "probe_manifest.json")
    generated_paths = [Path(r["image_path"]) for r in manifest["records"]]
    prompts = [r["prompt"] for r in manifest["records"]]

    val_images_dir = real_reference_dir(cfg, namespace)
    real_paths = sorted(val_images_dir.glob("*.jpg"))[: args.num_real_reference]
    if not real_paths:
        print(f"No real reference images found under {val_images_dir} — run the preprocessing pipeline first.")
        return 1

    results = {"generated_dir": str(generated_dir), "split_namespace": namespace, "num_real_reference": len(real_paths)}

    if cfg.validation.clip_score:
        results["clip_score"] = compute_clip_score(generated_paths, prompts, device)
        print(f"CLIP score: {results['clip_score']:.4f}")

    if cfg.validation.fid.inception:
        try:
            results["fid_inception"] = compute_inception_fid(real_paths, generated_paths, device)
            print(f"FID (Inception): {results['fid_inception']:.2f}")
        except Exception as e:
            print(f"Inception FID failed ({e}); skipping.")

    if cfg.validation.fid.domain_torchxrayvision:
        try:
            results["fid_domain_torchxrayvision"] = compute_domain_fid(real_paths, generated_paths, device)
            print(f"FID (domain, torchxrayvision): {results['fid_domain_torchxrayvision']:.2f}")
        except Exception as e:
            print(f"Domain FID failed ({e}); skipping.")

    write_json(generated_dir / "fid_clipscore_results.json", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
