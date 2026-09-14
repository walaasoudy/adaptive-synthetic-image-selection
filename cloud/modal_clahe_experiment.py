"""CLAHE ablation on Modal: does contrast enhancement help a real-image CXR classifier?

Decides whether CLAHE should be added to preprocessing BEFORE Stage 1 builds its latent cache.
Nothing under scripts/ or configs/ changes and no thesis artifact is written: CLAHE is applied
on the fly by swapping scripts.utils.classifier.CXRRecordDataset for a subclass, and every
checkpoint/prediction goes under experiments/<tag>/ on the volume, never the auxiliary path.

Design (fixed before any result is seen):
  arms      baseline (current 768 px preprocessed JPEGs) vs clahe (clip 2.0, 8x8 tiles, applied
            at 768 px before the resize to the classifier resolution, i.e. where preprocessing
            would apply it)
  data      train gen_train, evaluate gen_val -- the auxiliary classifier's own pool; the
            classifier_* and heldout splits are never touched
  model     the auxiliary classifier's settings from configs/stage3_asism.yaml, 3000 steps,
            no intermediate evaluation (so no checkpoint selection on the evaluation split)
  seeds     42, 43, 44 per arm
  adopt     CLAHE only if ALL hold: mean macro-AUROC gain >= +0.005; clahe > baseline for every
            seed; paired patient-level bootstrap CI on the seed-averaged predictions excludes 0

Run from the repository root, after prepare_data has finished:
    modal run cloud/modal_clahe_experiment.py::smoke                 # 20 steps, 256 images, <$0.10
    modal run --detach cloud/modal_clahe_experiment.py::experiment   # 6 runs + analysis, ~$2-2.5
    modal run cloud/modal_clahe_experiment.py::analyze               # re-print the report

Unattended (laptop closed): wait for prepare_data, smoke, then the full experiment if it passed:
    modal run --detach cloud/modal_clahe_experiment.py::after_prepare_data
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

REPO_LOCAL = Path(__file__).resolve().parents[1]
REPO = "/root/repo"
VOL_MOUNT = "/vol"
PROJECT_ROOT = Path(VOL_MOUNT) / "project"
OVERLAY = PROJECT_ROOT / "thesis_overlay.yaml"
NAMESPACE = "production-thesis-v1"
EXPERIMENTS = PROJECT_ROOT / "experiments"

ARMS = ("baseline", "clahe")
SEEDS = (42, 43, 44)
CLAHE_CLIP, CLAHE_TILES = 2.0, 8
MIN_MEAN_GAIN = 0.005

_CODE_IGNORE = ["**/__pycache__", "**/*.pyc"]

# Same image as cloud/modal_stage1.py (frozen torch matrix + requirements), so the classifier code
# runs under the identical environment; copied rather than imported to avoid a second Modal app.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "torchvision==0.23.0",
                 index_url="https://download.pytorch.org/whl/cu128")
    .pip_install_from_requirements(str(REPO_LOCAL / "environment" / "requirements.txt"))
    .add_local_dir(REPO_LOCAL / "scripts", f"{REPO}/scripts", ignore=_CODE_IGNORE)
    .add_local_dir(REPO_LOCAL / "configs", f"{REPO}/configs", ignore=_CODE_IGNORE)
    .add_local_dir(REPO_LOCAL / "environment", f"{REPO}/environment", ignore=_CODE_IGNORE)
)

volume = modal.Volume.from_name("chest-synth-thesis", create_if_missing=False, version=2)
app = modal.App("chest-synth-clahe-experiment")


def _enter_repo() -> None:
    """Point the repo's config loaders at the volume, then make `scripts` importable."""
    if not OVERLAY.is_file():
        raise SystemExit(f"{OVERLAY} is missing. Run cloud/modal_stage1.py::prepare_data first.")
    os.environ.update({
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "THESIS_CONFIG_OVERLAY": str(OVERLAY),
        "TORCH_HOME": str(PROJECT_ROOT / ".cache" / "torch"),  # ImageNet weights, downloaded once
        "PYTHONUNBUFFERED": "1",
        "TQDM_MININTERVAL": "60",
    })
    os.chdir(REPO)
    if REPO not in sys.path:
        sys.path.insert(0, REPO)


def _install_clahe_dataset() -> None:
    """Swap the classifier dataset for one that applies CLAHE before the resize.

    train_classifier() and predict_probabilities() look CXRRecordDataset up as a module global at
    call time, so both training and evaluation read CLAHE images. DataLoader workers are forked and
    inherit the swap."""
    import cv2
    import numpy as np
    import scripts.utils.classifier as classifier

    base = classifier.CXRRecordDataset
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILES, CLAHE_TILES))

    class ClaheCXRRecordDataset(base):
        def __getitem__(self, index: int):
            import torch
            from PIL import Image
            from torchvision.transforms import functional as TF

            record = self.records[index]
            with Image.open(record["image_path"]) as source:
                gray = np.asarray(source.convert("L"))
            image = Image.fromarray(clahe.apply(gray)).convert("RGB")
            if image.size != (self.resolution, self.resolution):
                image = image.resize((self.resolution, self.resolution), Image.BICUBIC)
            tensor = TF.normalize(TF.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            return {
                "image": tensor,
                "target": torch.from_numpy(self.targets[index]),
                "mask": torch.from_numpy(self.masks[index]),
                "index": index,
            }

    classifier.CXRRecordDataset = ClaheCXRRecordDataset


def _records(split: str, max_samples: int) -> list[dict]:
    from scripts.utils.classifier import records_from_split
    from scripts.utils.config import load_stage1_config
    from scripts.utils.splits import load_split

    images_dir = Path(load_stage1_config().paths.images_dir) / NAMESPACE / split
    frame = load_split(split, NAMESPACE, purpose="schema_validation", caller="clahe_experiment")
    records = sorted(records_from_split(frame, images_dir), key=lambda record: record["image_id"])
    if not records:
        raise SystemExit(f"No preprocessed {split} images under {images_dir}; is prepare_data finished?")
    return records[:max_samples] if max_samples else records


@app.function(image=image, gpu="A10", volumes={VOL_MOUNT: volume},
              cpu=8.0, memory=16384, timeout=4 * 3600)
def run_arm(arm: str, seed: int, max_steps: int = 3000, max_samples: int = 0, tag: str = "clahe_v1") -> dict:
    """Train one arm/seed and save its gen_val predictions. Resumes from its own checkpoint."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    _enter_repo()
    if arm == "clahe":
        _install_clahe_dataset()

    import numpy as np
    import pandas as pd
    from scripts.utils.classifier import TrainingBudget, predict_probabilities, train_classifier
    from scripts.utils.config import load_named_config
    from scripts.utils.labels import CLASSIFIER_TARGET_LABELS, build_label_arrays
    from scripts.utils.splits import split_provenance

    aux = load_named_config("stage3_asism.yaml", "stage3").auxiliary_classifier
    train_records = _records("gen_train", max_samples)
    val_records = _records("gen_val", max_samples)

    run_dir = EXPERIMENTS / tag / f"{arm}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "classifier.pt"
    metadata = {"experiment": tag, "arm": arm, "seed": seed, "max_steps": max_steps,
                "max_samples": max_samples, "clahe_clip": CLAHE_CLIP, "clahe_tiles": CLAHE_TILES,
                "split_manifest_hash": split_provenance(NAMESPACE)["split_manifest_hash"]}
    print(f"[{arm} seed={seed}] train={len(train_records):,} val={len(val_records):,} steps={max_steps}", flush=True)

    model, accounting, history = train_classifier(
        train_records=train_records,
        val_records=val_records,
        budget=TrainingBudget(max_steps=max_steps, batch_size=32, seed=seed, eval_every_n_steps=0),
        dropout_p=float(aux.dropout_p),
        pretrained_source=str(aux.pretrained_source),
        resolution=int(aux.resolution),
        progress_desc=f"{arm}-{seed}",
        checkpoint_path=checkpoint,
        resume_from=checkpoint if checkpoint.is_file() else None,
        checkpoint_metadata=metadata,
    )

    probabilities, _ = predict_probabilities(model, val_records, int(aux.resolution))
    frame = pd.DataFrame([record["labels"] for record in val_records])
    _, targets, masks = build_label_arrays(frame[CLASSIFIER_TARGET_LABELS], CLASSIFIER_TARGET_LABELS)
    np.savez_compressed(
        run_dir / "gen_val_predictions.npz",
        probabilities=probabilities, targets=targets, masks=masks,
        patient_ids=np.array([str(record["patient_id"]) for record in val_records]),
        image_ids=np.array([record["image_id"] for record in val_records]),
    )
    final = history[-1]["macro_auroc"] if history else float("nan")
    (run_dir / "run.json").write_text(json.dumps(
        {**metadata, "final_macro_auroc": final, "optimizer_steps": accounting.optimizer_steps,
         "epochs_completed": accounting.epochs_completed}, indent=2), encoding="utf-8")
    volume.commit()
    print(f"[{arm} seed={seed}] gen_val macro-AUROC {final:.4f}", flush=True)
    return {"arm": arm, "seed": seed, "macro_auroc": final}


@app.function(image=image, volumes={VOL_MOUNT: volume}, cpu=2.0, memory=8192, timeout=3600)
def analyze(tag: str = "clahe_v1", n_resamples: int = 1000) -> dict:
    """Apply the pre-registered decision rule to the saved predictions."""
    seeds = SEEDS
    _enter_repo()
    volume.reload()
    import numpy as np
    from scripts.utils.metrics import macro_auroc_masked, paired_bootstrap_difference

    root = EXPERIMENTS / tag
    loaded = {(arm, seed): np.load(root / f"{arm}_seed{seed}" / "gen_val_predictions.npz")
              for arm in ARMS for seed in seeds}
    reference = loaded[(ARMS[0], seeds[0])]
    for key, data in loaded.items():
        if not np.array_equal(data["image_ids"], reference["image_ids"]):
            raise SystemExit(f"{key} was evaluated on different images; the comparison is not paired")
    targets, masks, patients = reference["targets"], reference["masks"], reference["patient_ids"]

    def macro(probabilities, rows=slice(None)):
        return macro_auroc_masked(probabilities[rows], targets[rows], masks[rows])["macro_auroc"]

    per_run = {f"{arm}_seed{seed}": macro(data["probabilities"]) for (arm, seed), data in loaded.items()}
    per_seed_gain = {seed: per_run[f"clahe_seed{seed}"] - per_run[f"baseline_seed{seed}"] for seed in seeds}
    mean_gain = float(np.mean(list(per_seed_gain.values())))

    ensemble = {arm: np.mean([loaded[(arm, seed)]["probabilities"] for seed in seeds], axis=0) for arm in ARMS}
    bootstrap = paired_bootstrap_difference(
        patients,
        lambda rows: macro(ensemble["clahe"], rows),
        lambda rows: macro(ensemble["baseline"], rows),
        n_resamples=n_resamples,
    )

    criteria = {
        "mean_gain_at_least_0.005": mean_gain >= MIN_MEAN_GAIN,
        "clahe_wins_every_seed": all(gain > 0 for gain in per_seed_gain.values()),
        "bootstrap_ci_excludes_zero": bool(np.isfinite(bootstrap["ci_lower"]) and bootstrap["ci_lower"] > 0),
    }
    report = {
        "experiment": tag, "namespace": NAMESPACE, "seeds": list(seeds),
        "clahe": {"clip": CLAHE_CLIP, "tiles": CLAHE_TILES},
        "per_run_macro_auroc": per_run,
        "per_seed_gain": {str(seed): gain for seed, gain in per_seed_gain.items()},
        "mean_gain": mean_gain,
        "ensemble_macro_auroc": {arm: macro(probabilities) for arm, probabilities in ensemble.items()},
        "paired_bootstrap_clahe_minus_baseline": bootstrap,
        "criteria": criteria,
        "decision": "ADOPT CLAHE" if all(criteria.values()) else "KEEP ORIGINAL IMAGES",
    }
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    volume.commit()

    print("=" * 78)
    for name, value in per_run.items():
        print(f"  {name:18s} macro-AUROC {value:.4f}")
    for seed, gain in per_seed_gain.items():
        print(f"  seed {seed}: clahe - baseline = {gain:+.4f}")
    print(f"  mean gain {mean_gain:+.4f}   bootstrap CI [{bootstrap['ci_lower']:+.4f}, {bootstrap['ci_upper']:+.4f}]"
          f"   p={bootstrap['p_value']:.3f}")
    for name, passed in criteria.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"DECISION: {report['decision']}   ({root / 'report.json'})")
    print("=" * 78)
    return report


@app.function(image=image, volumes={VOL_MOUNT: volume}, cpu=1.0, memory=2048, timeout=6 * 3600)
def experiment(max_steps: int = 3000, tag: str = "clahe_v1") -> dict:
    """All six runs in parallel, then the analysis -- orchestrated server-side, so it survives
    the laptop sleeping when launched with --detach."""
    results = list(run_arm.starmap([(arm, seed, max_steps, 0, tag) for arm in ARMS for seed in SEEDS]))
    for result in results:
        print(result, flush=True)
    return analyze.remote(tag)


@app.function(image=image, volumes={VOL_MOUNT: volume}, cpu=0.25, memory=1024, timeout=10 * 3600)
def after_prepare_data(max_wait_minutes: int = 180, max_steps: int = 3000, tag: str = "clahe_v1") -> dict:
    """Unattended chain for when nobody is at the laptop: wait for prepare_data's final volume
    commit, run the smoke, and launch the full experiment only if the smoke passed."""
    import time

    captions = PROJECT_ROOT / "data" / "chexpert" / "processed" / "captions" / NAMESPACE / "gen_val_captions.jsonl"
    auxiliary = PROJECT_ROOT / "checkpoints" / "auxiliary_classifier"
    deadline = time.time() + max_wait_minutes * 60
    while True:
        volume.reload()   # only committed state is visible; the captions file is committed last
        if captions.is_file():
            break
        if time.time() > deadline:
            raise SystemExit(f"prepare_data did not finish within {max_wait_minutes} min; nothing launched")
        print("waiting for prepare_data to commit ...", flush=True)
        time.sleep(120)
    auxiliary_before = auxiliary.exists()

    print("prepare_data is committed; running the CLAHE smoke", flush=True)
    smoke_result = run_arm.remote("clahe", 42, max_steps=20, max_samples=256, tag="clahe_smoke")
    volume.reload()
    predictions = EXPERIMENTS / "clahe_smoke" / "clahe_seed42" / "gen_val_predictions.npz"
    if not predictions.is_file():
        raise SystemExit(f"SMOKE FAILED: {predictions} was not written; full experiment NOT launched")
    if auxiliary.exists() and not auxiliary_before:
        raise SystemExit(f"SMOKE FAILED: something was written to {auxiliary}; full experiment NOT launched")
    print(f"SMOKE PASSED: {smoke_result}; launching the full experiment", flush=True)
    return experiment.remote(max_steps=max_steps, tag=tag)


@app.local_entrypoint()
def smoke() -> None:
    """One tiny CLAHE run: proves data paths, the dataset swap, checkpoint + prediction writes."""
    print(run_arm.remote("clahe", 42, max_steps=20, max_samples=256, tag="clahe_smoke"))
