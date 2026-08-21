"""Shared multi-label CXR classifier: model, dataset, masked loss, training loop, inference.

One implementation, three consumers, so they cannot drift apart:
  - the auxiliary reference classifier (Stage 3 prerequisite, docs/stages2_to_5_plan.md §2);
  - the ASISM proxy classifier used to rank candidate configurations (§4.7);
  - the A-E headline experiment classifiers (§7).

Two design points are load-bearing rather than incidental:

MC Dropout (§4.3) requires dropout ACTIVE at inference. `model.eval()` disables it, so
`enable_mc_dropout()` re-enables only the dropout modules while leaving BatchNorm in eval mode —
switching the whole model back to train() would also make BatchNorm use batch statistics, which
would corrupt the predictions the uncertainty estimate is computed from.

Masked loss (§6) implements the frozen uncertainty policy: -1 and blank labels contribute to
neither the loss nor any metric, and are never converted into a 0 or 1 the model would be trained
to reproduce.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts.utils.labels import CLASSIFIER_TARGET_LABELS, build_label_arrays


@dataclass
class TrainingBudget:
    """Frozen fairness protocol (§7.1): budget is expressed in OPTIMIZER STEPS, not epochs.

    At fixed epochs a larger dataset silently receives more gradient updates, which would conflate
    "more data" with "more training" — exactly the confound condition D exists to rule out.
    """

    max_steps: int
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 0
    eval_every_n_steps: int = 200
    num_workers: int = 0
    seed: int = 42


@dataclass
class RunAccounting:
    """Per-run record required by §7.1 — what was actually consumed, not what was configured."""

    optimizer_steps: int = 0
    epochs_completed: float = 0.0
    effective_dataset_size: int = 0
    real_sample_exposures: int = 0
    synthetic_sample_exposures: int = 0
    sampling_policy: str = "shuffled_concat"
    model_seed: int = 0
    dataset_id: str = ""
    config_hash: str = ""
    checkpoint_hash: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "optimizer_steps": self.optimizer_steps,
            "epochs_completed": round(self.epochs_completed, 4),
            "effective_dataset_size": self.effective_dataset_size,
            "real_sample_exposures": self.real_sample_exposures,
            "synthetic_sample_exposures": self.synthetic_sample_exposures,
            "sampling_policy": self.sampling_policy,
            "model_seed": self.model_seed,
            "dataset_id": self.dataset_id,
            "config_hash": self.config_hash,
            "checkpoint_hash": self.checkpoint_hash,
            **self.extra,
        }


def require_torch():
    """Import torch lazily with a clean, actionable message when the GPU stack is absent."""
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"UPSTREAM GATE: the training stack is not installed here ({exc}).\n"
            "Classifier training is GPU work — run it on the RunPod pod after:\n"
            "  pip install -r environment/requirements.txt"
        ) from exc
    import torch

    return torch


class CXRRecordDataset:
    """Multi-label CXR dataset over an explicit record list.

    Each record is {"image_path": str, "labels": {label: raw_value}, "is_synthetic": bool}. Keeping
    real and synthetic rows in one uniform record type is what lets the Stage 4 conditions differ
    only by which records they are handed, rather than by separate code paths per condition.
    """

    def __init__(self, records: list[dict], resolution: int = 320, augment: bool = False):
        self.records = records
        self.resolution = resolution
        self.augment = augment
        self._labels = CLASSIFIER_TARGET_LABELS

        frame = pd.DataFrame([record["labels"] for record in records])
        for label in self._labels:
            if label not in frame.columns:
                frame[label] = np.nan
        _, self.targets, self.masks = build_label_arrays(frame[self._labels], self._labels)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import torch
        from PIL import Image
        from torchvision.transforms import functional as TF

        record = self.records[index]
        with Image.open(record["image_path"]) as image:
            image = image.convert("RGB")
            if image.size != (self.resolution, self.resolution):
                image = image.resize((self.resolution, self.resolution), Image.BICUBIC)
            tensor = TF.to_tensor(image)

        # Horizontal flip is deliberately NOT offered: it inverts CXR left/right anatomy
        # (heart position, aortic arch, gastric bubble) — same constraint as Stage 1's data config.
        tensor = TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return {
            "image": tensor,
            "target": torch.from_numpy(self.targets[index]),
            "mask": torch.from_numpy(self.masks[index]),
            "index": index,
        }


def build_model(num_labels: int, dropout_p: float, pretrained_source: str, seed: int = 42):
    """DenseNet121 with a dropout layer before the classifier head.

    `pretrained_source` is recorded verbatim in run provenance. CheXpert-pretrained weights are
    rejected here rather than merely discouraged: most published CheXpert checkpoints do not
    publish patient-level provenance, so overlap with final_eval_heldout cannot be excluded (§2).
    """
    torch = require_torch()
    import torchvision

    if "chexpert" in str(pretrained_source).lower():
        raise SystemExit(
            f"Refusing CheXpert-pretrained initialization ({pretrained_source!r}).\n"
            "Its training patients cannot be shown disjoint from our evaluation splits, which "
            "would contaminate the headline comparison (docs/stages2_to_5_plan.md §2).\n"
            "Use 'imagenet', 'random', or a CXR checkpoint with published patient-level provenance."
        )

    torch.manual_seed(seed)
    if pretrained_source == "imagenet":
        weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
    elif pretrained_source == "random":
        weights = None
    else:
        raise SystemExit(
            f"Unsupported pretrained_source {pretrained_source!r}. "
            "Supported: 'imagenet', 'random'. A documented non-overlapping CXR checkpoint may be "
            "added here once its provenance is recorded (§2)."
        )

    model = torchvision.models.densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=dropout_p),
        torch.nn.Linear(in_features, num_labels),
    )
    return model


def enable_mc_dropout(model) -> int:
    """Put ONLY dropout modules into train mode, leaving BatchNorm in eval mode.

    Calling model.train() instead would also switch BatchNorm to batch statistics, which changes
    the predictions themselves rather than just sampling over dropout masks — the resulting spread
    would not be the epistemic uncertainty §4.3 intends to measure.
    """
    import torch

    count = 0
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d)):
            module.train()
            count += 1
    return count


def masked_bce_loss(logits, targets, masks):
    """Binary cross-entropy over confidently-labelled positions only (§6).

    Uncertain (-1) and blank labels are excluded from the loss entirely rather than mapped to 0/1,
    so the model is never trained to reproduce an assertion the radiologist declined to make.
    """
    import torch
    import torch.nn.functional as F

    per_element = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    masked = per_element * masks.float()
    denominator = masks.float().sum()
    if denominator.item() == 0:
        return torch.zeros((), device=logits.device, requires_grad=True)
    return masked.sum() / denominator


def train_classifier(
    train_records: list[dict],
    val_records: list[dict],
    budget: TrainingBudget,
    dropout_p: float,
    pretrained_source: str,
    resolution: int,
    device: str | None = None,
    progress_desc: str = "train",
    checkpoint_path: Path | None = None,
    resume_from: Path | None = None,
    checkpoint_metadata: dict | None = None,
) -> tuple[object, RunAccounting, list[dict]]:
    """Train to a FIXED optimizer-step budget. Returns (model, accounting, eval_history).

    Deliberately no early stopping inside this function: the ASISM proxy search forbids
    candidate-specific early stopping (§4.7), and A-E requires an identical protocol across
    conditions (§7.1). Checkpoint SELECTION (best-on-classifier_val) is applied by the caller that
    is entitled to do it, from the history returned here.
    """
    torch = require_torch()
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(budget.seed)
    torch.manual_seed(budget.seed)
    np.random.seed(budget.seed)

    train_dataset = CXRRecordDataset(train_records, resolution=resolution)
    sampler_generator = torch.Generator().manual_seed(budget.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=budget.batch_size,
        shuffle=True,
        num_workers=budget.num_workers,
        drop_last=False,
        generator=sampler_generator,
    )

    model = build_model(len(CLASSIFIER_TARGET_LABELS), dropout_p, pretrained_source, budget.seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=budget.learning_rate, weight_decay=budget.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, budget.max_steps))

    accounting = RunAccounting(
        effective_dataset_size=len(train_records),
        real_sample_exposures=0,
        synthetic_sample_exposures=0,
        model_seed=budget.seed,
    )
    n_synthetic = sum(1 for record in train_records if record.get("is_synthetic"))
    n_real = len(train_records) - n_synthetic

    history: list[dict] = []
    step = 0
    best_metric = float("-inf")
    best_checkpoint_identity = None
    expected_meta = checkpoint_metadata or {}
    best_checkpoint_path = (
        checkpoint_path.with_name(f"{checkpoint_path.stem}.best{checkpoint_path.suffix}")
        if checkpoint_path is not None else None
    )
    if resume_from is not None:
        payload = torch.load(resume_from, map_location=device, weights_only=False)
        if payload.get("checkpoint_schema_version") != 2:
            raise ValueError("incompatible classifier checkpoint schema")
        actual_meta = payload.get("provenance", {})
        mismatches = {key: (actual_meta.get(key), value) for key, value in expected_meta.items() if actual_meta.get(key) != value}
        if mismatches:
            raise ValueError(f"incompatible classifier checkpoint provenance: {mismatches}")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        step = int(payload["global_optimizer_step"])
        history = list(payload.get("eval_history", []))
        best_metric = float(payload.get("best_validation_metric", float("-inf")))
        best_checkpoint_identity = payload.get("best_checkpoint_identity")
        random.setstate(payload["rng_states"]["python"])
        np.random.set_state(payload["rng_states"]["numpy"])
        # ``map_location=device`` also moves the saved CPU RNG tensor to CUDA on GPU runs.
        # PyTorch requires a CPU ByteTensor when restoring the default CPU generator.
        torch_cpu_rng = payload["rng_states"]["torch_cpu"].detach().cpu()
        torch.set_rng_state(torch_cpu_rng)
        if torch.cuda.is_available() and payload["rng_states"].get("torch_cuda"):
            torch.cuda.set_rng_state_all(
                [state.detach().cpu() for state in payload["rng_states"]["torch_cuda"]]
            )
        sampler_generator.set_state(payload["sampler_state"].detach().cpu())

    def save_resumable() -> None:
        nonlocal best_checkpoint_identity
        if checkpoint_path is None:
            return
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        best_checkpoint_identity = best_checkpoint_identity or f"step-{step}"
        payload = {
            "checkpoint_schema_version": 2, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "global_optimizer_step": step, "step": step,
            "best_validation_metric": best_metric, "best_checkpoint_identity": best_checkpoint_identity,
            "early_stopping_state": {"enabled": False, "bad_evaluations": 0},
            "rng_states": {"python": random.getstate(), "numpy": np.random.get_state(),
                           "torch_cpu": torch.get_rng_state(),
                           "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
            "sampler_state": sampler_generator.get_state(), "sampler_rule": "torch RandomSampler generator state",
            "eval_history": history, "provenance": expected_meta,
        }
        fd, temp_name = tempfile.mkstemp(prefix=f".{checkpoint_path.name}.", suffix=".tmp", dir=checkpoint_path.parent)
        os.close(fd)
        try:
            torch.save(payload, temp_name)
            os.replace(temp_name, checkpoint_path)
        finally:
            Path(temp_name).unlink(missing_ok=True)

    def save_best() -> None:
        if best_checkpoint_path is None:
            return
        best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_schema_version": 2, "checkpoint_role": "best_validation_model",
            "model_state": model.state_dict(), "best_validation_metric": best_metric,
            "best_checkpoint_identity": best_checkpoint_identity, "provenance": expected_meta,
            "model_config": {"dropout_p": dropout_p, "pretrained_source": pretrained_source,
                             "resolution": resolution, "num_labels": len(CLASSIFIER_TARGET_LABELS)},
        }
        fd, temp_name = tempfile.mkstemp(prefix=f".{best_checkpoint_path.name}.", suffix=".tmp", dir=best_checkpoint_path.parent)
        os.close(fd)
        try:
            torch.save(payload, temp_name)
            os.replace(temp_name, best_checkpoint_path)
        finally:
            Path(temp_name).unlink(missing_ok=True)
    model.train()
    progress = tqdm(total=budget.max_steps, initial=step, desc=progress_desc, unit="step")

    while step < budget.max_steps:
        for batch in loader:
            if step >= budget.max_steps:
                break
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss = masked_bce_loss(logits, targets, masks)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}", refresh=False)

            if val_records and budget.eval_every_n_steps and step % budget.eval_every_n_steps == 0:
                metrics = evaluate_classifier(model, val_records, resolution, device)
                metrics["step"] = step
                history.append(metrics)
                if np.isfinite(metrics["macro_auroc"]) and metrics["macro_auroc"] >= best_metric:
                    best_metric = metrics["macro_auroc"]
                    best_checkpoint_identity = f"step-{step}"
                    save_best()
                save_resumable()
                model.train()

    progress.close()

    accounting.optimizer_steps = step
    seen = step * budget.batch_size
    accounting.epochs_completed = seen / max(len(train_records), 1)
    total = max(len(train_records), 1)
    accounting.real_sample_exposures = int(seen * n_real / total)
    accounting.synthetic_sample_exposures = int(seen * n_synthetic / total)

    if val_records:
        final_metrics = evaluate_classifier(model, val_records, resolution, device)
        final_metrics["step"] = step
        history.append(final_metrics)
        if np.isfinite(final_metrics["macro_auroc"]) and final_metrics["macro_auroc"] >= best_metric:
            best_metric = final_metrics["macro_auroc"]
            best_checkpoint_identity = f"step-{step}"
            save_best()
    save_resumable()

    if val_records and (best_checkpoint_path is None or not best_checkpoint_path.is_file()):
        raise RuntimeError(
            "No finite validation macro-AUROC was observed; refusing to publish an arbitrary final-step model as best"
        )

    if best_checkpoint_path is not None and best_checkpoint_path.is_file():
        best_payload = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best_payload["model_state"])
        accounting.extra["best_checkpoint_path"] = str(best_checkpoint_path)
        accounting.extra["best_validation_metric"] = best_metric

    return model, accounting, history


def predict_probabilities(
    model,
    records: list[dict],
    resolution: int,
    device: str | None = None,
    batch_size: int = 32,
    mc_dropout_passes: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Predict per-label probabilities.

    With mc_dropout_passes > 0, returns (mean, std) over stochastic forward passes — the §4.3
    uncertainty signal. With 0, returns (probabilities, None) from a single deterministic pass.
    """
    torch = require_torch()
    from torch.utils.data import DataLoader

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dataset = CXRRecordDataset(records, resolution=resolution)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    n_passes = max(1, mc_dropout_passes)
    if mc_dropout_passes > 0:
        activated = enable_mc_dropout(model)
        if activated == 0:
            raise SystemExit(
                "MC Dropout requested but the model contains no dropout modules — the uncertainty "
                "signal would silently degenerate to a constant. Check classifier.dropout_p (§2)."
            )

    all_passes = []
    with torch.no_grad():
        for _ in range(n_passes):
            batch_outputs = []
            for batch in loader:
                logits = model(batch["image"].to(device))
                batch_outputs.append(torch.sigmoid(logits).cpu().numpy())
            all_passes.append(np.concatenate(batch_outputs, axis=0))

    stacked = np.stack(all_passes, axis=0)
    mean = stacked.mean(axis=0)
    if mc_dropout_passes > 0:
        return mean, stacked.std(axis=0)
    return mean, None


def evaluate_classifier(model, records: list[dict], resolution: int, device: str | None = None) -> dict:
    """Masked macro-AUROC over the primary endpoint labels, plus effective N (§5.3, §6)."""
    from scripts.utils.metrics import macro_auroc_masked

    probabilities, _ = predict_probabilities(model, records, resolution, device)

    frame = pd.DataFrame([record["labels"] for record in records])
    for label in CLASSIFIER_TARGET_LABELS:
        if label not in frame.columns:
            frame[label] = np.nan
    _, targets, masks = build_label_arrays(frame[CLASSIFIER_TARGET_LABELS], CLASSIFIER_TARGET_LABELS)

    return macro_auroc_masked(probabilities, targets, masks)


def records_from_split(
    frame: pd.DataFrame,
    images_dir: Path,
    path_column: str = "Path",
    require_exists: bool = True,
) -> list[dict]:
    """Build classifier records from a real-data split, using Stage 1's preprocessed images.

    Rows whose preprocessed image is absent (filtered as lateral/corrupt during Stage 1
    preprocessing) are skipped rather than silently substituted.
    """
    from scripts.utils.identifiers import sanitize_image_id

    records = []
    for row in frame.to_dict("records"):
        image_id = sanitize_image_id(row[path_column])
        image_path = images_dir / f"{image_id}.jpg"
        if require_exists and not image_path.is_file():
            continue
        records.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "patient_id": row.get("patient_id"),
                "labels": {label: row.get(label) for label in CLASSIFIER_TARGET_LABELS},
                "is_synthetic": False,
            }
        )
    return records


def records_from_synthetic_manifest(
    manifest_path: Path,
    images_dir: Path,
    image_ids: Iterable[str] | None = None,
    require_exists: bool = True,
) -> list[dict]:
    """Build classifier records from Stage 2 output.

    The synthetic label vector is the INTENDED vector (§3) — the generator's conditioning intent,
    which Stage 3 measures agreement against and which Stage 4 then trains on. It is labelled
    `intended_label_vector` in the manifest and is never described as ground truth.
    """
    wanted = set(image_ids) if image_ids is not None else None
    records = []
    with open(manifest_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            image_id = entry["image_id"]
            if wanted is not None and image_id not in wanted:
                continue
            image_path = images_dir / f"{image_id}.jpg"
            if require_exists and not image_path.is_file():
                continue
            labels = {label: value for label, value in entry["intended_label_vector"].items()}
            labels["No Finding"] = 1 if entry.get("is_no_finding") else 0
            labels["Support Devices"] = int(entry.get("support_devices", 0))
            records.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "patient_id": f"synthetic::{image_id}",
                    "labels": labels,
                    "is_synthetic": True,
                }
            )
    return records
