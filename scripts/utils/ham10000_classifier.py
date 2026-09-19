"""Single-label multi-class classifier for HAM10000: softmax + CrossEntropy.

WHY NOT REUSE scripts/utils/classifier.py
  That trainer computes `binary_cross_entropy_with_logits` against a masked multi-label target and
  predicts with `torch.sigmoid`. For CheXpert that is correct: 14 independent observations, any
  subset of which can be positive.

  HAM10000 is one diagnosis out of 7, mutually exclusive. Independent sigmoids can assign
  P(mel)=0.8 and P(nv)=0.7 to the same image — a state the ground truth can never occupy — and both
  calibration and the argmax decision inherit that incoherence. Softmax makes the seven
  probabilities sum to 1 by construction, which is what the data says. That is a methodological
  choice about what the model is allowed to represent, not a performance tweak.

  The model architecture, the CUDA/loader settings and the training-budget dataclass ARE shared:
  they are dataset-agnostic and imported rather than copied. Only the loss, the target shape, the
  dataset's __getitem__ and the prediction activation differ, and those are exactly the four places
  the multi-label assumption lives.

  scripts/utils/classifier.py is not modified. The CheXpert path is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.utils.classifier import TrainingBudget, _loader_settings, build_model, require_torch
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS


class LesionRecordDataset:
    """(image, class_index) pairs. One integer target per image, not a label vector.

    Images are loaded as RGB and never converted to greyscale: colour is the diagnostic signal in
    dermoscopy, and the preprocessing step already guaranteed RGB on disk.
    """

    def __init__(self, records: list[dict], resolution: int):
        self.records = records
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        import torch
        from PIL import Image

        record = self.records[index]
        with Image.open(record["image_path"]) as image:
            rgb = image.convert("RGB")
            if rgb.size != (self.resolution, self.resolution):
                rgb = rgb.resize((self.resolution, self.resolution), Image.Resampling.BICUBIC)
            array = np.asarray(rgb, dtype=np.float32) / 255.0

        # HWC -> CHW, normalised to [-1, 1] to match the preprocessing convention used elsewhere.
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - 0.5) / 0.5
        return {"image": tensor, "target": int(record["class_index"])}


def records_from_split(frame, images_dir: Path, image_id_column: str = "image_id", diagnosis_column: str = "dx") -> list[dict]:
    """Build classifier records from a split CSV plus the preprocessed image directory.

    Rows whose preprocessed image is absent are SKIPPED rather than substituted, and the caller can
    see the difference between the split size and the record count.
    """
    from scripts.utils.ham10000 import normalize_diagnosis

    index_of = {label: position for position, label in enumerate(CLASSIFIER_TARGET_LABELS)}
    records = []
    for row in frame.to_dict("records"):
        image_id = str(row[image_id_column])
        path = Path(images_dir) / f"{image_id}.jpg"
        if not path.is_file():
            continue
        records.append(
            {
                "image_id": image_id,
                "image_path": str(path),
                "class_index": index_of[normalize_diagnosis(row[diagnosis_column])],
                "lesion_id": row.get("lesion_id"),
            }
        )
    return records


def class_weights_from_records(records: list[dict], n_classes: int | None = None) -> np.ndarray:
    """Inverse-frequency weights, normalised to mean 1.

    HAM10000 is ~67% `nv`. Unweighted CrossEntropy on that distribution has a strong incentive to
    predict `nv` and ignore the rare classes the thesis is specifically about. Whether to USE these
    weights is an experimental choice that belongs to the run, not to this helper — the function
    computes them, the caller decides.
    """
    n_classes = n_classes if n_classes is not None else len(CLASSIFIER_TARGET_LABELS)
    counts = np.zeros(n_classes, dtype=np.float64)
    for record in records:
        counts[record["class_index"]] += 1
    with np.errstate(divide="ignore"):
        weights = np.where(counts > 0, 1.0 / counts, 0.0)
    positive = weights[weights > 0]
    return weights / positive.mean() if positive.size else weights


def train_classifier(
    train_records: list[dict],
    budget: TrainingBudget,
    dropout_p: float,
    pretrained_source: str,
    resolution: int,
    class_weights: np.ndarray | None = None,
    device: str | None = None,
    progress_desc: str = "train",
):
    """Train to a FIXED optimizer-step budget with softmax + CrossEntropy. Returns (model, history).

    Equal optimizer steps (not equal epochs) is the same fairness rule the CheXpert Stage 4 uses: at
    fixed epochs a larger dataset silently receives more gradient updates, which would conflate
    "more data" with "more training" in exactly the comparison this thesis makes.
    """
    torch = require_torch()
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(budget.seed)

    model = build_model(len(CLASSIFIER_TARGET_LABELS), dropout_p, pretrained_source, seed=budget.seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=budget.learning_rate, weight_decay=getattr(budget, "weight_decay", 1e-4))

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(np.asarray(class_weights, dtype=np.float32), device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    settings = _loader_settings(device, budget.num_workers)
    loader = DataLoader(
        LesionRecordDataset(train_records, resolution),
        batch_size=budget.batch_size,
        shuffle=True,
        drop_last=False,
        **settings,
    )

    model.train()
    history: list[dict] = []
    step = 0
    progress = tqdm(total=budget.max_steps, desc=progress_desc, unit="step")
    while step < budget.max_steps:
        for batch in loader:
            if step >= budget.max_steps:
                break
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            logits = model(images)
            loss = loss_fn(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}")
            history.append({"step": step, "loss": float(loss.item())})
    progress.close()
    return model, history


def _softmax_passes(model, records: list[dict], resolution: int, device: str, n_passes: int) -> np.ndarray:
    """(n_passes, n_images, n_classes) softmax outputs. The loader order is fixed, so pass p and
    pass q describe the same images in the same order and can be compared element-wise."""
    torch = require_torch()
    from torch.utils.data import DataLoader

    settings = _loader_settings(device, None)
    loader = DataLoader(
        LesionRecordDataset(records, resolution),
        batch_size=32,
        shuffle=False,
        # Each pass iterates the loader again; keep the workers alive across passes.
        persistent_workers=settings.get("num_workers", 0) > 0 and n_passes > 1,
        **settings,
    )
    non_blocking = bool(settings.get("pin_memory"))

    passes = []
    with torch.no_grad():
        for _ in range(n_passes):
            outputs = []
            for batch in loader:
                logits = model(batch["image"].to(device, non_blocking=non_blocking))
                outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
            passes.append(
                np.concatenate(outputs, axis=0) if outputs else np.zeros((0, len(CLASSIFIER_TARGET_LABELS)))
            )
    return np.stack(passes, axis=0)


def predict_probabilities(model, records: list[dict], resolution: int, device: str | None = None) -> np.ndarray:
    """(n_images, n_classes) SOFTMAX probabilities — each row sums to 1.

    The sibling CheXpert function applies `sigmoid` per label; this one applies softmax across
    classes, which is the whole point of the different head.

    Deterministic, and it returns a bare array. MC Dropout lives in a SEPARATE function rather than
    behind a `passes` argument here, because the CheXpert sibling switches its return type from an
    array to a (mean, std) tuple depending on that argument — and both callers of this one
    (Stage 4's conditions and the ASISM auxiliary classifier) index the array directly.
    """
    torch = require_torch()

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return _softmax_passes(model, records, resolution, device, 1)[0]


def predict_probability_passes(
    model, records: list[dict], resolution: int, passes: int, device: str | None = None
) -> np.ndarray:
    """The raw (n_passes, n_images, n_classes) MC-Dropout softmax samples.

    Exposed alongside the (mean, std) helper below because the Stage 3 uncertainty signal splits
    total uncertainty into its aleatoric and epistemic parts, and that split needs each pass's own
    distribution: the expected entropy mean_p H[q_p] cannot be recovered from a mean and a standard
    deviation. Returning the samples here keeps that arithmetic in the signal module and the
    inference — the expensive part — in one place.

    Dropout only is switched back on, via the shared `enable_mc_dropout`: calling `model.train()`
    would also put BatchNorm into batch-statistics mode, which changes the predictions themselves
    instead of sampling over dropout masks, and the resulting spread would not be the epistemic
    uncertainty this measures.

    Refuses rather than degenerating:
      * a model with no dropout modules would make every pass identical and the signal a constant;
      * fewer than two passes gives a spread that is exactly zero by construction.
    """
    torch = require_torch()
    from scripts.utils.classifier import enable_mc_dropout

    if int(passes) < 2:
        raise ValueError(
            f"mc_dropout_passes={passes}: the spread over a single pass is identically zero, "
            "so the uncertainty signal would be a constant rather than a measurement"
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    if enable_mc_dropout(model) == 0:
        raise ValueError(
            "MC Dropout requested but the model contains no dropout modules — every pass would be "
            "identical and the uncertainty signal would silently degenerate to a constant. Check "
            "the classifier's dropout_p."
        )

    stacked = _softmax_passes(model, records, resolution, device, int(passes))
    model.eval()  # leave the model as it was handed over, not half in training mode
    return stacked


def predict_probabilities_mc_dropout(
    model, records: list[dict], resolution: int, passes: int, device: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(mean, std) over `passes` stochastic forward passes.

    A summary of `predict_probability_passes`, kept because callers that only need the mean
    prediction and its spread should not have to reduce the sample array themselves. Same guards —
    they live in the function that runs the passes.
    """
    stacked = predict_probability_passes(model, records, resolution, passes, device)
    return stacked.mean(axis=0), stacked.std(axis=0)


def true_class_indices(records: list[dict]) -> np.ndarray:
    return np.array([record["class_index"] for record in records], dtype=np.int64)


__all__ = [
    "LesionRecordDataset",
    "records_from_split",
    "class_weights_from_records",
    "train_classifier",
    "predict_probabilities",
    "predict_probability_passes",
    "predict_probabilities_mc_dropout",
    "true_class_indices",
]
