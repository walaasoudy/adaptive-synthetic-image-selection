"""Save and load a fitted V2 ranker, so the selection reads the fit instead of repeating it.

One JSON file, created exclusively: the point model, the bootstrap ensemble, the train-only
normaliser, the classes, the fit history and which images were in the fitting subsets. Weights are
float32; a float32 written as a decimal and read back is the same float32, so a loaded ranker scores
and selects exactly as the fitted one did. The file carries its own content hash and the provenance
the caller gives it; load_fitted refuses a file whose content or expected provenance does not match.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .contracts import fingerprint, write_new_json
from .features import SIGNALS, TrainingStandardizer
from .pipeline import AdditiveUtilityRanker, FittedRanking

FORMAT = "asism_v2_fitted_ranking/1"


def _state(model: AdditiveUtilityRanker) -> dict:
    return {name: value.tolist() for name, value in model.state_dict().items()}


def _model(state: dict, n_classes: int) -> AdditiveUtilityRanker:
    model = AdditiveUtilityRanker(n_classes)
    model.load_state_dict({name: torch.tensor(value, dtype=torch.float32) for name, value in state.items()})
    model.eval()
    return model


def save_fitted(path: Path, fitted: FittedRanking, provenance: dict) -> str:
    """-> the content hash. Refuses to replace an existing file."""
    body = {"format": FORMAT, "signals": list(SIGNALS), "classes": list(fitted.classes),
            "normalizer": {"train_ids": list(fitted.normalizer.train_ids),
                           "mean": list(fitted.normalizer.mean), "scale": list(fitted.normalizer.scale)},
            "history": fitted.history, "train_image_ids": sorted(fitted.train_image_ids),
            "validation_image_ids": sorted(fitted.validation_image_ids),
            "model": _state(fitted.model), "ensemble": [_state(m) for m in fitted.ensemble],
            "provenance": dict(provenance)}
    content = fingerprint(body)
    write_new_json(Path(path), {**body, "content_sha256": content})
    return content


def load_fitted(path: Path, expected_provenance: dict | None = None) -> tuple[FittedRanking, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    content = payload.pop("content_sha256", None)
    if payload.get("format") != FORMAT or content != fingerprint(payload):
        raise ValueError(f"{path}: not a fitted ranker of format {FORMAT}, or its content was changed")
    if payload["signals"] != list(SIGNALS):
        raise ValueError(f"{path}: fitted on other signals")
    for key, value in (expected_provenance or {}).items():
        if payload["provenance"].get(key) != value:
            raise ValueError(f"{path}: stale ranker, {key} does not match")
    classes = tuple(payload["classes"])
    normalizer = TrainingStandardizer(tuple(payload["normalizer"]["train_ids"]),
                                      tuple(payload["normalizer"]["mean"]),
                                      tuple(payload["normalizer"]["scale"]))
    fitted = FittedRanking(_model(payload["model"], len(classes)), normalizer, classes, payload["history"],
                           frozenset(payload["train_image_ids"]), frozenset(payload["validation_image_ids"]),
                           tuple(_model(state, len(classes)) for state in payload["ensemble"]))
    return fitted, {**payload["provenance"], "content_sha256": content}
