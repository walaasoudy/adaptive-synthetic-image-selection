"""Isolated CPU V2 path: measured set utility -> image weights (selection: stopping.py).

Set utility (Walaa, 2026-10-03, contract §9): sum pooling inside the size term,
    U(S) = b + lam * log(1 + sum_{i in S} w_i),   w_i = 1 + score(x_i, class_i) + class term.
w_i is the image's effective count: 1 is an average image, above 1 counts for more, 0 adds
nothing and a negative weight removes value. An image's marginal utility is its own weight's,
whatever else is in the set. The earlier mean-pooled form (mean score + lam * log(1 + n)) is
retired: there an image below the current set mean was predicted to hurt, so the count came from
dilution, not from utility. Image interactions are still not represented. No V1 module is imported.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math

import numpy as np
import pandas as pd
import torch
from torch import nn

from .contracts import validate_measurements
from .features import SIGNALS, TrainingStandardizer, validate_frame


SIZE_TERM_INITIAL = 0.02          # lam at the start of a fit: nonzero, or no gradient reaches the weights
SIZE_TERM_WHEN_FROZEN = 1.0       # one train size: lam is not identifiable and only fixes the score scale


class AdditiveUtilityRanker(nn.Module):
    """Within-class image ranking supervised by measured subset performance.

    U(S) = intercept + lam * log(1 + sum_i w_i),  w_i = 1 + score(x_i, class_i) + class_mix[class_i]
    (class_mix centred over classes). Class-specific score weights make signal directions learnable.
    `log_count` is lam. A set whose weights sum below 0 is worth the intercept.
    """

    def __init__(self, n_classes: int, n_features: int = len(SIGNALS)):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_classes, n_features))
        self.intercept = nn.Parameter(torch.tensor(0.5))
        self.log_count = nn.Parameter(torch.tensor(SIZE_TERM_INITIAL))
        self.class_mix = nn.Parameter(torch.zeros(n_classes))

    def score(self, x: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        return (self.weights[classes] * x).sum(-1)

    def image_weight(self, x: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        """The image's effective count w_i."""
        return 1 + self.score(x, classes) + (self.class_mix - self.class_mix.mean())[classes]

    def forward(self, x: torch.Tensor, classes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or classes.shape != mask.shape or classes.shape != x.shape[:2]:
            raise ValueError("Invalid batched subset dimensions")
        if mask.dtype != torch.bool or not mask.any(dim=1).all():
            raise ValueError("Expected nonempty Boolean subset mask")
        if not torch.isfinite(x[mask]).all():
            raise ValueError("Nonfinite active features")
        total = self.image_weight(x, classes).masked_fill(~mask, 0).sum(1)
        return self.intercept + self.log_count * torch.log1p(torch.relu(total))


@dataclass
class FittedRanking:
    model: AdditiveUtilityRanker
    normalizer: TrainingStandardizer
    classes: tuple[str, ...]
    history: dict
    train_image_ids: frozenset = frozenset()
    validation_image_ids: frozenset = frozenset()
    ensemble: tuple = ()

    def score_source(self, image_ids) -> list[str]:
        """Where each score comes from, recorded by the ranker itself rather than declared by a
        caller: 'train_fit' (its subsets set the weights), 'validation_early_stopping' (its
        subsets chose the epoch) or 'heldout' (in no fitting or early-stopping subset)."""
        return ["train_fit" if i in self.train_image_ids else
                "validation_early_stopping" if i in self.validation_image_ids else "heldout"
                for i in map(str, image_ids)]

    def score_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        validate_frame(frame)
        if "dx" not in frame or frame.dx.isna().any():
            raise ValueError("Missing diagnosis")
        if set(frame.dx.astype(str)) - set(self.classes):
            raise ValueError("Unknown diagnosis")
        x = torch.tensor(self.normalizer.transform(frame), dtype=torch.float32)
        class_ids = torch.tensor([self.classes.index(str(dx)) for dx in frame.dx], dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            scores = self.model.score(x, class_ids).numpy()
            weights = self.model.image_weight(x, class_ids).numpy()
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite ranking score")
        ids = frame.image_id.astype(str).to_numpy()
        return pd.DataFrame({"image_id": ids,
                             "dx": frame.dx.astype(str).to_numpy(),
                             "ranking_score": scores,
                             "image_weight": weights,
                             "score_source": self.score_source(ids)})


def _matrix(subsets: dict[str, dict], ids: list[str], frame: pd.DataFrame,
            normalizer: TrainingStandardizer, classes: tuple[str, ...]):
    indexed = frame.set_index(frame.image_id.astype(str))
    max_len = max(len(subsets[sid]["image_ids"]) for sid in ids)
    x = np.zeros((len(ids), max_len, len(SIGNALS)), dtype=np.float32)
    c = np.zeros((len(ids), max_len), dtype=np.int64)
    mask = np.zeros((len(ids), max_len), dtype=bool)
    for row, sid in enumerate(ids):
        member_ids = [str(v) for v in subsets[sid]["image_ids"]]
        if len(member_ids) != len(set(member_ids)) or set(member_ids) - set(indexed.index):
            raise ValueError("Duplicated or missing subset member")
        members = indexed.loc[member_ids].reset_index(drop=True)
        count = len(member_ids)
        x[row, :count] = normalizer.transform(members)
        c[row, :count] = [classes.index(str(dx)) for dx in members.dx]
        mask[row, :count] = True
    return (torch.from_numpy(x), torch.from_numpy(c), torch.from_numpy(mask))


def fit_ranker(frame: pd.DataFrame, subsets: dict[str, dict], measurements: list[dict],
               seeds: list[int], protocol: dict, *, seed: int = 42, max_epochs: int = 600,
               patience: int = 60, learning_rate: float = 0.03, bootstrap: int = 0) -> FittedRanking:
    """Train on measured subset outcomes; tune on image-disjoint validation.

    The protected test subsets are never inspected in fitting or early stopping.
    No image-level teacher targets are distilled, so no OOF teacher is needed.
    The real instrument must pass a separately frozen reliability/validity gate.

    `bootstrap` > 0 also fits that many models, each on a resample (with replacement) of the
    TRAIN subsets with the same configuration and early stopping on the same validation subsets.
    Their spread gives the confidence bound the contract §10 stopping rule needs. Sizes may differ
    between subsets (contract §8): the size term is identifiable only then, and the stopping rule
    refuses a model whose size term was frozen.
    """
    validate_frame(frame)
    if "dx" not in frame or frame.dx.isna().any():
        raise ValueError("Missing diagnosis")
    decision = protocol.get("instrument_decision")
    if decision != "accepted" and decision != "synthetic_toy_only":
        raise ValueError("Utility instrument has not passed a predeclared validity gate")
    if decision == "synthetic_toy_only" and protocol.get("dataset") != "synthetic_toy":
        raise ValueError("Toy bypass cannot be used on real data")
    if decision == "accepted":
        required = {"dataset", "metric", "recipe_sha256", "classifier_recipe_sha256",
                    "feature_artifacts_sha256", "candidate_pool_sha256", "outcome_split_sha256",
                    "measurement_code_sha256", "instrument_gate_sha256"}
        if (required - set(protocol) or
                any(not isinstance(protocol[key], str) or not protocol[key]
                    for key in required)):
            raise ValueError("Accepted instrument lacks complete provenance")
    if not subsets or not {"train", "validation", "test"} <= {r["role"] for r in subsets.values()}:
        raise ValueError("Train, validation and test subset roles required")
    # Identifiability is a property of the subsets that set the weights: the TRAIN subsets only.
    # A second size, or a second class allocation, that appears only in validation or test
    # subsets gives the fit nothing to estimate the size or class-mix term from.
    train_subset_ids = [sid for sid, item in subsets.items() if item["role"] == "train"]
    sizes = {len(subsets[sid]["image_ids"]) for sid in train_subset_ids}
    if max_epochs < 1 or bootstrap < 0 or patience < 1 or learning_rate <= 0:
        raise ValueError("Invalid training configuration")
    memberships: dict[str, set[str]] = {role: set() for role in ("train", "validation", "test")}
    member_map = {}
    for sid, subset in subsets.items():
        role = subset["role"]
        if role not in memberships:
            raise ValueError("Unknown subset role")
        ids = list(map(str, subset["image_ids"]))
        memberships[role].update(ids)
        member_map[sid] = ids
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if memberships[left] & memberships[right]:
            raise ValueError("Train/validation/test image leakage")
    if set().union(*memberships.values()) - set(frame.image_id.astype(str)):
        raise ValueError("Subset references absent feature row")
    image_classes = frame.set_index(frame.image_id.astype(str)).dx.astype(str)
    all_classes = set(image_classes)
    for role in ("train", "validation", "test"):
        if set(image_classes.loc[sorted(memberships[role])]) != all_classes:
            raise ValueError(f"Class coverage missing in {role} subsets")
    test_ids = {sid for sid, item in subsets.items() if item["role"] == "test"}
    if any(row.get("subset_id") in test_ids for row in measurements):
        raise ValueError("Test subset outcomes cannot enter ranker fitting")
    fitting_members = {sid: ids for sid, ids in member_map.items() if sid not in test_ids}
    values = validate_measurements(measurements, fitting_members, seeds, protocol)
    normalizer = TrainingStandardizer.fit(frame, sorted(memberships["train"]))
    classes = tuple(sorted(frame.dx.astype(str).unique()))
    if len(classes) < 2:
        raise ValueError("At least two classes required")
    tensors = {}
    targets = {}
    for role in ("train", "validation"):
        subset_ids = sorted(sid for sid, entry in subsets.items() if entry["role"] == role)
        tensors[role] = _matrix(subsets, subset_ids, frame, normalizer, classes)
        targets[role] = torch.tensor([np.mean(values[sid]) for sid in subset_ids], dtype=torch.float32)
    frozen = []
    if len(sizes) == 1:
        # At one size lam and the scale of the scores cannot be told apart: lam is held at
        # SIZE_TERM_WHEN_FROZEN. The within-class order is still learned; counts are not.
        frozen.append("log_count")
    # Class FRACTIONS, not counts: the model's class-mix term is a function of the fractions, so
    # proportional subsets of different sizes (2:2, 4:4, ...) still leave it a constant.
    allocations = set()
    for sid in train_subset_ids:
        member_classes = image_classes.loc[list(map(str, subsets[sid]["image_ids"]))].value_counts()
        allocations.add(tuple(sorted((dx, round(n / member_classes.sum(), 12))
                                     for dx, n in member_classes.items())))
    if len(allocations) == 1:
        # The same class fractions in every train subset make the class-mix term a constant,
        # confounded with the intercept: it cannot be identified, so it is not fitted.
        frozen.append("class_mix")
    config = (len(classes), tuple(frozen), max_epochs, patience, learning_rate)
    model, best, best_epoch = _train(tensors["train"], targets["train"], tensors["validation"],
                                     targets["validation"], seed, *config)
    ensemble = []
    n_train = len(targets["train"])
    for b in range(bootstrap):
        rows = torch.from_numpy(np.random.default_rng([seed, b]).integers(0, n_train, n_train))
        resampled = tuple(t[rows] for t in tensors["train"])
        member, _, _ = _train(resampled, targets["train"][rows], tensors["validation"],
                              targets["validation"], seed + 1 + b, *config)
        ensemble.append(member)
    return FittedRanking(model, normalizer, classes,
                         {"best_epoch": best_epoch, "validation_mse": best,
                          "train_subsets": n_train,
                          "validation_subsets": len(targets["validation"]),
                          "subset_sizes": sorted(sizes),
                          "training_seed": seed, "instrument_decision": decision,
                          "frozen_unidentifiable": frozen, "bootstrap_models": bootstrap},
                         frozenset(memberships["train"]), frozenset(memberships["validation"]),
                         tuple(ensemble))


def _train(train, train_y, validation, validation_y, seed, n_classes, frozen, max_epochs,
           patience, learning_rate):
    """One fit with validation early stopping. Frozen parameters keep their initial value."""
    torch.manual_seed(seed)
    model = AdditiveUtilityRanker(n_classes)
    for name in frozen:
        getattr(model, name).requires_grad_(False)
    if "log_count" in frozen:
        with torch.no_grad():
            model.log_count.fill_(SIZE_TERM_WHEN_FROZEN)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=learning_rate, weight_decay=1e-5)
    best, best_state, best_epoch = float("inf"), None, -1
    for epoch in range(max_epochs):
        model.train()
        loss = torch.nn.functional.mse_loss(model(*train), train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val = float(torch.nn.functional.mse_loss(model(*validation), validation_y))
        if math.isfinite(val) and val < best - 1e-9:
            best, best_epoch = val, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break
    if best_state is None:
        raise ValueError("No finite validation fit")
    model.load_state_dict(best_state)
    model.eval()
    return model, best, best_epoch
