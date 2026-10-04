"""Isolated CPU V2 path: measured set utility -> image weights (selection: stopping.py).

Set utility (Walaa, 2026-10-03, contract §9): sum pooling inside the size term,
    U(S) = b + lam * log(1 + sum_{i in S} w_i),   w_i = 1 + score(x_i, class_i) + class term.
w_i is the image's effective count: 1 is an average image, above 1 counts for more, 0 adds
nothing and a negative weight removes value. An image's marginal utility is its own weight's,
whatever else is in the set. The earlier mean-pooled form (mean score + lam * log(1 + n)) is
retired: there an image below the current set mean was predicted to hurt, so the count came from
dilution, not from utility. Image interactions are still not represented. No V1 module is imported.

The per-image score (contract §9, amendment of 2026-10-04 and its one documented fix): a network
with one hidden layer and one output per class, NetworkUtilityRanker. The class-specific linear score, AdditiveUtilityRanker, is the form before
that amendment and stays as the reported baseline; both give an image one weight of its own, so
stopping.py reads either.
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
HIDDEN_UNITS = 8                  # contract §9, amendment of 2026-10-04: one hidden layer of 8 tanh units
ARCHITECTURES = ("linear", "mlp")


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

    def pool(self, x: torch.Tensor, classes: torch.Tensor, mask: torch.Tensor):
        """-> per-class signal sums [subsets, classes, signals] and per-class counts [subsets, classes].
        The sum of a subset's image weights is linear in these, so a fit needs nothing else."""
        member = (torch.nn.functional.one_hot(classes, self.weights.shape[0]) * mask.unsqueeze(-1)).to(x.dtype)
        return torch.einsum("bmc,bmf->bcf", member, x.masked_fill(~mask.unsqueeze(-1), 0)), member.sum(1)

    def forward_pooled(self, sums: torch.Tensor, counts: torch.Tensor, size_centre: float = 0.0) -> torch.Tensor:
        """forward() from pool()'s output. `size_centre` is subtracted from the size term; it is
        used only while fitting on standardised targets and is folded into the intercept after."""
        total = (counts.sum(1) + (self.weights * sums).sum((1, 2))
                 + (counts * (self.class_mix - self.class_mix.mean())).sum(1))
        return self.intercept + self.log_count * (torch.log1p(torch.relu(total)) - size_centre)

    # What a fit needs of a batch of subsets is computed once and reused every epoch.
    def prepare(self, x: torch.Tensor, classes: torch.Tensor, mask: torch.Tensor) -> tuple:
        return self.pool(x, classes, mask)

    def predict_prepared(self, prepared: tuple, size_centre: float = 0.0) -> torch.Tensor:
        return self.forward_pooled(*prepared, size_centre)

    @staticmethod
    def prepared_sizes(prepared: tuple) -> torch.Tensor:
        return prepared[1].sum(1)

    @staticmethod
    def resample(prepared: tuple, rows: torch.Tensor) -> tuple:
        """The prepared batch restricted to (and repeating) the subsets `rows`: a bootstrap draw."""
        return tuple(part[rows] for part in prepared)

    def restrict_signals(self, mask: tuple[bool, ...]) -> None:
        """Hold the unflagged signals out of the score: their weights start at 0 and a zeroed
        gradient keeps them there."""
        keep = torch.tensor(mask, dtype=torch.float32)
        self.weights.register_hook(lambda gradient: gradient * keep)


class NetworkUtilityRanker(AdditiveUtilityRanker):
    """The same set utility with the per-image score given by a small network (contract §9,
    amendment of 2026-10-04):

        score(x_i, class_i) = output_{class_i}(tanh(hidden(x_i)))

    One hidden layer over the four signals, one output unit per class. The outputs start at zero,
    so every image starts at weight 1, as the linear score does. w_i, lam, the intercept and the
    class term are as in AdditiveUtilityRanker, and an image's weight is still its own, whatever
    else is in the set.

    The class selects the output and is not an input: that is the one documented fix of the
    amendment. With the class as an added input, eight units could not give each class its own
    signal directions and the network failed the capacity check on planted data.
    """

    def __init__(self, n_classes: int, n_features: int = len(SIGNALS), hidden_units: int = HIDDEN_UNITS):
        super().__init__(n_classes, n_features)
        if hidden_units < 1:
            raise ValueError("The hidden layer needs at least one unit")
        del self.weights
        self.n_classes = n_classes
        self.hidden = nn.Linear(n_features, hidden_units)
        self.output = nn.Linear(hidden_units, n_classes)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("signal_keep", torch.ones(n_features))

    def score(self, x: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        every_class = self.output(torch.tanh(self.hidden(x * self.signal_keep)))
        return every_class.gather(-1, classes.unsqueeze(-1)).squeeze(-1)

    def pool(self, x, classes, mask):
        raise NotImplementedError("a network score is not linear in the signals; use prepare()")

    def forward_pooled(self, sums, counts, size_centre=0.0):
        raise NotImplementedError("a network score is not linear in the signals; use predict_prepared()")

    def prepare(self, x: torch.Tensor, classes: torch.Tensor, mask: torch.Tensor) -> tuple:
        """-> the distinct images of the batch (signals, class) and how often each subset holds
        each of them [subsets, images]. An image is scored once per epoch however many subsets it
        is in; a subset's sum of weights is its row of the membership matrix times the weights."""
        rows = torch.cat((x[mask], classes[mask].unsqueeze(-1).to(x.dtype)), dim=1)
        distinct, inverse = torch.unique(rows, dim=0, return_inverse=True)
        subset_of = torch.arange(x.shape[0]).unsqueeze(1).expand_as(mask)[mask]
        membership = torch.zeros(x.shape[0], distinct.shape[0], dtype=x.dtype)
        membership.index_put_((subset_of, inverse), torch.ones(len(inverse), dtype=x.dtype), accumulate=True)
        return distinct[:, :-1].contiguous(), distinct[:, -1].to(torch.long), membership

    def predict_prepared(self, prepared: tuple, size_centre: float = 0.0) -> torch.Tensor:
        x, classes, membership = prepared
        total = membership @ self.image_weight(x, classes)
        return self.intercept + self.log_count * (torch.log1p(torch.relu(total)) - size_centre)

    @staticmethod
    def prepared_sizes(prepared: tuple) -> torch.Tensor:
        return prepared[2].sum(1)

    @staticmethod
    def resample(prepared: tuple, rows: torch.Tensor) -> tuple:
        x, classes, membership = prepared
        return x, classes, membership[rows]

    def restrict_signals(self, mask: tuple[bool, ...]) -> None:
        """Hold the unflagged signals at zero on the way in. Each class keeps its own output."""
        with torch.no_grad():
            self.signal_keep.copy_(torch.tensor(mask, dtype=torch.float32))


def new_ranker(architecture: str, n_classes: int, hidden_units: int = HIDDEN_UNITS) -> AdditiveUtilityRanker:
    if architecture == "linear":
        return AdditiveUtilityRanker(n_classes)
    if architecture == "mlp":
        return NetworkUtilityRanker(n_classes, hidden_units=hidden_units)
    raise ValueError(f"Unknown ranker architecture {architecture!r}; expected one of {ARCHITECTURES}")


@dataclass
class FittedRanking:
    model: AdditiveUtilityRanker          # or its subclass NetworkUtilityRanker
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
               patience: int = 60, learning_rate: float = 0.03, bootstrap: int = 0,
               weight_decay: float = 1e-5, initial_lam: float = SIZE_TERM_INITIAL,
               signal_mask: tuple[bool, ...] | None = None,
               standardise_targets: bool = False, architecture: str = "linear",
               hidden_units: int = HIDDEN_UNITS) -> FittedRanking:
    """Train on measured subset outcomes; tune on image-disjoint validation.

    The protected test subsets are never inspected in fitting or early stopping.
    No image-level teacher targets are distilled, so no OOF teacher is needed.
    The real instrument must pass a separately frozen reliability/validity gate.

    `bootstrap` > 0 also fits that many models, each on a resample (with replacement) of the
    TRAIN subsets with the same configuration and early stopping on the same validation subsets.
    Their spread gives the confidence bound the contract §10 stopping rule needs. Sizes may differ
    between subsets (contract §8): the size term is identifiable only then, and the stopping rule
    refuses a model whose size term was frozen.

    `signal_mask` (one flag per SIGNALS entry) holds the unflagged signals out of the score for
    every class. All False is the size-and-class-only model the acceptance check compares the
    ranker with; one True is a single-signal baseline. None uses all four.

    `architecture` is the per-image score: "mlp" is the network of the contract §9 amendment of
    2026-10-04 (one hidden layer of `hidden_units`), "linear" the class-specific linear score it
    replaced, kept as a baseline. The approved value comes from prereg.fit_arguments.

    `standardise_targets` fits on (target - train mean) / train SD with the size term centred on
    its train mean, then writes the intercept and lam back in the metric's own units. The model
    and its predictions are the same; only the optimiser's problem is better conditioned. Without
    it, targets that move by about 0.001 leave gradients of the order of the weight decay and an
    intercept nearly collinear with the size term (design check 2026-10-03 §10, amendment in §11).
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
    if (max_epochs < 1 or bootstrap < 0 or patience < 1 or learning_rate <= 0 or weight_decay < 0
            or initial_lam == 0):
        raise ValueError("Invalid training configuration")
    mask = tuple(bool(v) for v in (signal_mask if signal_mask is not None else (True,) * len(SIGNALS)))
    if len(mask) != len(SIGNALS):
        raise ValueError("signal_mask needs one flag per signal")
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
    config = (len(classes), tuple(frozen), max_epochs, patience, learning_rate, weight_decay,
              initial_lam, mask, bool(standardise_targets), architecture, int(hidden_units))
    pooler = new_ranker(architecture, len(classes), int(hidden_units))
    pooled = {role: pooler.prepare(*tensors[role]) for role in ("train", "validation")}
    model, best, best_epoch, at_limit = _train(pooled["train"], targets["train"], pooled["validation"],
                                               targets["validation"], seed, *config)
    ensemble = []
    members_at_limit = 0
    n_train = len(targets["train"])
    for b in range(bootstrap):
        rows = torch.from_numpy(np.random.default_rng([seed, b]).integers(0, n_train, n_train))
        resampled = pooler.resample(pooled["train"], rows)
        member, _, _, limit = _train(resampled, targets["train"][rows], pooled["validation"],
                                     targets["validation"], seed + 1 + b, *config)
        members_at_limit += int(limit)
        ensemble.append(member)
    return FittedRanking(model, normalizer, classes,
                         {"best_epoch": best_epoch, "validation_mse": best,
                          "train_subsets": n_train,
                          "validation_subsets": len(targets["validation"]),
                          "epoch_limit_reached": at_limit,
                          "bootstrap_models_at_epoch_limit": members_at_limit,
                          "subset_sizes": sorted(sizes),
                          "training_seed": seed, "instrument_decision": decision,
                          "frozen_unidentifiable": frozen, "bootstrap_models": bootstrap,
                          "signals_used": [s for s, used in zip(SIGNALS, mask) if used],
                          "architecture": ({"kind": "mlp", "hidden_units": int(hidden_units), "activation": "tanh",
                                            "outputs": "one_per_class"}
                                           if architecture == "mlp" else {"kind": "linear"}),
                          "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                          "fit_config": {"max_epochs": max_epochs, "patience": patience,
                                         "learning_rate": learning_rate, "weight_decay": weight_decay,
                                         "initial_lam": initial_lam, "optimizer": "adam, full batch",
                                         "standardise_targets": bool(standardise_targets)}},
                         frozenset(memberships["train"]), frozenset(memberships["validation"]),
                         tuple(ensemble))


def _train(train, train_y, validation, validation_y, seed, n_classes, frozen, max_epochs,
           patience, learning_rate, weight_decay, initial_lam, mask, standardise,
           architecture="linear", hidden_units=HIDDEN_UNITS):
    """One fit with validation early stopping, on prepared subsets (the ranker's prepare()).
    Frozen parameters keep their initial value. -> model, validation MSE in the metric's units,
    best epoch, and whether the epoch limit ended the fit instead of the patience."""
    torch.manual_seed(seed)
    model = new_ranker(architecture, n_classes, hidden_units)
    mean, sd, centre = 0.0, 1.0, 0.0
    if standardise:
        mean, sd = float(train_y.mean()), float(train_y.std())
        if not (math.isfinite(sd) and sd > 0):
            raise ValueError("Train targets do not vary; nothing to fit")
        centre = float(torch.log1p(model.prepared_sizes(train)).mean())
        train_y, validation_y = (train_y - mean) / sd, (validation_y - mean) / sd
    with torch.no_grad():
        model.log_count.fill_(initial_lam / sd)
        if standardise:
            model.intercept.fill_(0.0)
    if not all(mask):
        model.restrict_signals(mask)
    for name in frozen:
        getattr(model, name).requires_grad_(False)
    if "log_count" in frozen:
        with torch.no_grad():
            model.log_count.fill_(SIZE_TERM_WHEN_FROZEN)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=learning_rate, weight_decay=weight_decay)
    best, best_state, best_epoch, at_limit = float("inf"), None, -1, True
    for epoch in range(max_epochs):
        model.train()
        loss = torch.nn.functional.mse_loss(model.predict_prepared(train, centre), train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val = float(torch.nn.functional.mse_loss(model.predict_prepared(validation, centre), validation_y))
        if math.isfinite(val) and val < best - 1e-9:
            best, best_epoch = val, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            at_limit = False
            break
    if best_state is None:
        raise ValueError("No finite validation fit")
    model.load_state_dict(best_state)
    with torch.no_grad():                    # back to the metric's own units, size term uncentred
        model.intercept.copy_((model.intercept - model.log_count * centre) * sd + mean)
        model.log_count.mul_(sd)
    model.eval()
    return model, best * sd * sd, best_epoch, at_limit
