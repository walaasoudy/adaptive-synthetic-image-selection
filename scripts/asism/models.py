"""Trainable components for learned ASISM.

The set model learns only from measured subset-level downstream utility.  Image scores are then
distilled from marginal set utility; no subset AUROC is copied onto every member image.
"""
from __future__ import annotations

import torch
from torch import nn


def mlp(dimensions: list[int], dropout: float = 0.0, final_sigmoid: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (source, target) in enumerate(zip(dimensions, dimensions[1:])):
        last = index == len(dimensions) - 2
        layers.append(nn.Linear(source, target))
        if not last:
            layers.extend([nn.LayerNorm(target), nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
    if final_sigmoid:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class SetUtilityNetwork(nn.Module):
    """Permutation-invariant Deep Sets predictor for measured AUROC delta.

    Default: masked MEAN pooling of per-image encodings (the frozen pipeline; the ``(n-1)``
    size-normalization in ``05_train_learned_asism.py::marginal_targets`` is derived from it).

    ``superset_conditioning=True`` (ablation only, Phase-11): the set is summarised by masked mean
    AND std pooling — both order-invariant — and a fixed summary vector of the full candidate pool
    is concatenated to the utility-head input. This follows Xie et al., ICLR 2024 ("Enhancing Neural
    Subset Selection"): a set-utility target that depends on the ground set benefits from a richer
    permutation-invariant sufficient statistic than the mean alone. NOTE: with std pooling the
    marginal-target size-normalization is only first-order correct for the mean component, so this
    flag is evaluated as an ablation, never silently swapped into the frozen path.
    """

    def __init__(self, input_dim: int, image_hidden=(128, 64), utility_hidden=(32,),
                 superset_conditioning: bool = False):
        super().__init__()
        self.superset_conditioning = bool(superset_conditioning)
        self.image_encoder = mlp([input_dim, *image_hidden], dropout=0.2)
        pooled_dim = image_hidden[-1] * (2 if self.superset_conditioning else 1)
        head_input = pooled_dim + (2 * input_dim if self.superset_conditioning else 0)
        self.utility_head = mlp([head_input, *utility_hidden, 1])
        if self.superset_conditioning:
            self.register_buffer("superset_summary", torch.zeros(2 * input_dim))

    def set_superset_summary(self, summary) -> None:
        if not self.superset_conditioning:
            raise ValueError("set_superset_summary called but superset_conditioning is False")
        vector = torch.as_tensor(summary, dtype=self.superset_summary.dtype,
                                 device=self.superset_summary.device)
        if vector.shape != self.superset_summary.shape:
            raise ValueError(
                f"superset summary must have shape {tuple(self.superset_summary.shape)}, "
                f"got {tuple(vector.shape)}"
            )
        self.superset_summary.copy_(vector)

    def encode_set(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.image_encoder(features)
        weights = mask.unsqueeze(-1).to(encoded.dtype)
        counts = weights.sum(1).clamp_min(1.0)
        mean = (encoded * weights).sum(1) / counts
        if not self.superset_conditioning:
            return mean
        variance = (weights * (encoded - mean.unsqueeze(1)) ** 2).sum(1) / counts
        return torch.cat([mean, variance.clamp_min(0.0).sqrt()], dim=-1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = self.encode_set(features, mask)
        if self.superset_conditioning:
            summary = self.superset_summary.unsqueeze(0).expand(pooled.shape[0], -1)
            pooled = torch.cat([pooled, summary], dim=-1)
        return self.utility_head(pooled).squeeze(-1)


class MultiSignalUtilityRankingNetwork(nn.Module):
    """Small MLP that fuses the Go/No-Go-admitted ASISM signals (similarity, IQA, uncertainty,
    explainability, agreement, distinctiveness) into ONE scalar image-utility score.

    Deliberately NOT a multi-objective model: there is a single regression/ranking target (the
    distilled marginal set-utility from SetUtilityNetwork), optimized by both a Smooth-L1 term and a
    pairwise ranking term. MMoE / SDMGrad / NHDE are cited in the literature review only to make that
    contrast explicit — none of their multi-task/Pareto machinery is used here.

    ``input_dim`` is the number of Go/No-Go-admitted signal FEATURE COLUMNS, not the number of
    signals: some signals contribute several columns. With the default ``learned_asism.feature_columns``
    that is nine (similarity x3, IQA x3, uncertainty, explainability, agreement; distinctiveness's
    column is not yet wired into the learned feature set). Hidden layers are 128 -> 64 -> 32 -> 1.
    """

    def __init__(self, input_dim: int, hidden=(128, 64, 32), dropout=0.2):
        super().__init__()
        self.network = mlp([input_dim, *hidden, 1], dropout=dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class AdaptiveThresholdNetwork(nn.Module):
    """Produces a [0,1] class-aware threshold from class identity and class context."""

    def __init__(self, number_of_classes: int, context_dim: int, embedding_dim=16,
                 hidden=(64, 32), dropout=0.1):
        super().__init__()
        self.class_embedding = nn.Embedding(number_of_classes, embedding_dim)
        self.network = mlp([context_dim + embedding_dim, *hidden, 1], dropout, final_sigmoid=True)

    def forward(self, class_ids: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([self.class_embedding(class_ids), context], dim=-1)).squeeze(-1)


def soft_selection_gate(scores: torch.Tensor, thresholds: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.sigmoid((scores - thresholds) / temperature)


def pairwise_ranking_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    delta_target = targets[:, None] - targets[None, :]
    valid = delta_target.abs() > 1e-12
    if not valid.any():
        return predictions.sum() * 0.0
    signs = delta_target.sign()[valid]
    delta_prediction = (predictions[:, None] - predictions[None, :])[valid]
    return torch.nn.functional.softplus(-signs * delta_prediction).mean()
