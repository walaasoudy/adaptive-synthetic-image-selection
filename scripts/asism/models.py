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
    """Permutation-invariant Deep Sets predictor for measured AUROC delta."""

    def __init__(self, input_dim: int, image_hidden=(128, 64), utility_hidden=(32,)):
        super().__init__()
        self.image_encoder = mlp([input_dim, *image_hidden], dropout=0.2)
        self.utility_head = mlp([image_hidden[-1], *utility_hidden, 1])

    def encode_set(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.image_encoder(features)
        weights = mask.unsqueeze(-1).to(encoded.dtype)
        return (encoded * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.utility_head(self.encode_set(features, mask)).squeeze(-1)


class MultiObjectiveRankingNetwork(nn.Module):
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
