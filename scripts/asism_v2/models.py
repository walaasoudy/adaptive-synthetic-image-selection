"""CPU-testable V2 architecture candidate, NOT a validated scientific selector.

No V1 checkpoints are compatible. Raw marginal utility must not receive the V1
(n-1) mean-pooling normalization. Fixed-size swap contrasts are the intended
selection target; this module does not fit or approve that target.
"""
import torch
from torch import nn


class SizeAwareSetUtility(nn.Module):
    def __init__(self, input_dim: int, width: int = 16):
        super().__init__()
        if input_dim < 1 or width < 1:
            raise ValueError("Positive dimensions required")
        self.encoder = nn.Sequential(nn.Linear(input_dim, width), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(2 * width + 1, width), nn.Tanh(), nn.Linear(width, 1))

    def encode_set(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or mask.shape != features.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("Expected features [batch, members, features] and Boolean mask")
        if not mask.any(dim=1).all() or not torch.isfinite(features[mask]).all():
            raise ValueError("Empty sets or nonfinite active features")
        # Sanitize BEFORE the encoder: NaN padding must not poison backward gradients.
        safe = features.masked_fill(~mask.unsqueeze(-1), 0)
        encoded = self.encoder(safe)
        weights = mask.unsqueeze(-1).to(encoded.dtype)
        count = weights.sum(1)
        mean = (encoded * weights).sum(1) / count
        variance = ((encoded - mean.unsqueeze(1)).square() * weights).sum(1) / count
        return torch.cat((mean, variance, torch.log1p(count)), dim=1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode_set(features, mask)).squeeze(-1)
