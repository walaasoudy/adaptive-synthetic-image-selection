#!/usr/bin/env python3
"""Stage 1 validation: seeded gen_val subset, repeatable loss, untouched training RNG, no autograd."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train.train_lora_sdxl import run_validation, select_subset_indices  # noqa: E402

SEED = 2024


class TinyUNet(torch.nn.Module):
    """Stand-in for the LoRA UNet: one trainable weight, records whether autograd was enabled."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.grad_enabled_calls: list[bool] = []

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs, return_dict):
        self.grad_enabled_calls.append(torch.is_grad_enabled())
        return (noisy_latents * self.scale,)


class TinyScheduler:
    config = SimpleNamespace(num_train_timesteps=1000, prediction_type="epsilon")

    def add_noise(self, latents, noise, timesteps):
        return latents + noise * (timesteps.float() / 1000).view(-1, 1, 1, 1)


class RandomLatentDataset(torch.utils.data.Dataset):
    """Draws its latent from the global RNG per access, like CachedSDXLDataset does."""

    def __len__(self):
        return 6

    def __getitem__(self, index):
        return {
            "latents": torch.randn(4, 8, 8),
            "pooled_embeds": torch.zeros(2),
            "time_ids": torch.zeros(6),
            "prompt_embeds": torch.zeros(3, 2),
        }


def _validate(unet: TinyUNet, seed: int = SEED) -> float:
    loader = torch.utils.data.DataLoader(RandomLatentDataset(), batch_size=2, shuffle=False, num_workers=0)
    accelerator = SimpleNamespace(device=torch.device("cpu"))
    return run_validation(accelerator, unet, loader, TinyScheduler(), torch.float32, seed)


def test_repeated_validation_gives_identical_loss():
    unet = TinyUNet()
    first = _validate(unet)
    torch.randn(100)  # training consumes RNG between validation passes
    assert _validate(unet) == first


def test_different_seed_changes_the_draws():
    unet = TinyUNet()
    assert _validate(unet, seed=SEED) != _validate(unet, seed=SEED + 1)


def test_validation_leaves_training_rng_untouched():
    torch.manual_seed(7)
    expected = torch.randn(5)
    torch.manual_seed(7)
    _validate(TinyUNet())
    assert torch.equal(torch.randn(5), expected)


def test_validation_builds_no_autograd_graph():
    unet = TinyUNet()
    _validate(unet)
    assert unet.grad_enabled_calls and not any(unet.grad_enabled_calls)
    assert torch.is_grad_enabled()


def test_subset_is_seeded_sorted_and_sized():
    first = select_subset_indices(19000, 1000, SEED)
    assert first == select_subset_indices(19000, 1000, SEED)
    assert len(first) == 1000 and len(set(first)) == 1000
    assert first == sorted(first) and 0 <= first[0] and first[-1] < 19000
    assert first != select_subset_indices(19000, 1000, SEED + 1)


def test_subset_ignores_global_rng_state():
    torch.manual_seed(1)
    first = select_subset_indices(500, 50, SEED)
    torch.manual_seed(2)
    assert select_subset_indices(500, 50, SEED) == first


def test_subset_none_or_oversized_keeps_every_record():
    assert select_subset_indices(300, None, SEED) is None
    assert select_subset_indices(300, 300, SEED) is None
    assert select_subset_indices(300, 5000, SEED) is None


def test_subset_rejects_non_positive_size():
    for bad in (0, -1):
        try:
            select_subset_indices(300, bad, SEED)
        except ValueError:
            continue
        raise AssertionError(f"max_val_samples={bad} was accepted")


if __name__ == "__main__":
    import traceback

    tests = [(name, value) for name, value in sorted(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for name, function in tests:
        try:
            function()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
    raise SystemExit(1 if failed else 0)
