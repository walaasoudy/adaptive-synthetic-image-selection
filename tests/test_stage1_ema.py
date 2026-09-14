#!/usr/bin/env python3
"""Stage 1 LoRA EMA: warmup schedule, dilution without warmup, and resume round-trip."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train.train_lora_sdxl import LoraEMA  # noqa: E402


class TinyAdapter(torch.nn.Module):
    """Stand-in for a LoRA-wrapped UNet: one trainable (zero-init like lora_B) and one frozen tensor."""

    def __init__(self):
        super().__init__()
        self.lora_B = torch.nn.Parameter(torch.zeros(4))
        self.frozen = torch.nn.Parameter(torch.ones(4), requires_grad=False)


def _train_to_constant(ema: LoraEMA, model: TinyAdapter, steps: int, value: float = 1.0) -> None:
    with torch.no_grad():
        model.lora_B.fill_(value)
    for _ in range(steps):
        ema.update(model)


def test_shadow_tracks_trainable_params_only():
    ema = LoraEMA(TinyAdapter(), decay=0.9999)
    assert set(ema.shadow) == {"lora_B"}


def test_warmup_first_update_copies_weights():
    model = TinyAdapter()
    ema = LoraEMA(model, decay=0.9999, warmup=True)
    _train_to_constant(ema, model, steps=1, value=3.0)
    assert ema.current_decay() == 0.0
    assert torch.allclose(ema.shadow["lora_B"], torch.full((4,), 3.0))


def test_warmup_decay_schedule_matches_diffusers_formula():
    model = TinyAdapter()
    ema = LoraEMA(model, decay=0.9999, warmup=True)
    for update in range(1, 3001):
        ema.update(model)
        step = update - 1
        expected = 0.0 if step <= 0 else min(0.9999, (1 + step) / (10 + step))
        assert abs(ema.current_decay() - expected) < 1e-12, (update, ema.current_decay(), expected)


def test_warmup_decay_is_capped_at_configured_decay():
    model = TinyAdapter()
    ema = LoraEMA(model, decay=0.9, warmup=True)
    for _ in range(500):
        ema.update(model)
    assert ema.current_decay() == 0.9


def test_warmup_final_shadow_reflects_training_after_2000_steps():
    model = TinyAdapter()
    ema = LoraEMA(model, decay=0.9999, warmup=True)
    _train_to_constant(ema, model, steps=2000)
    assert float(ema.shadow["lora_B"].min()) > 0.99


def test_fixed_decay_dilutes_toward_zero_init():
    """Documents the bug the warmup fixes: 0.9999**2000 of the no-op init survives."""
    model = TinyAdapter()
    ema = LoraEMA(model, decay=0.9999, warmup=False)
    _train_to_constant(ema, model, steps=2000)
    expected = 1.0 - 0.9999 ** 2000  # ~0.181
    assert abs(float(ema.shadow["lora_B"][0]) - expected) < 1e-4
    assert ema.current_decay() == 0.9999


def test_resume_matches_uninterrupted_run():
    model = TinyAdapter()
    uninterrupted = LoraEMA(model, decay=0.9999, warmup=True)
    _train_to_constant(uninterrupted, model, steps=700)

    model_a = TinyAdapter()
    first_leg = LoraEMA(model_a, decay=0.9999, warmup=True)
    _train_to_constant(first_leg, model_a, steps=300)
    state = first_leg.state_dict()

    resumed = LoraEMA(TinyAdapter(), decay=0.9999, warmup=True)
    resumed.load_state_dict(state)
    assert resumed.num_updates == 300
    _train_to_constant(resumed, model_a, steps=400)

    assert resumed.num_updates == uninterrupted.num_updates == 700
    assert torch.allclose(resumed.shadow["lora_B"], uninterrupted.shadow["lora_B"], atol=1e-6)


def test_resume_refuses_legacy_state_without_num_updates():
    legacy = {"decay": 0.9999, "shadow": {"lora_B": torch.zeros(4)}}
    ema = LoraEMA(TinyAdapter(), decay=0.9999, warmup=True)
    try:
        ema.load_state_dict(legacy)
    except ValueError as exc:
        assert "warmup" in str(exc)
    else:
        raise AssertionError("a legacy fixed-decay EMA state must not resume into a warmup run")


def test_resume_refuses_schedule_change():
    state = LoraEMA(TinyAdapter(), decay=0.9999, warmup=False).state_dict()
    ema = LoraEMA(TinyAdapter(), decay=0.9999, warmup=True)
    try:
        ema.load_state_dict(state)
    except ValueError as exc:
        assert "warmup" in str(exc)
    else:
        raise AssertionError("resuming must refuse a different EMA schedule")


def test_fixed_decay_still_resumes_legacy_state():
    legacy = {"decay": 0.9999, "shadow": {"lora_B": torch.full((4,), 0.5)}}
    ema = LoraEMA(TinyAdapter(), decay=0.9999, warmup=False)
    ema.load_state_dict(legacy)
    assert torch.allclose(ema.shadow["lora_B"], torch.full((4,), 0.5))


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
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    raise SystemExit(1 if failed else 0)
