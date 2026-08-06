#!/usr/bin/env python3
"""Generate a fixed-seed probe-prompt sample grid from a Stage 1 LoRA checkpoint
(docs/stage1_plan.md §9 — qualitative plausibility + informal directional-conditioning check).

The probe prompt set is fixed and regenerated with the same seeds at every checkpoint, so samples
are directly comparable checkpoint-to-checkpoint: one prompt per pathology positive, one no-finding
prompt, and a few multi-label combinations. Prompts are built with scripts/utils/caption_builder.py
— the same module used at training time — so probe conditioning matches train-time phrasing exactly.

Usage:
    python scripts/eval/generate_probe_samples.py --checkpoint checkpoints/stage1_lora_sdxl/<run_id>/final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.caption_builder import CaptionConfig, PATHOLOGY_COLUMNS, build_caption  # noqa: E402
from scripts.utils.config import load_stage1_config  # noqa: E402
from scripts.utils.manifest import write_json  # noqa: E402

DEVICE_COLUMN = "Support Devices"
NO_FINDING_COLUMN = "No Finding"

MULTI_LABEL_COMBOS = [
    ["Cardiomegaly", "Pleural Effusion"],
    ["Atelectasis", "Consolidation"],
    ["Edema", "Pneumonia"],
]


def build_probe_rows(caption_config: CaptionConfig) -> list[dict]:
    """One synthetic label row per single-pathology positive, one No Finding, a few multi-label
    combos. Age/sex are fixed to a neutral reference point so only the finding clause varies."""
    base_row = {"Sex": "Male", "Age": 50, "Frontal/Lateral": "Frontal", "AP/PA": "AP"}
    rows = []

    no_finding_row = dict(base_row, **{NO_FINDING_COLUMN: 1})
    rows.append({"name": "no_finding", "row": no_finding_row})

    for col in PATHOLOGY_COLUMNS:
        if col in (NO_FINDING_COLUMN, DEVICE_COLUMN):
            continue
        row = dict(base_row, **{c: 0 for c in PATHOLOGY_COLUMNS})
        row[col] = 1
        rows.append({"name": col.replace(" ", "_").lower(), "row": row})

    for combo in MULTI_LABEL_COMBOS:
        row = dict(base_row, **{c: 0 for c in PATHOLOGY_COLUMNS})
        for col in combo:
            row[col] = 1
        rows.append({"name": "_".join(c.replace(" ", "").lower() for c in combo), "row": row})

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True, help="LoRA weights dir (e.g. .../lora_weights/step_1000 or .../final)")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    args = parser.parse_args()

    cfg = load_stage1_config()
    caption_config = CaptionConfig(
        age_bucket_width_years=cfg.captions.age_bucket_width_years,
        uncertain_label_policy=cfg.captions.uncertain_label_policy,
        no_finding_overrides_positives=cfg.captions.no_finding_overrides_positives,
        num_paraphrase_variants=cfg.captions.num_paraphrase_variants,
        template_version=cfg.captions.template_version,
    )

    checkpoint_path = Path(args.checkpoint)
    # e.g. ".../lora_weights/step_1000" -> "lora_weights_step_1000", ".../final" -> "final"
    checkpoint_name = "_".join(checkpoint_path.parts[-2:]) if checkpoint_path.parent.name == "lora_weights" else checkpoint_path.name
    out_dir = Path(cfg.paths.outputs_dir) / checkpoint_name
    out_dir.mkdir(parents=True, exist_ok=True)

    from diffusers import StableDiffusionXLPipeline

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        cfg.model.base_model_id, torch_dtype=torch.bfloat16, cache_dir=cfg.paths.hf_cache_dir
    )
    pipeline.load_lora_weights(str(checkpoint_path))
    pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.set_progress_bar_config(disable=False)

    negative_prompt = "text, watermark, labels, color image, cropped anatomy, deformed chest, low quality"
    probe_rows = build_probe_rows(caption_config)

    records = []
    for i, probe in enumerate(probe_rows):
        prompt = build_caption(probe["row"], caption_config, variant_index=0)
        seed = cfg.validation.probe_seed + i
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=cfg.data.resolution,
            width=cfg.data.resolution,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]
        image_path = out_dir / f"{probe['name']}.png"
        image.save(image_path)
        records.append({"name": probe["name"], "prompt": prompt, "seed": seed, "image_path": str(image_path)})
        print(f"[{i + 1}/{len(probe_rows)}] {probe['name']}: {prompt}")

    write_json(out_dir / "probe_manifest.json", {"checkpoint": str(checkpoint_path), "records": records})
    print(f"Wrote {len(records)} probe samples -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
