"""HAM10000 generator data roles and native-aspect LoRA inputs — shared by input preparation
(scripts/data/ham10000/04_prepare_lora_inputs.py) and training (scripts/train/ham10000_train_lora_sdxl.py).

One module so the preparation step and the training step cannot disagree about which splits the
generator may see or what a valid training image is.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ALL_SPLITS = ("gen_train", "gen_val", "classifier_train", "classifier_val", "asism_tuning_heldout", "final_eval_heldout")

# The generator learns from gen_train and is monitored on gen_val. Nothing else, ever.
REQUIRED_ROLES = {"train_split": "gen_train", "monitor_split": "gen_val"}


class LoraRoleError(SystemExit):
    """A LoRA data role points at, or contains images from, a split the generator must not see."""


def validate_roles(split_cfg) -> dict[str, str]:
    """The LoRA data roles, asserted to be exactly gen_train / gen_val."""
    roles = {}
    for key, expected in REQUIRED_ROLES.items():
        value = str(split_cfg.get(key))
        if value != expected:
            raise LoraRoleError(f"split.{key} must be {expected!r} for HAM10000 LoRA training; got {value!r}")
        roles[key] = value
    return roles


def forbidden_image_ids(split_dir: Path, allowed_splits) -> dict[str, frozenset[str]]:
    """image ids of every split the generator must never see, keyed by split name. Ids only."""
    allowed = set(allowed_splits)
    forbidden = {}
    for name in ALL_SPLITS:
        if name in allowed:
            continue
        path = Path(split_dir) / f"{name}.csv"
        if not path.is_file():
            raise LoraRoleError(f"cannot prove isolation from {name}: {path} is missing")
        forbidden[name] = frozenset(pd.read_csv(path, usecols=["image_id"])["image_id"].astype(str))
    return forbidden


def assert_role_isolation(image_ids, split_name: str, forbidden: dict[str, frozenset[str]]) -> None:
    """Refuse if any id also belongs to a forbidden split — a content check, not a name check."""
    ids = {str(item) for item in image_ids}
    for name, other in forbidden.items():
        leaked = sorted(ids & other)
        if leaked:
            raise LoraRoleError(f"{len(leaked)} image(s) of {split_name} also belong to {name} (e.g. {leaked[:3]}); refusing")


def native_aspect_resize(image, size):
    """Aspect-preserving resize to exactly `size`, with NO crop and NO padding.

    Refuses a source whose aspect ratio differs from the target's: fitting it would need cropping
    (loses lesion content) or padding (teaches the generator bars), both excluded by the geometry
    contract in configs/ham10000_stage1.yaml.
    """
    from PIL import Image

    width, height = image.size
    target_width, target_height = int(size[0]), int(size[1])
    if width * target_height != height * target_width:
        raise ValueError(f"source {width}x{height} is not the contracted aspect ratio {target_width}:{target_height}")
    return image.convert("RGB").resize((target_width, target_height), Image.Resampling.BICUBIC)


def validate_rgb_jpeg(path: Path, size) -> tuple[bool, str | None]:
    """Fully decode a JPEG and require RGB at exactly `size` (width, height). Existence is never enough."""
    from PIL import Image

    path = Path(path)
    if not path.is_file():
        return False, "missing"
    try:
        with Image.open(path) as image:
            if image.format != "JPEG":
                return False, f"format_{image.format}"
            if image.mode != "RGB":
                return False, f"mode_{image.mode}"
            if image.size != (int(size[0]), int(size[1])):
                return False, f"size_{image.size[0]}x{image.size[1]}"
            image.load()
    except Exception as exc:
        return False, f"unreadable_{type(exc).__name__}"
    return True, None


def atomic_save_rgb_jpeg(image, destination: Path, size, quality: int) -> None:
    """Write a JPEG via temp file + os.replace, validated at its REAL (width, height).

    Not 02_preprocess_images.atomic_save_jpeg: that one validates a square canvas and rejects every
    native-aspect image.
    """
    import os
    import tempfile

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        image.save(temporary, format="JPEG", quality=int(quality))
        valid, reason = validate_rgb_jpeg(temporary, size)
        if not valid:
            raise OSError(f"temporary JPEG validation failed: {reason}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "atomic_save_rgb_jpeg",
    "ALL_SPLITS",
    "REQUIRED_ROLES",
    "LoraRoleError",
    "validate_roles",
    "forbidden_image_ids",
    "assert_role_isolation",
    "native_aspect_resize",
    "validate_rgb_jpeg",
]
