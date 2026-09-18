"""Letterbox geometry for HAM10000: the ONE place the padded-canvas layout is computed.

WHY THIS IS ITS OWN MODULE
  Preprocessing places the source image on a padded square; explainability needs to know exactly
  which part of that square is real image and which part is padding this pipeline added. If those
  two stages each computed the layout, a later change to one (a different rounding, an offset
  convention) would silently desynchronise them and `peripheral_mass` would measure the wrong
  region with no error anywhere. Both import `letterbox_geometry` from here instead, and the box is
  additionally PERSISTED per image at preprocessing time, so explainability reads the box that was
  actually used rather than recomputing one.

CONVENTION
  content_box = (x0, y0, x1, y1), normalised to the output canvas, half-open: the real image
  occupies pixel columns [x0*S, x1*S) and rows [y0*S, y1*S) of an S x S canvas. Because the offsets
  and resized extents are integers, these fractions are exact multiples of 1/S — nothing here is an
  estimate from pixel values.
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

CONTENT_BOX_COLUMNS = ["image_id", "source_width", "source_height", "x0", "y0", "x1", "y1"]


def letterbox_geometry(width: int, height: int, target_size: int) -> dict:
    """Exact layout of an aspect-preserving resize of (width, height) onto a target_size square."""
    if width <= 0 or height <= 0 or target_size <= 0:
        raise ValueError(f"invalid geometry: {width}x{height} -> {target_size}")
    scale = target_size / max(width, height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    offset_x = (target_size - new_width) // 2
    offset_y = (target_size - new_height) // 2
    return {
        "resized_size": (new_width, new_height),
        "offset": (offset_x, offset_y),
        "content_box": (
            offset_x / target_size,
            offset_y / target_size,
            (offset_x + new_width) / target_size,
            (offset_y + new_height) / target_size,
        ),
    }


def letterbox_with_content_box(image, image_id: str, target_size: int, pad_colour) -> tuple:
    """THE standardisation step for every image the classifier or ASISM ever scores — real or
    synthetic. Returns (RGB canvas, content-box row) from one geometry computation, so the box
    written next to an image is by construction the layout applied to it.
    """
    from PIL import Image

    rgb = image.convert("RGB")
    width, height = rgb.size
    geometry = letterbox_geometry(width, height, target_size)
    canvas = Image.new("RGB", (target_size, target_size), tuple(int(v) for v in pad_colour))
    canvas.paste(rgb.resize(geometry["resized_size"], Image.Resampling.BICUBIC), geometry["offset"])
    x0, y0, x1, y1 = geometry["content_box"]
    row = {"image_id": str(image_id), "source_width": int(width), "source_height": int(height), "x0": x0, "y0": y0, "x1": x1, "y1": y1}
    return canvas, row


class UnknownPaddingError(ValueError):
    """A generated image arrived with borders the pipeline did not add, or at an unexpected size."""


def edge_strip_uniformity(image, strip_fraction: float = 0.04) -> dict[str, float]:
    """Pixel std of each outer strip (top/bottom/left/right) of an RGB image."""
    import numpy as np

    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = array.shape[:2]
    band_y, band_x = max(1, round(height * strip_fraction)), max(1, round(width * strip_fraction))
    strips = {"top": array[:band_y], "bottom": array[-band_y:], "left": array[:, :band_x], "right": array[:, -band_x:]}
    return {name: float(strip.std()) for name, strip in strips.items()}


def standardize_generated_image(
    image,
    image_id: str,
    generation_size: tuple[int, int],
    target_size: int,
    pad_colour,
    min_edge_std: float = 2.0,
) -> tuple:
    """Synthetic-image entry into the SAME geometry contract real images use.

    Contract: Stage 2 generates at `generation_size`, the native HAM10000 aspect ratio, with NO
    padding. This function then applies `letterbox_with_content_box` exactly as preprocessing does
    for real images, so a synthetic candidate gets an exact content box from the same code path.

    Nothing is guessed. If the image is not at the contracted size, or any outer strip is flat
    (std < min_edge_std) — i.e. the generator drew bars/padding of its own whose extent is unknown —
    it is REFUSED with UnknownPaddingError rather than assigned an estimated box or the generic band.
    Uniformity is used only to refuse, never to infer a layout.
    """
    expected = (int(generation_size[0]), int(generation_size[1]))
    if tuple(image.size) != expected:
        raise UnknownPaddingError(f"{image_id}: generated at {image.size[0]}x{image.size[1]}, contract is {expected[0]}x{expected[1]}")
    flat = {name: std for name, std in edge_strip_uniformity(image).items() if std < min_edge_std}
    if flat:
        raise UnknownPaddingError(f"{image_id}: flat border strip(s) {sorted(flat)} — generator-drawn padding of unknown extent")
    return letterbox_with_content_box(image, image_id, target_size, pad_colour)


def validate_content_box(box) -> tuple[float, float, float, float]:
    """Return the box as a float 4-tuple, or raise if it is not a proper normalised rectangle."""
    if box is None or len(box) != 4:
        raise ValueError(f"content_box must be (x0, y0, x1, y1), got {box!r}")
    x0, y0, x1, y1 = (float(value) for value in box)
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"content_box must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, got {box!r}")
    return (x0, y0, x1, y1)


def write_content_boxes(path: Path, rows: list[dict]) -> None:
    """Atomically write the per-image content-box table (CONTENT_BOX_COLUMNS)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CONTENT_BOX_COLUMNS)
            writer.writeheader()
            for row in sorted(rows, key=lambda item: str(item["image_id"])):
                writer.writerow({column: row[column] for column in CONTENT_BOX_COLUMNS})
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_content_boxes(path: Path) -> dict[str, tuple[float, float, float, float]]:
    """image_id -> validated content_box, from a table written by write_content_boxes.

    Fails loudly on a missing file or a duplicate image_id: a box table that cannot be trusted must
    not be quietly replaced by the generic border band.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"content-box table not found: {path}")
    boxes: dict[str, tuple[float, float, float, float]] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = str(row["image_id"])
            if image_id in boxes:
                raise ValueError(f"duplicate image_id {image_id!r} in {path}")
            boxes[image_id] = validate_content_box((row["x0"], row["y0"], row["x1"], row["y1"]))
    return boxes


__all__ = [
    "CONTENT_BOX_COLUMNS",
    "letterbox_geometry",
    "letterbox_with_content_box",
    "UnknownPaddingError",
    "edge_strip_uniformity",
    "standardize_generated_image",
    "validate_content_box",
    "write_content_boxes",
    "load_content_boxes",
]
