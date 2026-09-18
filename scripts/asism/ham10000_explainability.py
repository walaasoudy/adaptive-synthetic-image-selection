"""Grad-CAM explainability rows for HAM10000, with the exact letterbox content box wired through.

This is the step that carries `content_box` from the preprocessing table into `peripheral_mass`.
The CheXpert runner (scripts/asism/01_compute_signals.py::run_explainability) scores overlap with
anatomical boxes and is not modified; this is the dermoscopy counterpart.

CONTENT-BOX POLICY
  * `content_boxes` given  -> every record MUST have a box; a missing id raises. Silently falling
    back to the generic border band for some images would mix two different measurements in one
    column.
  * `content_boxes=None`   -> generic border band for every record, and each row says so in
    `explainability_content_box_source`. Such rows can never produce a selection-eligible value
    (see calibrate_explainability).
"""

from __future__ import annotations

import json
from typing import Callable

import numpy as np

from scripts.asism.ham10000_signals import compute_explainability_statistics
from scripts.utils.ham10000_geometry import validate_content_box


def gradcam(model, image_tensor, class_index: int) -> np.ndarray:
    """Grad-CAM for one image and one class on DenseNet's last dense block.

    Same construction as the CheXpert runner (channel weights = spatially averaged gradients of the
    class logit, ReLU of the weighted activation sum), duplicated rather than imported because that
    code is inline in a CheXpert entrypoint. Returns the CAM at feature-map resolution — NOT
    upsampled — so each cell is one piecewise-constant region, which is what content_weights assumes.
    """
    import torch

    activations, gradients = {}, {}
    layer = model.features.denseblock4
    forward = layer.register_forward_hook(lambda _m, _i, output: activations.__setitem__("value", output.detach()))
    backward = layer.register_full_backward_hook(lambda _m, _gi, grad_out: gradients.__setitem__("value", grad_out[0].detach()))
    try:
        model.zero_grad(set_to_none=True)
        image = image_tensor.unsqueeze(0) if image_tensor.dim() == 3 else image_tensor
        logits = model(image)
        logits[0, int(class_index)].backward()
        weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations["value"]).sum(dim=1)).squeeze(0)
        return cam.cpu().numpy()
    finally:
        forward.remove()
        backward.remove()


def explainability_rows(
    records: list[dict],
    cam_for_record: Callable[[dict], np.ndarray],
    content_boxes: dict[str, tuple[float, float, float, float]] | None,
    border_fraction: float = 0.12,
    mass_fraction: float = 0.80,
) -> list[dict]:
    """Raw explainability statistics per record, each measured with that image's own content box.

    `records` need `image_id` and `class_index`; `cam_for_record(record)` returns the CAM. The CAM
    function is injected so the box wiring is testable without a model.
    """
    rows = []
    for record in records:
        image_id = str(record["image_id"])
        box = None
        if content_boxes is not None:
            if image_id not in content_boxes:
                raise KeyError(f"no content_box for {image_id!r}; refusing to fall back to the generic band")
            box = validate_content_box(content_boxes[image_id])
        cam = np.asarray(cam_for_record(record), dtype=np.float64)
        statistics = compute_explainability_statistics(
            cam, border_fraction=border_fraction, mass_fraction=mass_fraction, content_box=box
        )
        rows.append(
            {
                "image_id": image_id,
                "class_index": int(record["class_index"]),
                **statistics,
                "explainability_content_box": json.dumps(list(box)) if box is not None else None,
                "explainability_cam_shape": json.dumps(list(cam.shape)),
                "explainability_is_plausibility_signal_only": True,
            }
        )
    return rows


__all__ = ["gradcam", "explainability_rows"]
