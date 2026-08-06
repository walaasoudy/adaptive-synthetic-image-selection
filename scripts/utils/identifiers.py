"""Shared image-id derivation, used identically by preprocessing (03) and caption generation (04)
so the two stay joinable on image_id."""

from __future__ import annotations

import re


def sanitize_image_id(raw_path: str) -> str:
    p = str(raw_path).replace("\\", "/")
    for prefix in ("CheXpert-v1.0-small/", "CheXpert-v1.0/"):
        if p.startswith(prefix):
            p = p[len(prefix):]
    p = re.sub(r"\.(jpg|jpeg|png)$", "", p, flags=re.IGNORECASE)
    return p.replace("/", "__")
