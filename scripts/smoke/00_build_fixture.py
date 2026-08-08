#!/usr/bin/env python3
"""Create the deterministic synthetic CheXpert-shaped input used only by dev-smoke-v1."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.labels import CLASSIFIER_TARGET_LABELS


def main() -> int:
    root = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
    if root.name != "dev-smoke-v1":
        raise SystemExit(f"Smoke fixture PROJECT_ROOT must end in dev-smoke-v1, got {root}")
    raw = root / "data" / "chexpert" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(60):
        patient = f"patient{index:05d}"
        relative = Path("CheXpert-v1.0-small") / "train" / patient / "study1" / "view1_frontal.jpg"
        # Runtime resolvers strip the CheXpert-v1.0-small prefix before joining raw_dir.
        destination = raw / Path(*relative.parts[1:])
        destination.parent.mkdir(parents=True, exist_ok=True)
        yy, xx = np.mgrid[:256, :256]
        array = np.clip(25 + (xx + yy) * 0.25 + (index % 7) * 4, 0, 255).astype(np.uint8)
        Image.fromarray(array, mode="L").save(destination, quality=92)
        row = {"Path": relative.as_posix(), "Sex": "Male" if index % 2 else "Female",
               "Age": 35 + index % 45, "Frontal/Lateral": "Frontal", "AP/PA": "PA"}
        for label_index, label in enumerate(CLASSIFIER_TARGET_LABELS):
            row[label] = int((index + label_index) % 3 == 0)
        row["No Finding"] = int(not any(row[label] for label in CLASSIFIER_TARGET_LABELS[1:13]))
        rows.append(row)
    pd.DataFrame(rows).to_csv(raw / "train.csv", index=False)
    pd.DataFrame(rows[:4]).to_csv(raw / "valid.csv", index=False)
    marker = root / "SMOKE_ONLY.json"
    marker.write_text('{"namespace":"dev-smoke-v1","scientific_evidence":false}\n', encoding="utf-8")
    print(f"Smoke fixture ready: {root} (60 patients; never production evidence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
