"""Provenance-checked input loading shared by Learned ASISM stages."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from scripts.utils.artifact_contracts import asism_score_provenance, require_score_artifact, stage2_paths
from scripts.utils.config import load_named_config
from scripts.utils.manifest import read_json


def load_candidate_pool(cfg):
    stage2_cfg = load_named_config("stage2_generation.yaml", "stage2")
    expected = asism_score_provenance(cfg, stage2_cfg, str(cfg.split_namespace))
    gonogo = read_json(Path(cfg.paths.gonogo_report))
    surviving = list(gonogo["surviving_signals"])
    if not surviving:
        raise SystemExit("No signals survived Go/No-Go; Learned ASISM cannot train.")
    merged = None
    for signal in surviving:
        path = Path(cfg.paths.scores_dir) / f"{signal}_scores.parquet"
        require_score_artifact(path, signal, expected)
        frame = pd.read_parquet(path)
        merged = frame if merged is None else merged.merge(frame, on="image_id", how="inner")
    if merged is None or merged.empty:
        raise SystemExit("No non-empty candidate images remain after merging admitted signals.")
    intended = {}
    manifest = stage2_paths(stage2_cfg, str(cfg.split_namespace))["manifest_path"]
    with open(manifest, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                intended[str(row["image_id"])] = row["intended_label_vector"]
    merged["__stratum"] = [
        "|".join(sorted(label for label, value in intended.get(str(image_id), {}).items() if int(value) == 1))
        or "__no_finding__" for image_id in merged["image_id"]
    ]
    return merged, intended, surviving
