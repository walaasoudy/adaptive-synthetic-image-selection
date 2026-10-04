#!/usr/bin/env python3
"""The end-to-end wiring smoke changes only what it says it changes.

The smoke itself takes about 20 minutes and needs the candidate signal files, so it is not run
here. These tests hold its fixture to its description: the candidate manifest keeps every column
but image_path, the Stage 4 overlay touches the budget and the seed list and nothing else, and the
work directory cannot be inside the repository.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fixture_workspace import fixture_workspace  # noqa: E402

from scripts.smoke import asism_v2_learned_e2e_smoke as smoke  # noqa: E402
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS  # noqa: E402
from scripts.utils.ham10000_conditions import get_protocol  # noqa: E402


def test_the_fixture_manifest_differs_from_the_source_only_in_image_path():
    with fixture_workspace("e2e-smoke-fixture") as root:
        source = root / "source.csv"
        pd.DataFrame({"image_id": [f"syn_{dx}_{i}" for dx in CLASSIFIER_TARGET_LABELS for i in range(2)],
                      "image_path": "/pod/only/path.jpg",
                      "dx": [dx for dx in CLASSIFIER_TARGET_LABELS for _ in range(2)],
                      "seed": range(14)}).to_csv(source, index=False)
        work = root / "work"
        work.mkdir()
        target = smoke.build_workspace(work, source)

        assert target == work / "outputs" / "ham10000" / "stage2" / smoke.NAMESPACE / "all_candidates.csv"
        before, after = pd.read_csv(source), pd.read_csv(target)
        assert before.drop(columns="image_path").equals(after.drop(columns="image_path"))
        assert all(Path(path).is_file() for path in after["image_path"])
        for split in smoke.REAL_SPLITS:
            frame = pd.read_csv(work / "data" / "ham10000" / "processed" / "splits" / smoke.NAMESPACE / f"{split}.csv")
            assert frame["dx"].value_counts().to_dict() == {dx: smoke.IMAGES_PER_CLASS for dx in CLASSIFIER_TARGET_LABELS}
            images = work / "data" / "ham10000" / "processed" / "images" / smoke.NAMESPACE / split
            assert all((images / f"{image_id}.jpg").is_file() for image_id in frame["image_id"])
        assert "final_eval_heldout" not in smoke.REAL_SPLITS
        assert OmegaConf.to_container(OmegaConf.load(work / "smoke_overlay.yaml")) == smoke.OVERLAY


def test_the_overlay_changes_the_budget_and_the_seed_list_and_nothing_else():
    committed = OmegaConf.load(REPO / "configs" / get_protocol(smoke.PROTOCOL).stage4_config)
    overlay = smoke.OVERLAY["ham_stage4"]
    assert set(overlay) == {"model", "training", "seeds"}
    assert set(overlay["model"]) == {"pretrained_source", "resolution"}
    assert set(overlay["training"]) == {"max_steps", "batch_size", "eval_every_n_steps"}
    merged = OmegaConf.merge(committed, OmegaConf.create(overlay))
    for key in ("paths", "conditions", "split_namespace", "real_train_split", "selection_split", "loss", "classes"):
        assert merged[key] == committed[key], key
    assert set(overlay["seeds"]) <= {int(seed) for seed in committed.seeds}


def test_the_work_directory_must_be_new_and_outside_the_repository():
    with pytest.raises(SystemExit, match="outside the repository"):
        smoke.run(Path("x.csv"), Path("scores"), REPO / "tests" / "_runtime" / "never-created", 20)
    with pytest.raises(SystemExit, match="outside the repository"):
        smoke.run(Path("x.csv"), Path("scores"), REPO.parent, 20)          # exists


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
