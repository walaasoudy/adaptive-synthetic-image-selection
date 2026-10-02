#!/usr/bin/env python3
"""Train the ASISM v2 final Stage 4 grid: every (condition, seed) of the protocol, resumably.

The recipe is ham10000_train_conditions.run, unchanged; this only loops, skips runs that already
finished, and checks that every selected-condition run trained on exactly the file the selection
manifest hashed (so the images C and D were built from are the images used).

Order is seed-major (all conditions at seed 42, then 43, ...), so an interrupted grid is balanced
across conditions. A finished run is never retrained; a failed one is rerun at the same seed.

Usage (on the GPU pod):
    python -m scripts.followup.ham10000_asism_v2_stage4_grid --protocol asism_v2 --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.classify import ham10000_train_conditions as stage4
from scripts.utils.config import load_named_config
from scripts.utils.ham10000_conditions import get_protocol
from scripts.utils.manifest import read_json, sha256_file

DONE_FILES = ("run_manifest.json", "selection_metrics.json", "selection_predictions.parquet", "model.pt")


class GridError(SystemExit):
    pass


def expected_hashes(protocol, config) -> dict[str, str]:
    """condition-or-file -> sha256 the selection manifest recorded, for C and every D draw."""
    if not protocol.selection_manifest.covers:
        return {}
    selection = load_named_config(protocol.selection_manifest.config, protocol.selection_manifest.section)
    path = Path(selection.paths.outputs_dir) / str(config.split_namespace) / protocol.selection_manifest.filename
    if not path.is_file():
        raise GridError(f"no selection manifest at {path}; build C and D first")
    manifest = read_json(path)
    if manifest.get("stage4_seeds") != [int(s) for s in config.seeds]:
        raise GridError("the selection manifest was built for another seed list")
    return dict(manifest["files_sha256"])


def verify_inputs(protocol, config, seed: int, condition: str, hashes: dict[str, str]) -> str | None:
    manifest = stage4.resolve_synthetic_manifest(config, condition, seed)
    if not manifest or condition not in protocol.selection_manifest.covers:
        return None
    name = Path(manifest).name
    actual = sha256_file(manifest)
    if hashes.get(name) != actual:
        raise GridError(f"{condition}/seed{seed}: {name} sha256 {actual} is not the one the selection recorded")
    return actual


def run(protocol_name: str, device: str | None) -> list[str]:
    protocol = get_protocol(protocol_name)
    config = load_named_config(protocol.stage4_config, "ham_stage4")
    hashes = expected_hashes(protocol, config)
    results = Path(config.paths.results_dir) / str(config.split_namespace)
    trained = []
    for seed in [int(s) for s in config.seeds]:
        for condition in protocol.conditions:
            expected = verify_inputs(protocol, config, seed, condition, hashes)
            out_dir = results / condition / f"seed{seed}"
            if all((out_dir / f).is_file() for f in DONE_FILES):
                recorded = read_json(out_dir / "run_manifest.json")["data"].get("synthetic_manifest_sha256")
                if expected and recorded != expected:
                    raise GridError(f"{out_dir} trained on a different file than the selection recorded")
                continue
            print(f"[grid] {condition}/seed{seed}", flush=True)
            result = stage4.run(config, condition, seed, device, protocol.name)
            recorded = result["manifest"]["data"].get("synthetic_manifest_sha256")
            if expected and recorded != expected:
                raise GridError(f"{condition}/seed{seed} trained on {recorded}, the selection recorded {expected}")
            print(json.dumps({"run": f"{condition}/seed{seed}",
                              "balanced_accuracy_classifier_val": result["metrics"]["balanced_accuracy"]}), flush=True)
            trained.append(f"{condition}/seed{seed}")
    print(f"[grid] complete: {len(protocol.conditions) * len(config.seeds)} runs", flush=True)
    return trained


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", required=True, choices=["asism_v2", "asism_v2_none"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(args.protocol, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
