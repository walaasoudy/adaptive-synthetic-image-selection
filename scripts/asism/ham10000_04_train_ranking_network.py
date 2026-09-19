#!/usr/bin/env python3
"""Stage 3d — the Multi-Objective Ranking Network: fuse the admitted signals into one utility score.

TWO MODELS, IN ORDER.

1. A permutation-invariant Deep Sets model (`SetUtilityNetwork`) learns U(S): what a SUBSET of
   synthetic candidates does to downstream balanced accuracy. It is trained only on the subsets that
   were actually measured on a GPU — no subset target is invented.
2. Each image's target is then DISTILLED from that model as its mean size-normalised leave-one-out
   marginal contribution. The subset-level number is never copied onto its member images: a good
   subset can contain a bad image, and copying the set's score to all of them would teach the ranker
   that whatever accompanied a good set is good.
3. The ranking network (`MultiSignalUtilityRankingNetwork`) is fitted to those per-image targets from
   the signal features alone, so that at selection time a candidate can be scored without measuring
   anything.

ON THE NAME. The thesis calls this the Multi-Objective Ranking Network because it FUSES MULTIPLE
OBJECTIVES — realism, technical quality, epistemic uncertainty, attention typicality, class
agreement — into one decision. The optimisation is single-objective by construction: one distilled
target, two loss terms over it (Smooth-L1 for the value, a pairwise term for the order). There is no
Pareto front, no task-balancing, no gradient surgery, and the manifest says so. Calling a
multi-task-optimisation method by this name would be a different claim than the one being made.

WHY NOT FIT THE RANKER DIRECTLY ON SUBSET UTILITY. Because utility is a property of sets, and the
quantity selection needs is a property of images. The Deep Sets model is the bridge: it is the only
place where a set-level measurement is allowed to be turned into an image-level one, and it does so
by an explicit marginal-contribution argument rather than by attribution.

TRAIN/VAL ARE IMAGE-DISJOINT BY CONSTRUCTION, not by a random split of subsets — subsets share
images, so splitting them randomly would leak. Every subset carries the role of the image pool it
was drawn from, and the two pools were split before any subset existed.

Usage:
    python scripts/asism/ham10000_04_train_ranking_network.py --namespace ham-stratified-v1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.asism.ham10000_ranking import (  # noqa: E402
    PoolGate,
    active_feature_columns,
    contributing_signals,
    load_candidate_pool,
    validate_utility_results,
)
from scripts.asism.learned import (  # noqa: E402
    banzhaf_msr_targets,
    normalize_ranking_targets,
    read_jsonl,
    safe_feature_frame,
    spearman_with_reason,
)
from scripts.asism.models import (  # noqa: E402
    MultiSignalUtilityRankingNetwork,
    SetUtilityNetwork,
    pairwise_ranking_loss,
)
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.manifest import get_git_commit_hash, write_json  # noqa: E402

MIN_TRAIN_SUBSETS = 3


def _shared_training_helpers():
    """The set-model training loop, the marginal-target distillation and the ranking loop.

    They live in scripts/asism/05_train_learned_asism.py, whose filename cannot be imported normally.
    They are IMPORTED rather than copied because the mathematics is genuinely dataset-independent —
    the size-normalisation argument, the leave-one-out estimator, the pairwise accuracy — and a copy
    would drift: a correction made to one dataset's version would silently not reach the other's.
    What is HAM10000-specific is the target these functions are applied to, which is supplied here.
    """
    path = Path(__file__).resolve().parent / "05_train_learned_asism.py"
    spec = importlib.util.spec_from_file_location("_asism_shared_training", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_subsets_by_role(subsets: list[dict]) -> tuple[list[dict], list[dict]]:
    missing = [row["subset_id"] for row in subsets if "role" not in row]
    if missing:
        raise PoolGate(
            f"{len(missing)} subset(s) carry no role (e.g. {missing[:3]}); rebuild them with "
            "ham10000_03_build_utility_subsets.py --phase build"
        )
    return (
        [row for row in subsets if row["role"] == "train"],
        [row for row in subsets if row["role"] == "val"],
    )


def image_overlap(a: list[dict], b: list[dict]) -> int:
    ids_a = {image_id for row in a for image_id in row["image_ids"]}
    ids_b = {image_id for row in b for image_id in row["image_ids"]}
    return len(ids_a & ids_b)


def run(namespace: str, device: str | None = None) -> dict:
    import torch

    from scripts.utils.seed import set_seed

    stage2 = load_named_config("ham10000_stage2.yaml", "ham_stage2")
    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    learned_cfg = stage3.learned_asism
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(learned_cfg.seed))

    out_dir = Path(stage3.paths.outputs_dir) / namespace / "learned"
    subsets_path = out_dir / "utility_subsets.jsonl"
    results_path = out_dir / "utility_results.jsonl"
    for path, command in (
        (subsets_path, "--phase build"),
        (results_path, "--phase measure"),
    ):
        if not path.is_file():
            raise PoolGate(
                f"UPSTREAM GATE: {path} is missing.\n"
                "The ranking network trains only on MEASURED subset utility — there is no fallback "
                "that estimates it from the signals, which are the model's own inputs. Run: "
                f"python scripts/asism/ham10000_03_build_utility_subsets.py --namespace {namespace} {command}"
            )

    metric = str(learned_cfg.utility_metric)
    subsets = read_jsonl(subsets_path)
    results = read_jsonl(results_path)
    measured = validate_utility_results(subsets, results, metric)
    utility_by_id = dict(zip(measured["subset_id"], measured["utility_delta"].astype(float)))

    frame, surviving, pool_report = load_candidate_pool(namespace, stage3, Path(stage2.paths.stage2_root))
    columns = active_feature_columns([str(column) for column in learned_cfg.feature_columns], surviving)
    features, feature_stats = safe_feature_frame(frame, columns)
    lookup = {
        str(image_id): row
        for image_id, row in zip(frame["image_id"], features.to_numpy(dtype=np.float32))
    }

    train_subsets, val_subsets = split_subsets_by_role(subsets)
    if len(train_subsets) < MIN_TRAIN_SUBSETS:
        raise PoolGate(f"only {len(train_subsets)} train-role subsets; at least {MIN_TRAIN_SUBSETS} are needed")
    overlap = image_overlap(train_subsets, val_subsets)
    if overlap:
        raise PoolGate(
            f"{overlap} image(s) appear in both a train-role and a val-role subset. The split is "
            "supposed to be disjoint BY CONSTRUCTION; this means the subsets were not built by "
            "ham10000_03_build_utility_subsets.py --phase build against this pool."
        )

    shared = _shared_training_helpers()
    set_cfg = learned_cfg.set_utility_network
    set_model = SetUtilityNetwork(
        input_dim=len(columns),
        image_hidden=tuple(int(value) for value in set_cfg.image_hidden_dims),
        utility_hidden=tuple(int(value) for value in set_cfg.utility_hidden_dims),
    ).to(device)

    history = shared.train_set_model(
        set_model, train_subsets, val_subsets, utility_by_id, lookup, set_cfg, device,
        int(learned_cfg.subset_design.feasibility_thresholds.min_val_subsets),
    )

    # Targets are distilled from the TRAIN-role subsets only. A target distilled from a val-role
    # subset would describe an image the set model was early-stopped against.
    targets, exposure = shared.marginal_targets(set_model, train_subsets, lookup, device)
    minimum_exposures = int(learned_cfg.subset_design.minimum_image_exposures)
    thin = sorted(image_id for image_id, count in exposure.items() if count < minimum_exposures)
    targets = {image_id: value for image_id, value in targets.items() if exposure[image_id] >= minimum_exposures}
    if len(targets) < int(learned_cfg.ranking_network.min_training_images):
        raise PoolGate(
            f"only {len(targets)} image(s) were measured in at least {minimum_exposures} subsets; "
            f"the ranking network needs {int(learned_cfg.ranking_network.min_training_images)}. "
            "Increase total_subsets or subset_sizes and re-measure — a target averaged over one or "
            "two subsets is noise, not a utility estimate."
        )

    # A second, independent estimator over the SAME measurements — no extra GPU cost. Reported, not
    # substituted: agreement between the two is evidence the distilled ordering is not an artifact of
    # the Deep Sets model, and disagreement is a finding for the write-up rather than something to
    # quietly average away.
    banzhaf, banzhaf_counts = banzhaf_msr_targets(train_subsets, utility_by_id)
    shared_ids = sorted(set(targets) & set(banzhaf))
    banzhaf_rho, banzhaf_reason = spearman_with_reason(
        [targets[image_id] for image_id in shared_ids], [banzhaf[image_id] for image_id in shared_ids]
    )

    normalised = normalize_ranking_targets(
        targets,
        method=str(learned_cfg.ranking_network.target_normalization),
        winsorize_quantile=float(learned_cfg.ranking_network.target_winsorize_quantile),
    )

    ranking_cfg = learned_cfg.ranking_network
    ranker = MultiSignalUtilityRankingNetwork(
        input_dim=len(columns),
        hidden=tuple(int(value) for value in ranking_cfg.hidden_dims),
        dropout=float(ranking_cfg.dropout),
    ).to(device)
    shared.train_ranker(ranker, lookup, normalised, ranking_cfg, device)

    # Held-out fidelity: images that appear ONLY in val-role subsets were never seen by the set model
    # in training and never contributed a target. Their targets are distilled here purely to measure
    # whether the ranker's ordering transfers — they are not trained on.
    val_targets, val_exposure = shared.marginal_targets(set_model, val_subsets, lookup, device)
    val_targets = {
        image_id: value
        for image_id, value in val_targets.items()
        if image_id not in targets and val_exposure[image_id] >= minimum_exposures
    }
    val_accuracy = shared.pairwise_ranking_accuracy(ranker, lookup, val_targets, device) if val_targets else None
    train_accuracy = shared.pairwise_ranking_accuracy(ranker, lookup, normalised, device)

    # Score every candidate in the pool, including the ones no subset ever contained.
    ranker.eval()
    with torch.no_grad():
        ids = [str(image_id) for image_id in frame["image_id"]]
        batch = torch.tensor(np.asarray([lookup[image_id] for image_id in ids]), dtype=torch.float32, device=device)
        scores = ranker(batch).cpu().numpy()

    scores_frame = pd.DataFrame(
        {
            "image_id": ids,
            "dx": frame["dx"].astype(str).to_numpy(),
            "ranking_score": scores.astype(float),
            "ranking_target": [normalised.get(image_id, float("nan")) for image_id in ids],
            "ranking_target_exposures": [int(exposure.get(image_id, 0)) for image_id in ids],
            "ranking_was_training_image": [image_id in normalised for image_id in ids],
        }
    )
    scores_path = out_dir / "ranking_scores.parquet"
    scores_frame.to_parquet(scores_path, index=False)

    checkpoint_dir = Path(learned_cfg.checkpoint_dir) / namespace
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ranker.state_dict(), checkpoint_dir / "ranking_model.pt")
    torch.save(set_model.state_dict(), checkpoint_dir / "set_utility_model.pt")

    contributing = contributing_signals(columns)
    manifest = {
        "stage": "ham10000_ranking_network",
        "namespace": namespace,
        "utility_metric": metric,
        "feature_columns": columns,
        "surviving_signals": surviving,
        "contributing_signals": contributing,
        # The name says multi-signal; the manifest records how many signals actually reached it.
        "is_multi_signal": len(contributing) > 1,
        "optimisation": "single_objective_two_loss_terms(smooth_l1 + pairwise_ranking)",
        "n_train_subsets": len(train_subsets),
        "n_val_subsets": len(val_subsets),
        "train_val_image_overlap": overlap,
        "n_training_images": len(normalised),
        "n_heldout_images_for_fidelity": len(val_targets),
        "images_below_minimum_exposure": len(thin),
        "minimum_image_exposures": minimum_exposures,
        "set_model_history": history,
        "banzhaf_agreement_spearman": banzhaf_rho,
        "banzhaf_agreement_undefined_reason": banzhaf_reason,
        "banzhaf_images_compared": len(shared_ids),
        "pairwise_ranking_accuracy_train": train_accuracy,
        "pairwise_ranking_accuracy_heldout": val_accuracy,
        "pairwise_accuracy_measures": (
            "distillation fidelity against the set model's own marginal estimates, NOT independent "
            "evidence of downstream classifier utility"
        ),
        "feature_normalisation": feature_stats,
        "pool": pool_report,
        "scores_path": str(scores_path),
        "checkpoint_dir": str(checkpoint_dir),
        "seed": int(learned_cfg.seed),
        "git_commit_hash": get_git_commit_hash(),
    }
    write_json(out_dir / "ranking_network_manifest.json", manifest)

    print(f"Ranking network trained on {len(normalised)} images from {len(train_subsets)} subsets", flush=True)
    print(f"  signals contributing: {contributing}", flush=True)
    print(f"  pairwise accuracy    train {train_accuracy} / held-out {val_accuracy}", flush=True)
    print(f"  Banzhaf agreement    rho={banzhaf_rho} over {len(shared_ids)} images", flush=True)
    print(f"  scores -> {scores_path}", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    manifest = run(args.namespace, device=args.device)
    print(json.dumps({key: manifest[key] for key in ("n_training_images", "contributing_signals", "scores_path")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
