#!/usr/bin/env python3
"""Train set utility and image ranking models from measured proxy-subset utility.

The command intentionally stops when proxy results are missing. This prevents fabricated image
targets. Image targets are estimated as mean leave-one-out marginal utility across the measured
subsets in which each image appeared.

Train/validation split is NOT a random split of subsets — subsets share images, so that would leak.
Every subset already carries a "role" ("train" | "val") from 04_build_utility_subsets.py's
--phase build, which built train-role and val-role subsets from two disjoint image pools. Splitting
by that role therefore guarantees zero image overlap between what the SetUtilityNetwork is trained
on and what it is validated on, and gives the ranking network the same disjoint split for free
(an image can only ever appear in train-role OR val-role subsets, never both).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.stats
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.asism.learned import active_feature_columns, contributing_signals, read_jsonl, safe_feature_frame, validate_utility_results  # noqa: E402
from scripts.asism.candidate_pool import load_candidate_pool  # noqa: E402
from scripts.asism.models import MultiObjectiveRankingNetwork, SetUtilityNetwork, pairwise_ranking_loss  # noqa: E402
from scripts.utils.artifact_contracts import stage3_paths  # noqa: E402
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.manifest import hash_dict, read_json, sha256_file, write_frozen_json  # noqa: E402
from scripts.utils.seed import set_seed  # noqa: E402


def padded_batch(records, feature_lookup, device):
    maximum = max(len(row["image_ids"]) for row in records)
    dimension = len(next(iter(feature_lookup.values())))
    features = torch.zeros(len(records), maximum, dimension, dtype=torch.float32, device=device)
    mask = torch.zeros(len(records), maximum, dtype=torch.bool, device=device)
    for row_index, row in enumerate(records):
        vectors = [feature_lookup[image_id] for image_id in row["image_ids"] if image_id in feature_lookup]
        if vectors:
            tensor = torch.as_tensor(np.asarray(vectors), dtype=torch.float32, device=device)
            features[row_index, :len(vectors)] = tensor
            mask[row_index, :len(vectors)] = True
    return features, mask


def split_subsets_by_role(subsets: list[dict]) -> tuple[list[dict], list[dict]]:
    missing_role = [row["subset_id"] for row in subsets if "role" not in row]
    if missing_role:
        raise SystemExit(
            f"{len(missing_role)} subset(s) have no 'role' field (e.g. {missing_role[:3]}). "
            "Rebuild utility_subsets.jsonl with 04_build_utility_subsets.py --phase build."
        )
    train = [row for row in subsets if row["role"] == "train"]
    val = [row for row in subsets if row["role"] == "val"]
    return train, val


def image_overlap_fraction(subsets_a: list[dict], subsets_b: list[dict]) -> float:
    images_a = {image_id for row in subsets_a for image_id in row["image_ids"]}
    images_b = {image_id for row in subsets_b for image_id in row["image_ids"]}
    union = images_a | images_b
    if not union:
        return 0.0
    return len(images_a & images_b) / len(union)


@torch.no_grad()
def predict_utility(model, subsets, lookup, device) -> list[float]:
    model.eval()
    if not subsets:
        return []
    x, mask = padded_batch(subsets, lookup, device)
    return model(x, mask).cpu().numpy().tolist()


def train_set_model(model, train_subsets, val_subsets, utility_by_id, lookup, config, device,
                    min_val_subsets: int) -> dict:
    """Trains only on train_subsets. Early-stops on val_subsets MSE (image-disjoint by construction)
    using early_stopping_patience — this is fitting a fixed, already-measured dataset, not the
    proxy-classifier candidate search that docs/stages2_to_5_plan.md §4.7/§7.1 forbids early
    stopping for, so patience-based stopping is appropriate here.

    Image-disjoint validation is a design invariant, not an optional nicety: if val_subsets doesn't
    meet min_val_subsets (feasibility_thresholds.min_val_subsets), this refuses to train at all
    rather than silently skipping early stopping and validation reporting.
    """
    if len(val_subsets) < min_val_subsets:
        raise SystemExit(
            f"Learned ASISM requires >= {min_val_subsets} val-role subsets for image-disjoint "
            f"validation (feasibility_thresholds.min_val_subsets); got {len(val_subsets)}. "
            "Training refuses to proceed without real held-out validation — rebuild "
            "utility_subsets.jsonl (04_build_utility_subsets.py --phase build) with a "
            "val_pool_fraction/total_subsets that satisfies this, or revise the threshold with "
            "explicit sign-off, before retrying."
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate),
                                  weight_decay=float(config.weight_decay))
    rng = np.random.default_rng(42)
    patience = int(config.early_stopping_patience)
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    total_epochs = int(config.epochs)
    stopped_at_epoch = total_epochs

    for epoch in range(total_epochs):
        model.train()
        order = rng.permutation(len(train_subsets))
        for start in range(0, len(order), int(config.batch_size)):
            rows = [train_subsets[index] for index in order[start:start + int(config.batch_size)]]
            x, mask = padded_batch(rows, lookup, device)
            target = torch.tensor([utility_by_id[row["subset_id"]] for row in rows], device=device)
            prediction = model(x, mask)
            loss = float(config.utility_mse_weight) * torch.nn.functional.mse_loss(prediction, target)
            loss = loss + float(config.pairwise_ranking_weight) * pairwise_ranking_loss(prediction, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        val_predictions = predict_utility(model, val_subsets, lookup, device)
        val_targets = [utility_by_id[row["subset_id"]] for row in val_subsets]
        val_loss = float(np.mean((np.asarray(val_predictions) - np.asarray(val_targets)) ** 2))
        if val_loss < best_val_loss - 1e-9:
            best_val_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                stopped_at_epoch = epoch + 1
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_predictions = predict_utility(model, val_subsets, lookup, device)
    val_targets = [utility_by_id[row["subset_id"]] for row in val_subsets]
    mae = float(np.mean(np.abs(np.asarray(val_predictions) - np.asarray(val_targets))))
    spearman, spearman_undefined_reason = None, None
    correlation = scipy.stats.spearmanr(val_predictions, val_targets)
    statistic = getattr(correlation, "statistic", None)
    raw_spearman = float(statistic) if statistic is not None else float(correlation[0])
    if math.isnan(raw_spearman):
        # scipy returns NaN when predictions or targets are (numerically) constant — correlation is
        # undefined, not zero. NaN is not valid JSON, so this must never reach the manifest as a
        # bare float; None + an explicit reason is the only honest representation.
        spearman_undefined_reason = "constant_predictions_or_targets"
    else:
        spearman = raw_spearman

    return {
        "n_train_subsets": len(train_subsets),
        "n_val_subsets": len(val_subsets),
        "image_overlap_fraction": image_overlap_fraction(train_subsets, val_subsets),
        "early_stopped_at_epoch": stopped_at_epoch,
        "spearman": spearman,
        "spearman_undefined_reason": spearman_undefined_reason,
        "mae": mae,
    }


@torch.no_grad()
def marginal_targets(model, subsets, lookup, device):
    """Mean SIZE-NORMALIZED predicted U(S)-U(S without i); never copy the subset target to members.

    The raw leave-one-out difference is not comparable across subsets of different sizes, so it
    cannot be averaged as-is. SetUtilityNetwork pools its per-image encodings by MEAN
    (`set_utility_network.pooling: "mean"`, models.py::SetUtilityNetwork.encode_set), so for a subset
    S of size n with mean encoding m_S:

        m_S - m_(S\\i) = (f(x_i) - m_S) / (n - 1)

    and therefore U(S) - U(S\\i) shrinks like 1/(n-1). With subset_sizes spanning 100..250, the same
    image quality would yield targets differing by ~2.5x purely from which subsets the image landed
    in — noise injected straight into the ranking network's supervision, unrelated to image quality.

    Multiplying by (n - 1) removes that factor and leaves a size-invariant quantity proportional to
    how far the image's encoding deviates from its subset's mean, which is the quantity that
    actually carries "is this image better than its peers". Only then is averaging across an image's
    subsets meaningful.

    `raw_contribution * (n - 1)` is exact for the linear part of the head and a first-order
    approximation for its nonlinearity; both are far closer to size-invariant than the unscaled
    difference, which is size-dependent by construction.
    """
    totals, counts = {}, {}
    model.eval()
    for row in subsets:
        ids = [image_id for image_id in row["image_ids"] if image_id in lookup]
        if len(ids) < 2:
            continue
        size_normalizer = len(ids) - 1
        full_x, full_mask = padded_batch([{"image_ids": ids}], lookup, device)
        full = float(model(full_x, full_mask).item())
        for image_id in ids:
            reduced = [candidate for candidate in ids if candidate != image_id]
            x, mask = padded_batch([{"image_ids": reduced}], lookup, device)
            contribution = (full - float(model(x, mask).item())) * size_normalizer
            totals[image_id] = totals.get(image_id, 0.0) + contribution
            counts[image_id] = counts.get(image_id, 0) + 1
    return {image_id: totals[image_id] / counts[image_id] for image_id in totals}, counts


def train_ranker(model, features, targets, config, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate))
    ids = list(targets)
    rng = np.random.default_rng(43)
    model.train()
    for _ in range(int(config.epochs)):
        rng.shuffle(ids)
        for start in range(0, len(ids), int(config.batch_size)):
            batch = ids[start:start + int(config.batch_size)]
            x = torch.tensor(np.asarray([features[i] for i in batch]), dtype=torch.float32, device=device)
            y = torch.tensor([targets[i] for i in batch], dtype=torch.float32, device=device)
            prediction = model(x)
            loss = torch.nn.functional.smooth_l1_loss(prediction, y)
            loss += float(config.pairwise_ranking_weight) * pairwise_ranking_loss(prediction, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()


@torch.no_grad()
def pairwise_ranking_accuracy(model, features, targets, device) -> float | None:
    """Fraction of correctly ordered pairs among images with distinct targets. 0.5 = chance level.

    This measures how well the ranker reproduces the SetUtilityNetwork's OWN marginal-utility
    estimates on held-out images — i.e. distillation fidelity — not independent evidence of
    downstream classifier utility. That independent evidence, if it exists, comes from proxy
    verification against real training runs (Phase 2 §2.7/§2.8), not from this metric.
    """
    model.eval()
    ids = list(targets)
    if len(ids) < 2:
        return None
    x = torch.tensor(np.asarray([features[i] for i in ids]), dtype=torch.float32, device=device)
    predictions = model(x).cpu().numpy()
    target_values = np.asarray([targets[i] for i in ids])
    correct, total = 0, 0
    for i in range(len(ids)):
        delta_target = target_values[i + 1:] - target_values[i]
        delta_prediction = predictions[i + 1:] - predictions[i]
        valid = np.abs(delta_target) > 1e-12
        if not valid.any():
            continue
        total += int(valid.sum())
        correct += int((np.sign(delta_target[valid]) == np.sign(delta_prediction[valid])).sum())
    return (correct / total) if total else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=None)
    args = parser.parse_args()
    cfg = load_named_config("stage3_asism.yaml", "stage3")
    namespace = args.namespace or str(cfg.split_namespace)
    for key, value in stage3_paths(cfg, namespace).items():
        if key in cfg.paths: cfg.paths[key] = str(value)
    # BUG FIXED 2026-08-21: torch's global RNG was never seeded, so SetUtilityNetwork's and
    # MultiObjectiveRankingNetwork's weight initialization (and dropout) differed every run —
    # confirmed to occasionally collapse the ranker enough that 06_learn_thresholds_select.py
    # selects 0 images. Reuses learned_asism.subset_design.seed (already the canonical seed for
    # this pipeline's subset construction, per 04_build_utility_subsets.py) rather than adding a
    # second seed knob that could silently drift out of sync with it.
    set_seed(int(cfg.learned_asism.subset_design.seed))
    subset_path, result_path = Path(cfg.paths.utility_subsets), Path(cfg.paths.utility_results)
    if not subset_path.is_file() or not result_path.is_file():
        raise SystemExit("Learned ASISM needs frozen utility_subsets.jsonl and measured utility_results.jsonl; run Stage 3 proxy experiments first.")
    subsets, results = read_jsonl(subset_path), read_jsonl(result_path)
    train_subsets, val_subsets = split_subsets_by_role(subsets)
    result_frame = validate_utility_results(subsets, results)
    utility = dict(zip(result_frame.subset_id, result_frame.utility_delta))

    merged, _, surviving = load_candidate_pool(cfg)
    columns = active_feature_columns(list(cfg.learned_asism.feature_columns), surviving)
    design_report = read_json(Path(cfg.paths.subset_design_report))
    if design_report.get("surviving_signals") != sorted(surviving) or \
            design_report.get("active_feature_columns") != columns:
        raise SystemExit(
            "UPSTREAM GATE: utility subsets were designed for a different Go/No-Go feature set. "
            "Re-run 04_build_utility_subsets.py --phase feasibility and --phase build before training."
        )
    normalized, normalization = safe_feature_frame(merged, columns)
    lookup = {str(image_id): row.to_numpy(np.float32) for image_id, (_, row) in zip(merged.image_id, normalized.iterrows())}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_cfg = cfg.learned_asism.set_utility_network
    set_model = SetUtilityNetwork(len(columns), tuple(set_cfg.image_hidden_dims), tuple(set_cfg.utility_hidden_dims)).to(device)
    min_val_subsets = int(cfg.learned_asism.subset_design.feasibility_thresholds.min_val_subsets)
    validation = train_set_model(set_model, train_subsets, val_subsets, utility, lookup, set_cfg, device,
                                 min_val_subsets)

    train_targets, train_exposures = marginal_targets(set_model, train_subsets, lookup, device)
    val_targets_raw, val_exposures = marginal_targets(set_model, val_subsets, lookup, device)
    minimum = int(cfg.learned_asism.subset_design.minimum_image_exposures)
    train_targets = {key: value for key, value in train_targets.items() if train_exposures[key] >= minimum}
    val_targets_raw = {key: value for key, value in val_targets_raw.items() if val_exposures[key] >= minimum}
    if len(train_targets) < 2:
        raise SystemExit("Too few train-role images have sufficient subset exposure for image-ranking training.")
    rank_cfg = cfg.learned_asism.ranking_network
    ranker = MultiObjectiveRankingNetwork(len(columns), tuple(rank_cfg.hidden_dims), float(rank_cfg.dropout)).to(device)
    train_ranker(ranker, lookup, train_targets, rank_cfg, device)
    ranker_pairwise_accuracy = pairwise_ranking_accuracy(ranker, lookup, val_targets_raw, device)

    output = Path(cfg.paths.learned_dir); output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": set_model.state_dict(), "input_dim": len(columns)}, output / "set_utility_model.pt")
    torch.save({"state_dict": ranker.state_dict(), "input_dim": len(columns)}, output / "ranking_model.pt")
    with open(output / "image_marginal_targets.jsonl", "w", encoding="utf-8") as handle:
        for image_id, target in sorted(train_targets.items()):
            handle.write(json.dumps({"image_id": image_id, "marginal_utility": target,
                                     "exposures": train_exposures[image_id], "role": "train"}) + "\n")
        for image_id, target in sorted(val_targets_raw.items()):
            handle.write(json.dumps({"image_id": image_id, "marginal_utility": target,
                                     "exposures": val_exposures[image_id], "role": "val"}) + "\n")
    signals_used = contributing_signals(columns)
    manifest = {"schema_version": 1, "method": "set_utility_to_marginal_ranking_v1", "frozen": True,
                "feature_columns": columns, "normalization": normalization,
                "contributing_signals": signals_used,
                # Mirrors 02_gonogo.py's asism_variant_status rule for the WEIGHTED selector, applied
                # here to the learned one: a ranking network fed by fewer than three admitted signals
                # is reported as an ablation/alternative, not as the primary method (§4.6).
                "learned_variant_status": (
                    "primary" if len(signals_used) >= 3 else
                    f"reduced_variant — only {len(signals_used)} signal(s) ({signals_used}) survived "
                    "Go/No-Go and feed the ranking network; report this selector as an "
                    "ablation/alternative rather than the primary method (§4.6)"
                ),
                "utility_definition": "augmented_macro_auroc-real_only_macro_auroc",
                "n_subsets": len(subsets), "n_train_subsets": len(train_subsets), "n_val_subsets": len(val_subsets),
                "n_rank_targets": len(train_targets),
                "subset_sha256": sha256_file(subset_path), "results_sha256": sha256_file(result_path),
                "config_hash": hash_dict(OmegaConf.to_container(cfg.learned_asism, resolve=True), length=64),
                "validation": validation,
                "ranker_validation": {
                    "pairwise_accuracy": ranker_pairwise_accuracy,
                    "measures": "distillation_fidelity_not_downstream_utility",
                    "n_val_images": len(val_targets_raw),
                }}
    write_frozen_json(output / "learned_training_manifest.json", manifest)
    print(f"Learned ASISM models -> {output}")
    print(f"Set-utility validation: spearman={validation['spearman']}, mae={validation['mae']}, "
          f"image_overlap_fraction={validation['image_overlap_fraction']}")
    print(f"Ranker held-out pairwise accuracy: {ranker_pairwise_accuracy}")
    print("Next: learn/freeze class thresholds and emit learned selected_manifest.jsonl.")
    return 0


if __name__ == "__main__": raise SystemExit(main())
