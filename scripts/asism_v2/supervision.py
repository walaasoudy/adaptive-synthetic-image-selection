"""The designed utility subsets the V2 ranker is supervised by (contract §8). CPU only.

Why designed. Earlier labels were measured on random subsets of one size, which barely differ from
one another: their difference was below the training noise. Here the subsets are built to differ in
size, in class fractions and, for half of them, in which end of one signal they are drawn from.

    roles        each candidate image is given one role (train / validation / test), within its
                 class. A subset holds images of one role only, so the three sets of subsets share
                 no image.
    size         cycles through the role's sizes.
    fractions    Dirichlet(concentration * role class proportions), then capped by what the role
                 holds of each class.
    composition  "random": drawn at random within each class.
                 "tilted": within each class drawn from the upper (or lower) half of one signal; when
                 the class needs more images than half of what the role holds, the top (or bottom)
                 images by that signal.

This module only builds and freezes the plan. It trains nothing and reads no split outcome.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import fingerprint, write_new_json
from .features import SIGNALS, validate_frame
from .preserve import sha256

ROLES = ("train", "validation", "test")
DIRECTIONS = ("upper", "lower")
CANDIDATES_SHA256 = "bf8047b5dde96259c1268e5d8a3db6bac0d37392088df45182fde350ac7ef51a"
SIGNAL_ARTIFACT = {"similarity_knn_mean": "similarity", "iqa_composite": "iqa",
                   "uncertainty_mutual_information": "uncertainty",
                   "explainability_calibrated_typicality": "explainability"}
PLAN_NAME = "utility_plan.json"


class SupervisionError(RuntimeError):
    """Raised instead of building a plan the design cannot carry."""


def ids_sha256(ids) -> str:
    return fingerprint({"ids": sorted(map(str, ids))})


def load_signal_table(candidates_csv: Path, scores_dir: Path, safety: dict,
                      expected_candidates_sha256: str = CANDIDATES_SHA256) -> tuple[pd.DataFrame, dict]:
    """-> (image_id, dx, image_path, the four signals) for the safe pool, and what was checked."""
    from scripts.utils.ham10000 import normalize_diagnosis

    candidates_sha = sha256(Path(candidates_csv))
    if candidates_sha != expected_candidates_sha256:
        raise SupervisionError(f"{candidates_csv}: sha256 {candidates_sha} is not the frozen candidate pool")
    pool = pd.read_csv(candidates_csv)
    pool["image_id"] = pool["image_id"].astype(str)
    pool["dx"] = [normalize_diagnosis(v) for v in pool["dx"]]
    artifacts, hashes = {}, {}
    for artifact in sorted(set(SIGNAL_ARTIFACT.values())):
        path = Path(scores_dir) / f"{artifact}_scores.parquet"
        frame = pd.read_parquet(path)
        frame["image_id"] = frame["image_id"].astype(str)
        if set(frame["image_id"]) != set(pool["image_id"]) or frame["image_id"].duplicated().any():
            raise SupervisionError(f"{path.name} does not cover exactly the candidate pool")
        artifacts[artifact] = frame.set_index("image_id")
        hashes[artifact] = sha256(path)
    pool = pool.set_index("image_id", drop=False)
    for column, artifact in SIGNAL_ARTIFACT.items():
        pool[column] = artifacts[artifact][column].reindex(pool.index).to_numpy(dtype=float)
    removed = {}
    if safety.get("reject_invalid_iqa"):
        invalid = artifacts["iqa"].index[~artifacts["iqa"]["iqa_valid"].astype(bool)]
        removed.update({i: "invalid_iqa" for i in invalid})
    if safety.get("reject_near_duplicates"):
        near = artifacts["similarity"].index[artifacts["similarity"]["novelty_is_near_duplicate"].astype(bool)]
        removed.update({i: "near_duplicate" for i in near})
    safe = pool.drop(index=list(removed)).reset_index(drop=True)
    keep = ["image_id", "dx", *SIGNALS] + (["image_path"] if "image_path" in safe else [])
    safe = safe[keep]
    validate_frame(safe)
    report = {"candidates_csv_sha256": candidates_sha, "signal_artifact_sha256": hashes,
              "safety_applied": {k: bool(v) for k, v in safety.items()}, "safety_removed": len(removed),
              "safe_pool": int(len(safe)), "safe_pool_ids_sha256": ids_sha256(safe["image_id"]),
              "per_class": {c: int(n) for c, n in safe["dx"].value_counts().sort_index().items()}}
    return safe, report


def _largest_remainder(total: int, shares: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Integers summing to `total`, proportional to `shares`, none above its cap."""
    if total > caps.sum():
        raise SupervisionError(f"{total} images asked from a role that holds {int(caps.sum())}")
    counts = np.zeros(len(shares), dtype=int)
    while counts.sum() < total:
        room = caps - counts
        open_ = room > 0
        weight = np.where(open_, shares, 0.0)
        if weight.sum() <= 0:
            weight = open_.astype(float)
        target = (total - counts.sum()) * weight / weight.sum()
        step = np.minimum(np.floor(target).astype(int), room)
        if step.sum() == 0:                               # hand out single images by remainder
            order = sorted(np.flatnonzero(open_), key=lambda c: (-(target[c] - np.floor(target[c])), c))
            step[order[0]] = 1
        counts += step
    return counts


def assign_roles(frame: pd.DataFrame, fractions: dict, seed: int) -> dict[str, str]:
    """image_id -> role, drawn within each class so every role holds every class."""
    if tuple(fractions) != ROLES or abs(sum(fractions.values()) - 1) > 1e-9:
        raise SupervisionError(f"role fractions must be given for {ROLES} and sum to 1")
    roles = {}
    for index, (dx, group) in enumerate(frame.groupby("dx", sort=True)):
        ids = sorted(group["image_id"].astype(str))
        counts = _largest_remainder(len(ids), np.array([fractions[r] for r in ROLES]),
                                    np.full(len(ROLES), len(ids)))
        if (counts == 0).any():
            raise SupervisionError(f"class {dx} has too few images ({len(ids)}) to appear in every role")
        order = np.random.default_rng([int(seed), index]).permutation(len(ids))
        start = 0
        for role, count in zip(ROLES, counts):
            roles.update({ids[k]: role for k in order[start:start + count]})
            start += count
    return roles


def _draw(members: pd.DataFrame, counts: dict[str, int], tilt: tuple[str, str] | None, rng) -> list[str]:
    chosen = []
    for dx in sorted(counts):
        count = counts[dx]
        if count == 0:
            continue
        group = members[members["dx"] == dx]
        if tilt is None:
            eligible = sorted(group["image_id"])
        else:
            signal, direction = tilt
            ordered = group.sort_values([signal, "image_id"], ascending=[direction == "lower", True])
            eligible = list(ordered["image_id"].iloc[:max(int(np.ceil(len(group) / 2)), count)])
        chosen += [eligible[k] for k in sorted(rng.choice(len(eligible), size=count, replace=False))]
    return chosen


def build_plan(frame: pd.DataFrame, prereg: dict, pool_report: dict | None = None) -> dict:
    """The frozen design: roles, every subset's members, and the (subset, seed) cells to measure."""
    validate_frame(frame)
    design = prereg["supervision"]
    frame = frame.assign(image_id=frame["image_id"].astype(str), dx=frame["dx"].astype(str))
    classes = sorted(frame["dx"].unique())
    roles = assign_roles(frame, dict(design["role_fractions"]), int(design["role_seed"]))
    tilts = [(signal, direction) for signal in SIGNALS for direction in DIRECTIONS]
    subsets, seen = {}, set()
    for role_index, role in enumerate(ROLES):
        members = frame[frame["image_id"].map(roles) == role]
        caps = np.array([(members["dx"] == c).sum() for c in classes])
        proportions = caps / caps.sum()
        sizes = [int(s) for s in design["sizes"][role]]
        tilted_seen = 0
        for k in range(int(design["subsets"][role])):
            size = sizes[k % len(sizes)]
            block = k // len(sizes)
            tilted = (block % 2 == 1) if float(design["tilted_fraction"]) == 0.5 else None
            if tilted is None:
                raise SupervisionError("only tilted_fraction 0.5 (alternating blocks) is implemented")
            tilt = tilts[tilted_seen % len(tilts)] if tilted else None
            tilted_seen += int(tilted)
            for attempt in range(50):
                rng = np.random.default_rng([int(design["design_seed"]), role_index, k, attempt])
                shares = rng.dirichlet(float(design["class_fraction_concentration"]) * proportions)
                counts = dict(zip(classes, map(int, _largest_remainder(size, shares, caps))))
                ids = _draw(members, counts, tilt, rng)
                key = ids_sha256(ids)
                if key not in seen:
                    break
            else:
                raise SupervisionError(f"could not draw a new {role} subset of size {size}")
            seen.add(key)
            subsets[f"{role}-{k:03d}"] = {
                "role": role, "size": size, "kind": "tilted" if tilt else "random",
                "tilt_signal": tilt[0] if tilt else None, "tilt_direction": tilt[1] if tilt else None,
                "class_counts": counts, "image_ids": ids, "members_sha256": key}
    seeds = [int(s) for s in design["training_seeds"]]
    plan = {
        "experiment": "asism_v2_utility_supervision",
        "design_document": "docs/asism_v2_quantity_design_check_2026-10-03.md §8",
        "prereg_sha256": prereg["prereg_sha256"],
        "metric": design["metric"], "proxy": dict(design["proxy"]),
        "pool": pool_report or {"safe_pool": int(len(frame)), "safe_pool_ids_sha256": ids_sha256(frame["image_id"])},
        "role_counts": {role: {c: int(((frame["image_id"].map(roles) == role) & (frame["dx"] == c)).sum())
                               for c in classes} for role in ROLES},
        "roles": dict(sorted(roles.items())),
        "subsets": subsets,
        "training_seeds": seeds,
        "cells": [{"subset_id": sid, "seed": seed} for seed in seeds for sid in subsets],
        "runs_planned": len(subsets) * len(seeds),
    }
    check_plan(plan, frame)
    return plan


def check_plan(plan: dict, frame: pd.DataFrame) -> None:
    """The properties the fit and the stopping rule depend on, checked rather than assumed."""
    roles = plan["roles"]
    dx = frame.set_index(frame["image_id"].astype(str))["dx"].astype(str)
    classes = set(dx)
    for sid, subset in plan["subsets"].items():
        ids = subset["image_ids"]
        if len(ids) != subset["size"] or len(set(ids)) != len(ids):
            raise SupervisionError(f"{sid}: wrong size or duplicated images")
        if {roles[i] for i in ids} != {subset["role"]}:
            raise SupervisionError(f"{sid}: holds an image of another role")
        if ids_sha256(ids) != subset["members_sha256"]:
            raise SupervisionError(f"{sid}: members do not match their hash")
    for role in ROLES:
        own = [s for s in plan["subsets"].values() if s["role"] == role]
        if {c for s in own for c in dx.loc[s["image_ids"]]} != classes:
            raise SupervisionError(f"the {role} subsets do not cover every class")
    train = [s for s in plan["subsets"].values() if s["role"] == "train"]
    if len({s["size"] for s in train}) < 2:
        raise SupervisionError("the train subsets have one size: the size term would not be identifiable")
    fractions = {tuple(sorted((c, round(n / s["size"], 12)) for c, n in s["class_counts"].items())) for s in train}
    if len(fractions) < 2:
        raise SupervisionError("the train subsets have one class allocation: the class term would not be identifiable")


def freeze_plan(out_dir: Path, plan: dict) -> Path:
    """Written once. A second plan in the same directory is refused, never re-drawn."""
    path = Path(out_dir) / PLAN_NAME
    if path.exists():
        raise SupervisionError(f"a frozen plan already exists at {path}; it is never re-drawn")
    write_new_json(path, plan)
    return path


def fit_inputs(plan: dict) -> dict[str, dict]:
    """The `subsets` argument of pipeline.fit_ranker."""
    return {sid: {"role": s["role"], "image_ids": list(s["image_ids"])} for sid, s in plan["subsets"].items()}


def describe(plan: dict) -> dict:
    """Counts a reader checks before approving a GPU run."""
    rows = pd.DataFrame([{"role": s["role"], "size": s["size"], "kind": s["kind"]} for s in plan["subsets"].values()])
    exposure = pd.Series([i for s in plan["subsets"].values() for i in s["image_ids"]]).value_counts()
    return {"subsets_by_role_size_kind": {f"{r}/{n}/{k}": int(c) for (r, n, k), c in
                                          rows.groupby(["role", "size", "kind"]).size().items()},
            "runs_planned": plan["runs_planned"],
            "images_never_in_a_subset": int(len(plan["roles"]) - len(exposure)),
            "exposures_per_image": {"min": int(exposure.min()), "median": float(exposure.median()),
                                    "max": int(exposure.max())}}
