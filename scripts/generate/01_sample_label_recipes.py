#!/usr/bin/env python3
"""Stage 2a — sample the synthetic label-recipe table (docs/stages2_to_5_plan.md §3).

Produces the `intended_label_vector` table Stage 2b generates from. These are conditioning
intents, NOT ground truth: whether the resulting pixels actually agree is measured separately by
ASISM's agreement signal (§4.5).

Recipe validity is an auditable procedure, not an assertion (§3):
  1. empirical co-occurrence mined from gen_train at >= min_support_patients;
  2. medical-rule overrides, both directions (allow-list and block-list);
  3. No Finding recipes are the ALL-ZERO vector over the 11 primary disease labels, mutually
     exclusive with any positive pathology;
  4. Support Devices is a context attribute only — it never enters the primary disease vector;
  5. no -1 (uncertain) intent is ever encoded;
  6. frontal-only, matching Stage 1's view_filter;
  7. per-label quotas oversample rare classes subject to the support rule.

EVERY accept/reject decision is written to recipe_decisions.jsonl with its reason — nothing is
silently dropped, mirroring Stage 1's preprocessing-log discipline.

Usage:
    python scripts/generate/01_sample_label_recipes.py [--namespace dev] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.labels import (  # noqa: E402
    PRIMARY_ENDPOINT_LABELS,
    normalize_label,
    patient_level_support,
)
from scripts.utils.artifact_contracts import config_sha256, current_code_identity_hash, namespace_identity, stage2_paths  # noqa: E402
from scripts.utils.manifest import get_git_commit_hash, hash_dict, sha256_file, write_json  # noqa: E402
from scripts.utils.splits import resolve_splits_dir  # noqa: E402
from scripts.utils.splits import load_split, split_provenance  # noqa: E402

DEVICE_LABEL = "Support Devices"


def load_stage2_config():
    return load_named_config("stage2_generation.yaml", "stage2")


def mine_cooccurrence(frame: pd.DataFrame, min_support_patients: int) -> dict[frozenset, int]:
    """Count how many distinct real PATIENTS exhibit each positive-label combination.

    Patient-level (not image-level) so a single patient with many studies cannot manufacture
    apparent support for a combination on their own.
    """
    combo_patients: dict[frozenset, set[str]] = {}

    for row in frame.to_dict("records"):
        positives = frozenset(
            label
            for label in PRIMARY_ENDPOINT_LABELS
            if normalize_label(row.get(label)) == 1
        )
        if not positives:
            continue
        combo_patients.setdefault(positives, set()).add(row["patient_id"])

    return {
        combo: len(patients)
        for combo, patients in combo_patients.items()
        if len(patients) >= min_support_patients
    }


def compute_label_quotas(support: pd.DataFrame, recipe_cfg) -> dict[str, int]:
    """Per-label target counts, oversampling rare labels relative to real prevalence (§3.7).

    target = base_quota * (median_prevalence / prevalence) ** rarity_exponent, clipped.
    rarity_exponent=0 reduces this to a flat base_quota for every label.
    """
    prevalence = {}
    for row in support.to_dict("records"):
        total = max(row["positive_patients"] + row["negative_patients"], 1)
        prevalence[row["label"]] = max(row["positive_patients"] / total, 1e-6)

    values = sorted(prevalence.values())
    median = values[len(values) // 2] if values else 1.0

    quotas = {}
    for label, rate in prevalence.items():
        scale = (median / rate) ** float(recipe_cfg.rarity_exponent)
        target = int(round(float(recipe_cfg.base_quota) * scale))
        quotas[label] = int(
            min(max(target, int(recipe_cfg.min_per_label)), int(recipe_cfg.max_per_label))
        )
    return quotas


def blocked_by_medical_rule(combo: frozenset, block_rules: list[list[str]]) -> str | None:
    """A block rule fires when the recipe contains ALL labels in the rule — i.e. the rule describes
    a contradictory co-presentation, not merely one suspicious label."""
    for rule in block_rules:
        rule_set = frozenset(rule)
        if rule_set and rule_set.issubset(combo):
            return f"medical_rule_block:{sorted(rule_set)}"
    return None


def build_recipes(
    frame: pd.DataFrame,
    config,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    """Return (accepted_recipes, decisions, stats). `decisions` records every candidate considered
    with an explicit accept/reject reason."""
    recipe_cfg = config.recipes
    rng = random.Random(seed)

    support = patient_level_support(frame, PRIMARY_ENDPOINT_LABELS)
    quotas = compute_label_quotas(support, recipe_cfg)
    cooccurrence = mine_cooccurrence(frame, int(recipe_cfg.min_support_patients))

    allow_rules = [frozenset(rule) for rule in recipe_cfg.medical_rule_allow]
    block_rules = [list(rule) for rule in recipe_cfg.medical_rule_block]
    max_positive = int(recipe_cfg.max_positive_labels)

    # Candidate pool: mined combinations within the size cap, plus every single label (so a rare
    # label is never unreachable just because it rarely co-occurs), plus explicit allow-rules.
    candidates: dict[frozenset, str] = {}
    for combo in cooccurrence:
        if len(combo) <= max_positive:
            candidates[combo] = "empirical_cooccurrence"
    for label in PRIMARY_ENDPOINT_LABELS:
        candidates.setdefault(frozenset({label}), "single_label_guaranteed")
    for rule in allow_rules:
        if rule:
            candidates[rule] = "medical_rule_allow"

    decisions: list[dict] = []
    accepted_combos: list[tuple[frozenset, str]] = []

    for combo, origin in sorted(candidates.items(), key=lambda item: (len(item[0]), sorted(item[0]))):
        block_reason = blocked_by_medical_rule(combo, block_rules)
        if block_reason and origin != "medical_rule_allow":
            decisions.append(
                {
                    "combination": sorted(combo),
                    "origin": origin,
                    "accepted": False,
                    "reason": block_reason,
                    "support_patients": cooccurrence.get(combo, 0),
                }
            )
            continue
        if len(combo) > max_positive:
            decisions.append(
                {
                    "combination": sorted(combo),
                    "origin": origin,
                    "accepted": False,
                    "reason": f"exceeds_max_positive_labels:{max_positive}",
                    "support_patients": cooccurrence.get(combo, 0),
                }
            )
            continue

        support_patients = cooccurrence.get(combo, 0)
        if origin == "single_label_guaranteed" and support_patients == 0:
            reason = "accepted_single_label_below_support_but_required_for_coverage"
        elif origin == "medical_rule_allow":
            reason = "accepted_medical_rule_allow"
        else:
            reason = "accepted_empirical_support"

        decisions.append(
            {
                "combination": sorted(combo),
                "origin": origin,
                "accepted": True,
                "reason": reason,
                "support_patients": support_patients,
            }
        )
        accepted_combos.append((combo, reason))

    if not accepted_combos:
        raise SystemExit(
            "No label combinations survived recipe validation — check recipes.min_support_patients "
            "and medical_rule_block in configs/stage2_generation.yaml."
        )

    singles = [c for c, _ in accepted_combos if len(c) == 1]
    multis = [c for c, _ in accepted_combos if len(c) > 1]

    total_disease_target = sum(quotas.values())
    no_finding_share = float(recipe_cfg.no_finding_proportion)
    total_target = int(round(total_disease_target / max(1.0 - no_finding_share, 1e-6)))
    no_finding_target = total_target - total_disease_target

    age_buckets = list(recipe_cfg.age_buckets)
    sexes = list(recipe_cfg.sexes)
    device_rate = float(recipe_cfg.support_devices_proportion)
    single_share = float(recipe_cfg.single_label_proportion)

    recipes: list[dict] = []
    remaining = dict(quotas)
    index = 0

    def emit(combo: frozenset) -> None:
        nonlocal index
        age = rng.choice(age_buckets)
        sex = rng.choice(sexes)
        has_device = rng.random() < device_rate
        intended = {label: (1 if label in combo else 0) for label in PRIMARY_ENDPOINT_LABELS}
        recipe_id = f"recipe_{index:07d}"
        recipes.append(
            {
                "recipe_id": recipe_id,
                "intended_label_vector": json.dumps(intended, sort_keys=True),
                "positive_labels": json.dumps(sorted(combo)),
                "num_positive_labels": len(combo),
                "is_no_finding": len(combo) == 0,
                "age_bucket_start": age,
                "sex": sex,
                "support_devices": int(has_device),
                "view": "Frontal",
            }
        )
        index += 1

    # Disease recipes: draw until every label's quota is met, respecting the single/multi mix.
    guard = 0
    max_iterations = total_disease_target * 20 + 1000
    while any(count > 0 for count in remaining.values()) and guard < max_iterations:
        guard += 1
        use_single = rng.random() < single_share or not multis
        pool = singles if use_single else multis
        # Prefer combinations that still serve an unmet quota.
        hungry = [c for c in pool if any(remaining.get(label, 0) > 0 for label in c)]
        combo = rng.choice(hungry) if hungry else rng.choice(pool)
        emit(combo)
        for label in combo:
            if label in remaining:
                remaining[label] = max(0, remaining[label] - 1)

    for _ in range(max(0, no_finding_target)):
        emit(frozenset())

    rng.shuffle(recipes)
    for position, recipe in enumerate(recipes):
        recipe["recipe_id"] = f"recipe_{position:07d}"

    stats = {
        "num_recipes": len(recipes),
        "num_disease_recipes": sum(1 for r in recipes if not r["is_no_finding"]),
        "num_no_finding_recipes": sum(1 for r in recipes if r["is_no_finding"]),
        "num_candidate_combinations": len(candidates),
        "num_accepted_combinations": len(accepted_combos),
        "num_rejected_combinations": sum(1 for d in decisions if not d["accepted"]),
        "per_label_quota": quotas,
        "per_label_emitted": {
            label: sum(
                1
                for r in recipes
                if json.loads(r["intended_label_vector"]).get(label) == 1
            )
            for label in PRIMARY_ENDPOINT_LABELS
        },
        "single_vs_multi": dict(
            Counter("single" if r["num_positive_labels"] == 1 else
                    ("no_finding" if r["is_no_finding"] else "multi") for r in recipes)
        ),
        "quota_satisfied": all(count == 0 for count in remaining.values()),
    }
    return recipes, decisions, stats


def atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temp_name, index=False)
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=None, help="Override split_namespace from config")
    parser.add_argument("--limit", type=int, default=None, help="Cap recipe count (smoke testing)")
    args = parser.parse_args()

    config = load_stage2_config()
    namespace = args.namespace or str(config.split_namespace)

    provenance = namespace_identity(namespace)
    gen_train = load_split(
        "gen_train", namespace, purpose="schema_validation", caller="01_sample_label_recipes"
    )

    recipes, decisions, stats = build_recipes(gen_train, config, seed=int(config.generation.seed))

    if args.limit is not None and len(recipes) > args.limit:
        recipes = recipes[: args.limit]
        for position, recipe in enumerate(recipes):
            recipe["recipe_id"] = f"recipe_{position:07d}"
        # Recompute every count that describes the emitted set, so the manifest can never record
        # a mix that contradicts num_recipes.
        stats["limited_to"] = args.limit
        stats["quota_satisfied"] = False  # truncation invalidates the quota guarantee
        stats["num_recipes"] = len(recipes)
        stats["num_disease_recipes"] = sum(1 for r in recipes if not r["is_no_finding"])
        stats["num_no_finding_recipes"] = sum(1 for r in recipes if r["is_no_finding"])
        stats["per_label_emitted"] = {
            label: sum(
                1 for r in recipes if json.loads(r["intended_label_vector"]).get(label) == 1
            )
            for label in PRIMARY_ENDPOINT_LABELS
        }
        stats["single_vs_multi"] = dict(
            Counter(
                "single" if r["num_positive_labels"] == 1
                else ("no_finding" if r["is_no_finding"] else "multi")
                for r in recipes
            )
        )

    paths = stage2_paths(config, namespace)
    recipes_path = paths["recipes_path"]
    decisions_path = paths["recipe_decisions_path"]

    atomic_write_csv(recipes_path, pd.DataFrame(recipes))
    atomic_write_jsonl(decisions_path, decisions)

    recipe_config_block = {
        "recipes": OmegaConf.to_container(config.recipes, resolve=True),
        "split_provenance": provenance,
    }
    manifest = {
        "stage": "stage2_recipes",
        "recipes_path": str(recipes_path),
        "decisions_path": str(decisions_path),
        "schema_version": 2,
        "config_hash": hash_dict(recipe_config_block, length=64),
        "recipe_config_sha256": config_sha256(config.recipes),
        "git_commit_hash": get_git_commit_hash(),
        "code_identity_sha256": current_code_identity_hash(),
        "split_provenance": provenance,
        "split_namespace": namespace,
        "namespace_class": provenance["namespace_class"],
        "split_manifest_hash": provenance["split_manifest_hash"],
        "source_gen_train_csv_sha256": sha256_file(resolve_splits_dir(namespace) / "gen_train.csv"),
        "recipes_csv_sha256": sha256_file(recipes_path),
        "decisions_jsonl_sha256": sha256_file(decisions_path),
        "recipe_config": recipe_config_block["recipes"],
        "stats": stats,
    }
    write_json(paths["recipes_manifest"], manifest)

    print(f"Split namespace:    {namespace} (manifest {provenance['split_manifest_hash']})", flush=True)
    print(f"Candidate combos:   {stats['num_candidate_combinations']}", flush=True)
    print(f"  accepted:         {stats['num_accepted_combinations']}", flush=True)
    print(f"  rejected:         {stats['num_rejected_combinations']}", flush=True)
    print(f"Recipes emitted:    {stats['num_recipes']}", flush=True)
    print(f"  disease:          {stats['num_disease_recipes']}", flush=True)
    print(f"  no-finding:       {stats['num_no_finding_recipes']}", flush=True)
    print(f"  mix:              {stats['single_vs_multi']}", flush=True)
    print(f"  quotas satisfied: {stats['quota_satisfied']}", flush=True)
    print(f"Wrote:              {recipes_path}", flush=True)
    print(f"Decisions log:      {decisions_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
