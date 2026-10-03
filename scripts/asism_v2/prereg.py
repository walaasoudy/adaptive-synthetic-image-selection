"""The pre-registered V2 ranker configuration, frozen in code as well as in the YAML.

Approved by Walaa on 2026-10-03 (docs/asism_v2_quantity_design_check_2026-10-03.md §7 and §8) before
any real utility measurement. configs/ham10000_asism_v2_ranker.yaml is what a reader edits; FROZEN is
what was approved. load_prereg() refuses to continue when the two differ, so a value cannot change
without a change to this module, which is a dated amendment and a commit of its own.

Amendment, 2026-10-03, approved by Walaa before any real utility measurement (design check §11):
the fit runs on standardised targets with no weight decay, epoch limit 5000 and patience 200.
Before: weight decay 1e-5, epoch limit 600, patience 60, targets in the metric's own units. The
CPU dry run on planted data showed those settings stop at the epoch limit far from the planted
truth. Nothing else changed: the model, the learning rate, the seeds and every gate are as approved.

Amendment, 2026-10-03, approved by Walaa before any real utility measurement (design check §12):
within a class, images are offered in the order of their own lower bound (stopping.offer_order),
not of the point model's score. The stopping condition is unchanged.
"""
from __future__ import annotations

from omegaconf import OmegaConf

from scripts.utils.config import load_named_config

from .contracts import fingerprint
from .features import SIGNALS

CONFIG_FILE = "ham10000_asism_v2_ranker.yaml"
CONFIG_SECTION = "ham_asism_v2_ranker"

FROZEN = {
    "split_namespace": "ham-stratified-v1",
    "real_train_split": "classifier_train",
    "utility_split": "asism_tuning_heldout",
    "signals": list(SIGNALS),
    "safety": {"reject_invalid_iqa": True, "reject_near_duplicates": True},
    "supervision": {
        "metric": "macro_auroc_ovr",
        "proxy": {"resolution": 224, "max_steps": 300},
        "role_seed": 42,
        "role_fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "sizes": {"train": [125, 250, 500, 1000], "validation": [125, 250, 500], "test": [125, 250, 500]},
        "subsets": {"train": 120, "validation": 40, "test": 40},
        "design_seed": 42,
        "class_fraction_concentration": 20,
        "tilted_fraction": 0.5,
        "training_seeds": [42, 43, 44, 45, 46],
    },
    "reliability": {"min_reliability_of_subset_means": 0.80},
    "fit": {"seed": 42, "bootstrap_models": 200, "initial_lam": 0.02, "optimizer": "adam",
            "learning_rate": 0.03, "weight_decay": 0.0, "max_epochs": 5000, "patience": 200,
            "standardised_fit": True},
    "stopping": {"lower_bound_quantile": 0.05, "minimum_gain": None, "offer_order": "lower_bound",
                 "stability_fit_seeds": [42, 43, 44, 45, 46]},
    "acceptance": {"min_within_size_spearman": 0.50, "max_one_sided_p": 0.05, "permutations": 10000,
                   "permutation_seed": 42, "must_beat": "size_and_class_only",
                   "reported_baselines": ["similarity_only", "equal_weight_composite"]},
}


class PreregistrationError(RuntimeError):
    """The configuration is not the one that was approved."""


def load_prereg() -> dict:
    """-> the approved configuration plus `paths` and its sha256. Refuses any other value."""
    config = OmegaConf.to_container(load_named_config(CONFIG_FILE, CONFIG_SECTION), resolve=True)
    paths = config.pop("paths")
    if config != FROZEN:
        changed = sorted(key for key in set(config) | set(FROZEN) if config.get(key) != FROZEN.get(key))
        raise PreregistrationError(f"{CONFIG_FILE} differs from the approved configuration in {changed}")
    return {**FROZEN, "paths": paths, "prereg_sha256": fingerprint(FROZEN)}


def fit_arguments(prereg: dict, seed: int | None = None) -> dict:
    """Keyword arguments of pipeline.fit_ranker, from the approved configuration only."""
    fit = prereg["fit"]
    return {"seed": int(fit["seed"] if seed is None else seed), "bootstrap": int(fit["bootstrap_models"]),
            "max_epochs": int(fit["max_epochs"]), "patience": int(fit["patience"]),
            "learning_rate": float(fit["learning_rate"]), "weight_decay": float(fit["weight_decay"]),
            "initial_lam": float(fit["initial_lam"]),
            "standardise_targets": bool(fit["standardised_fit"])}
