# Supervisor decision required before learned ASISM components

## Resolution status (2026-08-21)

**Option 1 below has been implemented in code**, as of commit `d14488f`: `scripts/asism/models.py`
(`SetUtilityNetwork`, `MultiSignalUtilityRankingNetwork`, `AdaptiveThresholdNetwork`), the pipeline
`scripts/asism/04_build_utility_subsets.py` through `09_finalize_learned_selection.py`, and
`scripts/asism/learned.py`. This is documented as `docs/stages2_to_5_plan.md` §4.9 (v4 revision
note) and reflected in `configs/stage3_asism.yaml` and Stage 4 conditions F/G
(`docs/stages2_to_5_plan.md` §7).

**Naming follow-up (2026-08-21).** The memo below refers to a "Multi-Objective Ranking Network".
That component is now named **Multi-Signal Utility Ranking Network** (class
`MultiSignalUtilityRankingNetwork`), because the implemented model has one output head and one
scalar target — the signals are input features, not objectives. This is consistent with, not a
change to, the memo's own closing observation that "Pareto ranking is not implemented". Terminology
only; see `docs/stages2_to_5_plan.md` §4.9 v5 revision note. The memo body below is preserved
verbatim as a historical record and still uses the old name.

**What this resolves:** the pseudo-replication concern raised below. The ranking network
is never trained on a subset-level AUROC copied onto every member image; it is distilled from
`SetUtilityNetwork`'s measured *set-level* utility only (see `learned.py`'s module docstring and
`resolve_verified_only_targets`, which admits a target only from a proxy-*measured* candidate, never
a critic guess). `AdaptiveThresholdNetwork` is trained per class only where that class clears
`min_verified_contexts_per_class` on both proxy-verified train and image-disjoint held-out evidence
(`determine_per_class_official_method`); otherwise selection falls back to
`hard_proxy_best_among_verified` or the fixed-ratio baseline — a class is never handed to the network
on thin evidence.

**What remains open:** this document's original ask was for the supervisor to select the target
*and approve the additional ASISM tuning budget* before implementation proceeded. That approval was
not recorded before this code was written — implementation proceeded ahead of the sign-off this
document itself required. Before treating the learned selector (Stage 4 condition F) as anything
more than a candidate method for the thesis, get explicit supervisor confirmation of: (a) Option 1
as the accepted target/claim, (b) the compute budget actually spent (`configs/stage3_asism.yaml` →
`compute_budget`, `verification_compute_budget`, `full_policy_verification.compute_budget`), and (c)
whether to add the matched-random control described in `docs/stages2_to_5_plan.md` §7.1 — without
it, an F-over-B result cannot separate ASISM's ranking quality from the effect of using fewer
synthetic images. This status update records what exists; it is not a substitute for that approval.

**v2 note (2026-08-21):** the weighted-score selector referred to below is no longer part of the
thesis pipeline (§4.9 / §7 v3 revision notes) — the thesis defines ASISM as the full module
including both learned components, so there is no weighted-baseline condition. The memo below is
preserved as written for the historical record.

The original decision memo is preserved below, unchanged.

---

The repository does not currently contain a scientifically defensible supervised target for an
image-level ranking network or a class-aware threshold model. The available labels say what SDXL
was prompted to generate; they do not say whether adding a particular synthetic image improves a
real-image classifier. The bounded proxy search produces a noisy *subset-level* macro-AUROC for a
small number of candidate selectors. Treating that value as if it were an image-level target would
duplicate it across every selected image, create pseudo-replication, and invent causal credit that
the experiment did not observe.

For that reason no neural model has been added under the names “Multi-Objective Ranking Network” or
“Adaptive Threshold Learning.” The current `weighted_score_baseline` remains exactly a baseline.

Two defensible alternatives are available:

1. **Pre-registered weakly supervised set-utility learning.** A small permutation-invariant network
   consumes sets of normalized ASISM vectors and predicts patient-fold proxy macro-AUROC. A
   differentiable class-aware gate supplies thresholds. Training and validation use nested,
   patient-disjoint ASISM tuning folds, and comparisons retain individual, equal-weight,
   tuned-weight, and fixed-percentile baselines. This requires many more pre-registered subset
   trials; its target is learned set utility, not ground-truth image quality.
2. **Independent image-level utility labels.** Obtain blinded clinician pairwise quality/usefulness
   judgments, or pre-register a costly marginal-contribution/influence experiment in which each
   image or small batch receives an out-of-fold classifier-utility estimate. Train a ranking loss
   and class-aware threshold model on those targets with patient/source-group separation. This has
   a clearer image-level claim but needs new annotations or substantially more GPU computation.

Minimal decision: the supervisor must choose the intended target and claim—set-level proxy utility
or independently measured image-level utility—and approve the additional ASISM tuning budget/data
collection. Alternatively, the supervisor may explicitly approve weighted-score and proxy-tuned
selection as baselines and remove the literal learned-network requirements. Pareto ranking is not
implemented. Implementation should
not proceed until that decision is frozen in a new protocol version.
