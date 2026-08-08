# Supervisor decision required before learned ASISM components

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
