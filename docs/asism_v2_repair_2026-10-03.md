# ASISM V2 repair checkpoint — 2026-10-03

Status: engineering work only. No new GPU experiment approved or launched.

## Preservation

Additive snapshot: `preservation/2026-10-03-v1-audit-01/manifest.json`.
The snapshot records 308 copied files/archives and their SHA-256 hashes, historical
Stage 2 and V1 Stage 3–5 Git archives, current HEAD archive, working diff and history.
It does not change the source artifacts. E4 is excluded deliberately.
`C:/Users/walaa/ham10000_work/followup/noise_decomposition` could not be read.
Remote image/checkpoint completeness is still unverified. This is NOT a complete
reproduction package, and no claim that V1 is fully backed up is justified yet.

## Isolated implementation

`scripts/asism_v2/features.py` requires DINO similarity, IQA, MC-dropout mutual
information and calibrated explainability. It learns normalization using only
explicit training image IDs, rejects absent members/duplicate IDs/nonfinite inputs,
and does not assign a hand-designed utility weight or direction to any signal.

`models.py` is a size-aware set-model candidate (mean, variance, log-count), not a
trained ranker. `contracts.py` provides exclusive-create outputs, split-role checks,
complete repeated-measurement grid validation, configuration/membership hashes and
prediction payload verification. These helpers do not yet protect the historical
entry points or form an integrated training pipeline.

## Validation

CPU command in the thesis environment:

    python -m pytest -q -p no:cacheprovider tests/test_asism_v2_features.py tests/test_asism_v2_repair.py

Result: 16 passed in 4.67 s outside the restricted temporary-directory sandbox.
Initial restricted run: 13 passed, 3 setup errors (Windows temp access denied).
This was not a full repository test run. Per-channel perturbation tests prove input
connectivity, not learned importance, utility or downstream benefit.

## Required next gates

1. Resolve missing preservation evidence and verify remote manifests/checkpoints.
2. Design and freeze a repeated utility instrument with independent validation,
   reliability/noise criteria and a downstream transfer check. Existing V1 labels
   must not be relabelled as repaired supervision.
3. Connect a learned ranker with train-only preprocessing. Validate on independent
   measured subsets, then compare with similarity-only and signal ablations.
4. Design threshold supervision from independently measured policy utility; do not
   call a fixed-quota order statistic a trained adaptive threshold. E4 determines
   the quantity question independently; a threshold must not silently override it.
5. Add integrated leakage, output isolation, prediction and statistical regression
   tests, then run the full suite before approving an experimental runner.
6. Present exact inputs, splits, seeds, recipe, metrics, success/failure criteria,
   multiplicity policy and GPU cost before requesting a run. None is approved here.

The user's last reported E4 status is 50/60, not independently refreshed here.
V3a remains the auxiliary model for its validated role. No protected final-evaluation
outcomes were used for new training or evaluation. Supervisor policy remains pending.
