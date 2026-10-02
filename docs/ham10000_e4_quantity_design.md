# HAM10000 E4: quantity curve. Design for pre-registration

Status: **APPROVED by Walaa, 2026-10-02 (D1–D7), before any E4 code or result.** No
number in this document changes once any E4 result exists. No GPU run has started.

## 1. Why E4 is quantity-only

The CPU audit (`docs/ham10000_e4_cpu_audit.md`) found two things.

- **Which images:** at a fixed size, the difference between selections is below the training
  noise. Same-size ICC is 0.0–0.04 under three recipes at sizes 60–180, and 0.01–0.02 at size 616.
  Re-measuring that would spend GPU to confirm a known result, so E4 does not.
- **How many images:** the effect of adding synthetic images is visible. A against B is +0.076 BA
  (t = 9.6). It was measured only at 0, 616 and 3,168 images, only with the cheap proxy, and never
  per class.

E4 asks one question: at the Stage 4 recipe, can the marginal utility of adding more synthetic
images be resolved above the noise, finely enough to support a stopping rule?

E4 does not test ranking, and it does not test per-class quantity (see §8).

## 2. Fixed inputs (FROZEN by the contract, restated)

| Item | Value |
|---|---|
| Real part of every run | `classifier_train`: 1,641 images (nv 1085, bkl 188, mel 186, bcc 87, akiec 49, df 25, vasc 21) |
| Measured on | `asism_tuning_heldout`: 1,377 images |
| Never read | `classifier_val`, `final_eval_heldout` |
| Candidate pool | `all_candidates.csv`, sha256 `bf8047b5…`: 3,168 images (df 885, vasc 736, akiec 486, bcc 400, mel 276, bkl 273, nv 112) |
| Classifier | DenseNet-121, ImageNet weights. The Stage 4 recipe: 512 px, 3,000 optimizer steps, batch 32, lr 1e-4, wd 1e-4, dropout 0.2, unweighted CE |

## 3. Proposed design (each row needs approval)

| # | Decision | Proposal | Reason |
|---|---|---|---|
| D1 | Recipe | Stage 4 recipe, **512 px / 3,000 steps** | The quantity mechanism has to decide for Stage 4. The cheap proxy failed its reliability test (ICC 0.108), and the contract requires a written reason to reuse it. |
| D2 | Draw pool | The **safe pool**: candidates passing the IQA safety flags (blurry 198, low-contrast 4, border 29, near-uniform 0; the exact union is computed in code) | Selection always starts from the safe pool. Unsafe images are never candidates. |
| D3 | Sizes | **0, 250, 500, 1,000, 2,000, all safe** (6 points, roughly doubling) | Covers the range a quantity mechanism has to choose over. The old 60–180 range is far too narrow. |
| D4 | Draws | **2 nested chains.** Each chain is one random permutation of the safe pool (draw seeds 20261003 and 20261004). Size *s* is the first *s* images of the chain. | Nesting makes each marginal step "these images plus the next ones". Two chains keep the result from depending on one draw. |
| D5 | Seeds | Classifier seeds **42–46** per chain and size. Size 0 (real only) gets **10 seeds, 42–51**. | 10 runs per size. The noise at this recipe, from Round 2, is σ ≈ 0.011 AUROC and 0.04–0.066 BA. |
| D6 | Primary metric | **macro AUROC (OvR)** on `asism_tuning_heldout`. BA, macro-F1 and accuracy reported as secondary metrics. | AUROC is the most reliable metric at this recipe (Round 2 ICC 0.346 against 0.056 for BA). BA steps in units of 1/13 for df (≈ 0.011), which is the same size as the effects being resolved. |
| D7 | Runs and cost | 10 + 5 × 2 × 5 = **60 runs** × ≈ 578 s ≈ **9.6 GPU-h ≈ $7** (RTX 5090, ≈ $0.72/h) | Run time does not depend on size, because the optimizer step count is fixed. This is well inside the $65 cap and below the $23 estimate for E4 as originally written. |

Each run's training data is the 1,641 real images plus the drawn synthetic images, with
everything else identical across runs.

## 4. Pre-registered decision rule

Every quantity below is computed from the run records alone. No threshold is chosen, rounded or
moved after any result exists.

**Data.**

- Sizes S = (s0, …, s5) = (0, 250, 500, 1000, 2000, N), where N is the size of the safe pool.
- R(s) is the set of 10 runs at size s:
  - s = 0: seeds 42–51;
  - s > 0: chains 1 and 2 × seeds 42–46.
- y(r) is run r's macro AUROC (one-vs-rest, mean over the 7 classes), computed on
  `asism_tuning_heldout` from that run's own predictions. Seeds are not averaged, and probabilities
  are not ensembled.
- U(s) is the mean of y(r) over R(s).

**Comparison of two sizes a < b.**

- D(a, b) = U(b) − U(a).
- SE = √(var_a/10 + var_b/10), with sample variances (ddof = 1).
- Welch–Satterthwaite degrees of freedom.
- The 20 runs are treated as independent: no pairing by seed or chain.

**The two tests.**

1. **Overall effect.** Δall = D(s0, s5). It passes if and only if the lower end of its two-sided
   95% Welch interval is > 0.
2. **Marginal steps beyond the first.**
   - Δk = D(sk−1, sk) for k = 2, 3, 4, 5, that is 250→500, 500→1000, 1000→2000 and 2000→N.
   - Each is tested one-sided, H0: Δk ≤ 0 against H1: Δk > 0, with a Welch t.
   - The four p-values are Holm-adjusted together at familywise α = 0.05.
   - Step k is **resolved** if and only if its Holm-adjusted p ≤ 0.05.
   - Δ1 (0→250) is reported but is not part of the rule. The overall test already covers whether
     adding synthetic images helps at all.

**Outcomes.** These are exhaustive and mutually exclusive, and are evaluated at full floating-point
precision.

| Outcome | Condition | What follows |
|---|---|---|
| **GO** | Δall passes **and** at least one of Δ2–Δ5 is resolved | Marginal utility is resolvable past the first step, so a stopping rule has a signal to stop on. Next: the E4b per-class design (§8), then the quantity mechanism. |
| **COARSE** | Δall passes **and** none of Δ2–Δ5 is resolved | Adding synthetic images helps, but the marginal utility beyond the first step cannot be resolved. No learned stopping rule. Quantity is set by a pre-written rule and reported as a limitation. |
| **NO** | Δall does not pass, whatever the steps show | At the Stage 4 recipe, quantity is not measurable either. SC2 applies to quantity, and the thesis reports this as a finding. |

**Integrity.**

- The analysis refuses an incomplete or duplicated grid, and gives no verdict on part of the grid.
- A run that crashes is re-run in the same cell, with the same seed and the same chain. No cell is
  added, replaced or dropped.
- The rule is applied once, to the primary metric. Secondary metrics and per-chain curves are
  reported (§5) and decide nothing.
- The rule, the sizes, the seeds, the metric, α and the Holm family are not revised after results,
  and no outcome is followed by "one more" run.

## 5. Reported regardless of outcome

- U(s) with a 95% interval at every size, overall and per chain.
- Within-size SD at every size, and the real-only baseline mean and SD over 10 seeds (the contract's
  multi-seed baseline).
- Per-class recall at every size. This is descriptive. It is where a per-class effect would first
  appear, and it decides nothing here.
- The secondary metrics on the same curve.

## 6. CPU checks before the GPU request

- The split files and the pool manifest match their recorded hashes.
- No `classifier_val` or `final_eval_heldout` row enters any run, enforced by the existing guard.
- The chains are reproducible from their draw seeds, nested, and contain safe-pool images only.
- The training config equals the Stage 4 config, field by field.
- The v1 artifacts and the signal artifacts are byte-unchanged.
- A dry run on a tiny subset runs end to end on the CPU.

## 7. Code scope

Only E4 is touched: one new script, one config section and tests, on their own branch. ASISM v1,
V3a, the signals and Stage 4 are not modified. The existing proxy-training functions are reused
through the same audited calls as the noise-floor rounds.

## 8. Out of scope here (E4b, only if E4 is GO)

A stopping rule has to give class-dependent counts. A total-size curve cannot do that, and
per-class curves cost about seven times as much. E4b is designed and pre-registered separately,
with its own approval, only after a GO.
