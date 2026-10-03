# Thesis audit and completion plan (HAM10000 / ASISM)

*Audit date 2026-10-02, branch `run/ham10000-e4` at `69b659b`. Sources: the code on every branch, every doc (including branch-only docs), all run files in `C:\Users\walaa\ham10000_work`, and the git history. Read-only, except for the E4-consequence code described in §15.*

---

## 1. Executive diagnosis

1. **Two questions with very different status.**
   - *How many* synthetic images matter: B − A = +0.076 BA (t 9.6, 54 seeds per arm).
   - *Which* images matter: at a fixed size the effect has not been distinguishable from training noise in any measurement.
     - Same-size ICC is 0.00–0.04.
     - σ_s is 0.0010 AUROC against σ_e 0.010.
     - About 387 repeats per subset would be needed.
2. **The root cause of the "which" failure is not label noise as such.** The selection effect is genuinely small relative to the stochasticity of the DenseNet training floor. The noise decomposition puts 83–89% of variance in a training floor that a larger evaluation set does not remove. Every learned-utility path (subset labels, set model, ranker) inherits this. More seeds or a cheaper proxy cannot fix it within any budget the thesis has.
3. **E4 is the right next experiment and is ready.** It is frozen, coded, tested and dry-run. It asks only the question that can be answered. Running it is the single GPU step that unblocks everything.
4. **Two decisions that are not technical block the end of the thesis.**
   - **(D-A)** How ASISM v2 ranks images *within* a class. No ranking of the 4 signals exists in code, and uncertainty has no predeclared direction.
   - **(D-B)** The test-set policy. `final_eval_heldout` was read once, on 2026-09-21, for v1, and every v2 design choice came after that read.

   Both should be decided **now, in parallel with E4**, and must not wait for E4's result.
5. **The defensible thesis is a framework plus a measurement result, not a "learned adaptive selector".**
   - ASISM v2 is a pre-registered, audited pipeline.
   - The quantity decision comes from a pre-registered curve.
   - The selection-quality claim is reported as a bound, with a matched random control.
   - The learned ranking network and the adaptive-threshold network should be reported as tried and retired, with the evidence, not repaired.

## 2. Current project state

| Item | State |
|---|---|
| Stage 1 LoRA (`ham-lora-v1`, 8000 steps, seed 42) | Frozen |
| Stage 2 pool | 3,168 images. Frozen. Skewed toward rare classes (df 885 … nv 112) |
| Signals | Similarity (DINOv2 ViT-S/14 kNN) and IQA (v1); uncertainty and explainability (V3a DenseNet). Passed the 4-signal Go/No-Go (max \|ρ\| 0.472) |
| Agreement | Dropped (J1 failed V4: df 50.2% ≥ 50%; SC1 triggered) |
| v1 A/B/C on the test split | Done once: A 0.566, B 0.645, C 0.612 BA; C − B = −0.033, p = 0.047 |
| E4 | Plan frozen (N = 3,168, 60 cells), dry run passed on pod `ig1fulyi4bwnvh` (stopped). **0 of 60 runs** |
| E4 consequences | V1 and V3 approved; V2 only after GO |
| ASISM v2 ranking within class | **Not specified, not implemented** |
| Stage 5 for v2 | Code is on the unmerged branch `feat/ham10000-stage5-protocol` |
| Supervisor approvals | **None recorded.** Scope (#1) and test set (#8) pending |

## 3. Repository and code architecture (HAM path)

- **Splits and preprocessing:** `scripts/data/ham10000/*`. Lesion-grouped six-way split, stratified by class, seed 42. All 30 pairwise overlaps are 0.
- **Stage 1/2:** `scripts/train/ham10000_train_lora_sdxl.py`, `scripts/generate/ham10000_*`.
- **Stage 3 signals:** `scripts/asism/ham10000_01_compute_signals.py`, `ham10000_signals.py`, `ham10000_explainability.py`.
- **Gate:** `ham10000_02_gonogo.py`.
- **v1 learned path:** `ham10000_03` (utility subsets), `ham10000_04` (Deep Sets set model → ranking MLP), `ham10000_05` and `ham10000_thresholds.py` (threshold network fit on 7 points).
- **v2 selection:** `scripts/followup/ham10000_v2_select.py` (fill-to-300; superseded).
- **Diagnostics:** `scripts/followup/*` (noise floor, instrument check, headroom, noise decomposition, E4).
- **Stage 4:** `scripts/classify/ham10000_train_conditions.py`.
  - DenseNet-121, 512 px, 3,000 steps, AdamW at a constant LR of 1e-4.
  - No augmentation, no class weighting, last checkpoint.
- **Stage 5:** `scripts/eval/ham10000_stage5_evaluate.py` (v1 conditions hard-coded) and `ham10000_compare_conditions.py` (lesion bootstrap, Holm/BH).

Hazards found:
1. **The Go/No-Go overwrites the v1 report.** `ham10000_02_gonogo.run()` always writes `stage3/<ns>/gonogo_report.json`, even with `--scores-dir`. That file is read by `load_candidate_pool`, and therefore by E4. The E4 pool hash was verified on the pod, so E4 is unaffected. Any re-run of the gate on the pod must use a separate PROJECT_ROOT.
2. **`load_candidate_pool` reads only the v1 signal roots.** A v2 selection script has to point to the 4-signal set explicitly.
3. **Metric mismatch.** E4 decides on macro AUROC; Stage 5's primary metric is BA. This is a design fact to state, not a bug.
4. **The contract is stale.** `docs/ham10000_v2_experimental_contract.md` predates A6/A7 and E4.
5. **The v1 "budget cap" is 2,000, not the class target** (`ham10000_05:290-298`). This matters only for describing v1.
6. **Training is not deterministic.** `cudnn.deterministic` is never set, so results reproduce as distributions. This is accepted in the contract.

## 4. Complete experiment history (condensed)

| # | Experiment | Question | Result | Established | Not established | Repeat? |
|---|---|---|---|---|---|---|
| E-1 | LoRA (seed 42, n = 1) | Adapt SDXL | Working generator; val loss flat after about 6.5k steps | 3,168 valid candidates | Class fidelity (bkl ≈ 0% under every judge; variety −0.094) | No |
| E-2 | Snapshot pilot (35 images) | Pick a snapshot | final = 14/35 agreement; mel/bkl/df 0/5 | Choice was documented | That `final` is best (circular judge) | No |
| E-3 | v1 A/B/C, Stage 5 (3 seeds) | C > B? | C − B −0.033 [−0.063, −0.001] | Synthetic helps (B > A, C > A) | Anything about selection: confounded by size (616 vs 3,168) and composition | **Never on this split** |
| E-4 | Noise floor R1–R3 | Are utility labels reliable? | ICC ≤ 0.35 (best: R3 AUROC, 8 repeats); BA ICC ≈ 0–0.14 | Single-run labels ≈ 86% noise | That learned utility is impossible in principle | No |
| E-5 | Baseline correction | Was the v1 baseline right? | Single run +0.051 too high | Multi-seed baselines required | — | No |
| E-6 | Instrument check (240 runs) | Cheap proxy reliable and valid? | ICC 0.108 (fails R); ρ 0.755 with R3 (passes A) | The proxy *ranks* like Stage 4 but compresses spread 2.6× | — | No |
| E-7 | Selection headroom (268 runs) | Does *which* 616 matter? | σ_s 0.0061 BA / 0.0010 AUROC; C z = −0.28 | v1 C is an ordinary random draw; quantity explains B > C | Headroom at the Stage 4 recipe (≈0.016 by scaling) | No |
| E-8 | Noise decomposition (24 runs) | Does a bigger evaluation set help? | Floor = 83–89% of variance; repeats 381 → 326 at 4× evaluation | Evaluation size is not the lever | Separation of training and evaluation noise | No |
| E-9 | Small-budget prereg (N = 150) | Same-size signal at small N | **Not run** (prediction: 2% pass) | — | — | No — answered in direction by E-7/E-18 |
| E-10 | v2 fill-to-300 design | — | Never executed; superseded | Principle "C vs matched random D" | — | No |
| E-11 | Missing augmentation | — | Stage 4 has no augmentation or LR schedule | Limitation, equal across conditions | — | No (frozen) |
| E-12 | v2 aux classifier + Q1–Q5 | Is the judge sensible? | mel→nv 99.6%; Q1 fails | DenseNet judges misread the pool | — | No |
| E-13 | Separability S1–S3 | Is the pool degenerate? | Mixed: more separable, less varied | Synthetic variety is lower | — | No |
| E-14 | V3a (checkpoint via CV) | Fix the judge? | classifier_val BA 0.582; mel→nv 84.8% | Training length is not the cause; V3a frozen for uncertainty and explainability | — | No |
| E-15 | P1 / P1b DINOv2 probes | Is the nv sink DenseNet-specific? | Yes (mel→nv 12–14%), but P1b has its own df sink | Classifier-specific sink | That the images are class-faithful | No |
| E-16 | J1 judge | Usable agreement judge? | Fails V4 by 0.2 points | Agreement leaves v2 | — | **Forbidden (SC1)** |
| E-17 | 4-signal Go/No-Go | Gate the 4 signals | All 4 included, max \|ρ\| 0.472 | Signals are valid, non-redundant inputs | That they predict utility | No |
| E-18 | E4 CPU audit | Can E4 as written work? | Selection not measurable; quantity visible | Redesign to quantity only | — | No |
| E-19/20/21 | E4 design, dry run, consequences | How many? | Frozen, dry run passed, V1/V3 approved | — | **No result yet** | **Run once** |

## 5. Noise taxonomy

| Source | Real? | Evidence and magnitude | Root cause | Reducible? | Addressed? | Clean solution | CPU/GPU/decision |
|---|---|---|---|---|---|---|---|
| **Dataset:** imbalance | Yes | nv 67%; df 73 lesions; 1 df image = 0.011 BA on tuning, 0.05 recall on test | Data | Estimate only | Stratified splits, BA, AUROC | Report per-class with CIs; AUROC as the primary metric for E4 | — |
| Dataset: leakage | Low | 0 overlaps; lesion-grouped; patient id unavailable; near-duplicates across lesion ids unchecked | Data | Partly | Yes, except pixel-level near-duplicates | State as a limitation | CPU (optional) |
| Dataset: label ambiguity (bkl/mel/nv) | Yes | Every judge confuses bkl | Clinical | No | — | Limitation | — |
| Dataset: synthetic–real domain shift | Yes | S2 −0.094; bkl unrecognised; colour a\* shift | Generator | Not without a new generator | Measured | Limitation; do not retrain the LoRA | — |
| **Generator:** LoRA / seed variability | Unmeasured | n = 1 LoRA | Single training | Only with new LoRA runs | No | Out of scope; limitation | GPU (not recommended) |
| Generator: class failures | Yes | bkl 0–3.7% under all judges | Generator | No | Measured | Report; E4 sees it as part of quantity | — |
| **Signal:** IQA nearly constant | Yes | 0.96–0.99 | Clean pool | — | Gate passed | Treat IQA as a safety filter, not a ranker | Decision (D-A) |
| Signal: uncertainty direction | Yes | No predeclared direction | Design gap | — | No | Predeclare a direction or exclude it from ranking | **Decision (D-A)** |
| Signal: redundancy | No | max \|ρ\| 0.472 | — | — | Yes | — | — |
| Signal: model dependence | Yes | Uncertainty and explainability come from V3a, whose sink is nv | Judge | — | Documented | Limitation | — |
| **Judge:** DenseNet nv sink | Yes | mel→nv 84.8–99.6% | Imbalanced prior | — | Agreement removed | Closed (SC1) | — |
| Judge: DINOv2 probe df sink | Yes | P1b 794/1,435; J1 50.2% | Probe head and weighting | — | Closed | Closed (SC1); do not reopen | — |
| **Utility label** (A) | Yes | ICC 0.0–0.35; v1 labels ρ −0.60 vs R3 | σ_s ≪ σ_e | No, within budget | Yes (SC2 for learned utility) | Retire learned utility | — |
| **Classifier training noise** (B) | Yes, dominant | BA SD 0.066 (A, Stage 4, 4 seeds); AUROC SD ≈ 0.009–0.011; floor 83–89% | Non-deterministic SGD at small data | Only with many seeds | Measured | Many seeds in the final A/B/C/D; AUROC primary | GPU |
| **Synthetic image quality noise** (C) | Yes | See generator rows | Generator | No | — | Limitation | — |
| **Judge/model noise** (D) | Yes | See judge rows | — | — | Closed | — | — |
| **Quantity-effect noise** (E) | Yes, measurable | SE ≈ 0.0049 per size at 10 runs; MDE ≈ 0.015 AUROC | Training noise | Fixed by design | E4 design | Run E4 as frozen | GPU (E4) |
| **Ranking/selection noise** (F) | Yes | σ_s 0.0010 AUROC at 616 | Small true effect | No | Measured | Report a bound on \|C − D\|, not a win | GPU (final) |
| **Evaluation noise** (G) | Yes, minor | Floor dominates; evaluation size is not the lever | — | No | Measured | Lesion bootstrap plus seed variance in the final CIs | CPU |

The noise sources stay separate. E4 addresses only (E). The final C vs D addresses only (F). Seeds in the final run address (B). Nothing addresses (A): it is retired, not fixed.

## 6. Root causes (ranked by dependency)

1. **The true selection effect is small** relative to (2). The observation is σ_s ≈ 0.001 AUROC.
   - *Consequence:* utility labels, the ranker and any C vs D test at a feasible seed count cannot resolve "which".
   - *Fix:* none within budget. Change the claim to a bound.
   - *Validation:* the final C vs D CI.
2. **Stochastic training floor** (83–89% of variance).
   - *Consequence:* inflates every comparison.
   - *Fix:* seeds (only for comparisons that matter); AUROC as primary.
3. **Size and composition confounds in v1** (616 vs 3,168; nv share 50% vs 25%).
   - *Consequence:* v1's C − B says nothing about selection.
   - *Fix:* already applied — matched random D, and quantity separated out into E4.
4. **The generator's class fidelity is limited** (bkl, low variety). This caps the possible headroom of any selector. It is a limitation, not fixable without a new generator.
5. **Process root cause:** the learned components (ranker, threshold net) were built before the instrument they depend on was validated. This is already corrected by the pre-registration discipline.

## 7. Already solved (do not repeat)

- Split leakage, the baseline artifact, judge validity (SC1), signal redundancy (the \|ρ\| fix), whether evaluation size is a lever, whether the proxy agrees with Stage 4, and selection headroom at 616.
- None of E-1 to E-18 should be repeated.

## 8. Remaining unresolved problems

| # | Problem | Kind |
|---|---|---|
| R1 | Quantity answer (E4 result) | GPU, ready |
| R2 | Within-class ranking of the 4 signals: definition and the direction of uncertainty | **Scientific decision (D-A)** |
| R3 | Test-set policy | **Supervisor decision (D-B)** |
| R4 | Number of Stage 4 seeds for the final run | Power calculation (CPU), then decision |
| R5 | v2 Stage 5 entry point (unmerged branch) and C/D builders from q\* + V3 | CPU code |
| R6 | Stale contract; Go/No-Go overwrite hazard | CPU docs/code |

## 9. "WHICH" diagnosis

- **Known:** at 616 images, v1's learned selection is indistinguishable from random (z −0.28). Between-subset SD at a fixed size is 0.0061 BA / 0.0010 AUROC, against within-subset SD of 0.04 / 0.01.
- **Unresolved:** whether the *4-signal* ranking (never built) beats random at q\*.
- **Is the utility-learning approach fundamentally too noisy?** Yes at this pool, generator and recipe.
  - Labels would need ≈387 repeats per subset.
  - SC2 should be formally invoked for learned utility. The evidence already meets its wording.
- **Is a direct matched C vs D test scientifically justified?** Yes. It is the contract's confirmatory comparison and needs no utility labels. But it must be pre-registered with an honest power statement:
  - With k seeds per arm and σ ≈ 0.010 AUROC, SE ≈ 0.010·√(2/k).
  - At k = 10 the MDE is ≈ 0.013 AUROC, an order of magnitude above the plausible effect (≈0.001–0.003).
  - So the expected and acceptable outcome is "not resolved". The thesis reports the 95% CI of C − D as an upper bound on the selection benefit.
  - Several draws of D (for example 3 draws) are preferable to one, so that D's own selection variance is represented.
- **Special case:** if q\* = N (or close to it), C ≡ B ≡ D and there is nothing to compare. The thesis then states that the selection question is moot at the recommended count.

## 10. "HOW MANY" diagnosis (E4, unchanged)

- E4 genuinely answers: "at the Stage 4 recipe, on `asism_tuning_heldout`, at **pool composition**, is there a measurable gain from synthetic images, and is any step beyond 250 resolvable?"
- **Caveat to state:** pool composition is rare-class-heavy, so the curve measures quantity *and* rebalancing together. V3 is consistent with this, because counts are split at the composition E4 measured.

| Outcome | Meaning | q\* | Consequence |
|---|---|---|---|
| **NO** | No evidence that U(N) > U(0) | 0 | C = A; ASISM's count is "none"; SC2 for quantity; A vs B still reported |
| **COARSE** | Total gain, but no single step resolved after Holm | Smallest size whose L(s) ≤ 0 | C = top q\* per V3 split; D = random at the same per-class counts |
| **GO** | At least one step beyond 250 resolved | Reported | A separate E4b design is needed before any count (not designed; requires approval) |

- The power limit is stated: shortfalls under ≈0.015 AUROC go undetected, so q\* is "not shown worse", never "equivalent".

## 11. Original vs current ASISM

| Original component | Current | Why | Justified? | Weakens contribution? | Repair? |
|---|---|---|---|---|---|
| CLIP/DINOv2 similarity | DINOv2 ViT-S/14 kNN to gen_train | DINOv2 is recommended over Inception/CLIP for medical images (Stein et al. 2023) | Yes | No | No |
| IQA | Rule-based composite; acts as a safety filter | Pool is clean (0.96–0.99) | Yes | Slightly | No |
| Uncertainty: MC dropout / deep ensembles | MC dropout, head only (one Dropout 0.2), V3a | Cost; ensembles never built | Partly | Slightly | Not necessary; state it |
| Grad-CAM / Score-CAM | Grad-CAM peripheral-mass typicality | Score-CAM never built | Yes | Slightly | No |
| Agreement (classifier judge) | Removed | Every judge failed (SC1) | **Yes, strongly** | It becomes a finding | No — forbidden |
| Multi-objective ranking network (novel) | v1: learned, fails validation (ρ −0.36); v2: none | Utility labels below noise | Yes | **Yes**: the novel learned part is not supported | Not repairable within budget; retire with evidence |
| Adaptive threshold learning (novel) | v1: 7-point regression (LOCO residual 0.321) → retired; v2: E4 + V1 + V3 | Predeclared quantity curve | Yes | **Changes the claim** | Not necessary |

**"Adaptive" vs "predeclared after E4":**
- An adaptive learned quantity would support the claim "ASISM learns how many images each class needs". The evidence cannot support that.
- The predeclared E4 rule supports: "ASISM sets its count from a pre-registered quantity experiment, and the count is the smallest tested size not shown worse than all N". That is a weaker but defensible and reproducible claim.
- Under GO, an adaptive per-class rule becomes *possible* (E4b). It is not presumed.

## 12. Scientific risks

1. **The test split is already informed by v1** (D-B). This is the largest validity risk.
2. **C vs D will very likely not resolve.** It must be pre-registered as a bound, or it will read as a failure.
3. **q\* near N makes selection moot.**
4. **AUROC (E4) and BA (Stage 5) may disagree.**
5. **Temptation to add E4 runs after the verdict.** Forbidden by §7 of the consequences doc.
6. **The unmerged Stage 5 v2 code path is untested on the run line.**

## 13. Proposed final ASISM

- **Validated:**
  - safety gate (IQA validity, near-duplicates at 0.95);
  - the 4 signals as gated inputs;
  - the E4-derived count q\* (after E4);
  - V3 class allocation (predeclared).
- **Needs validation:** the within-class ranking. Validated only through the final C vs D, as a bound.
- **Novel:** the audited, pre-registered selection pipeline, with a judge-validation protocol (SC1) and a quantity-curve rule. The finding that classifier judges sink synthetic minority images into nv is, per the literature notes, not previously reported for dermoscopy.
- **Fallback:** under NO, ASISM adds nothing (C = A). Under GO, use q\* until E4b is approved.
- **Removed:** the agreement signal, the learned set-utility model, the ranking network, the adaptive threshold network, the P25 floor, fill-to-300, and the 50/2000 bounds.
- **Ranking (D-A, needs a decision before E4's result):**
  - *Recommended:* a non-learned, predeclared within-class score — the mean of within-class percentile ranks of similarity (higher = better), explainability typicality (higher = better) and uncertainty (direction to be declared), with IQA as a gate only.
  - *Alternative:* similarity rank alone.
  - Either must be written down before any C is built, and before the E4 result is seen if possible.

## 14. Proposed final experiment sequence (minimum)

1. **E4 measure + analyze** (frozen; 60 runs). Already designed — run once.
2. **CPU:**
   - compute q\* and the V3 counts (code ready, §15);
   - write the decisions D-A, D-B and the seed count;
   - build C and D manifests;
   - merge the Stage 5 v2 path;
   - fix the Go/No-Go overwrite hazard;
   - update the contract.
3. **Final Stage 4 A/B/C/D** at the Stage 4 recipe, k seeds each (k from the power calculation; recommended 10). Skipped under NO, where C = A and D does not exist, so only A and B are run.
4. **Stage 5 once** on the split chosen by D-B.

Never repeated: anything in §7, any E4 cell after the verdict, the judges, the utility labels.

Combinable: A and B in the final run reuse the same recipe. E4's size-0 and size-N runs are *not* reused as final A/B, because those are measured on `asism_tuning_heldout`.

## 15. CPU work

- **Done in this session:** `scripts/followup/ham10000_e4_consequences.py` plus `tests/test_ham10000_e4_consequences.py` (V1 q\*, V3 largest remainder, per-verdict counts; no ranking).
- **Pending, after the decisions:**
  - C/D builders;
  - Stage 5 v2 merge;
  - Go/No-Go `--out` fix;
  - contract refresh;
  - power calculation for k;
  - ECE/Brier (contract §12) if wanted.

## 16. GPU work

| Step | Runs | Time | Cost (RTX PRO 4500, ≈ $0.7/h) |
|---|---|---|---|
| E4 | 60 × ≈ 580 s | ≈ 9.6 h | ≈ $7 |
| Final A/B/C/D (k = 10) | 40 × ≈ 600 s | ≈ 6.7 h | ≈ $5 |
| Stage 5 predictions | 40 inferences | < 0.5 h | < $1 |
| **Total** | | **≈ 17 h** | **≈ $13 (cap $65)** |

## 17. Statistical plan

- **E4:** as frozen (Welch, Holm, GO/COARSE/NO; V1 q\*).
- **Final evaluation:**
  - Confirmatory: C vs D, a single test. Two-sided 95% CI of the difference, from a **seed × lesion bootstrap** (resample seeds within condition and lesions jointly), so that training variance is inside the interval. Today's comparison resamples lesions only.
  - Secondary: B − A, C − A, B − C, Benjamini–Hochberg corrected. Per-class recall with CIs.
  - Report the AUROC and the BA columns side by side.
- **Power:** state the MDE for k before the run.

## 18. Final A/B/C/D design

| Condition | Training set | Notes |
|---|---|---|
| A | classifier_train | k seeds |
| B | + all 3,168 safe synthetic | k seeds |
| C | + top-q\* by the D-A ranking, V3 per-class counts | k seeds |
| D | + random within class at C's counts | k seeds; preferably r = 3 independent draws × k/… seeds, fixed before the run |

- Identical recipe (Stage 4 YAML, unchanged).
- C and D are matched in size and class distribution. B is the unmatched quantity control.
- Selection manifests are hashed before any training.

## 19. Test-set policy (decision D-B, supervisor)

- **History:**
  - `final_eval_heldout` was read once for outcomes, run `ham-final-v1`, 2026-09-21T17:29Z: A/B/C × 3 seeds.
  - That result motivated the v2 design (matched D, fill-to-300, headroom reference 0.0165).
  - No later experiment read it (verified in the run files).
- **It is no longer pristine for "is ASISM better than B?"** It is still unused for C vs D and for any v2 condition.
- **Options:**
  - (a) Reuse it once more, disclosing the history. Defensible for C vs D, because no v2 condition was ever scored on it.
  - (b) Carve a new holdout from an unused split before any v2 result. There is no unused split; `asism_tuning_heldout` is used by E4.
- **Recommendation:** (a), with the disclosure written before Stage 5. The supervisor must sign it.

## 20. Exact dependency order

```
CURRENT STATE (E4 frozen, 0/60 runs)
 ├─► E4 measure (GPU) ─► E4 analyze + q*/V3 (CPU) ─┐
 └─► Decisions in parallel, BEFORE E4's result:    │
       D-A within-class ranking + uncertainty sign │
       D-B test-set policy (supervisor)            │
       k seeds (power calc)                        │
       SC2 formally invoked for learned utility    │
                                                   ▼
                     Build C/D manifests (CPU; skipped under NO)
                                                   ▼
                     ASISM v2 freeze (contract refresh, hashes)
                                                   ▼
                     Final Stage 4 A/B/C/D (GPU)
                                                   ▼
                     Stage 5 once (split per D-B)
                                                   ▼
                     Thesis results
```

This changes the requested graph in two places:
- "Selection validation" is not a separate experiment before the freeze. It *is* the final C vs D, because a standalone selection experiment would be the same comparison at a different time.
- The decisions run in parallel with E4, and must be fixed before E4's result, so that the ranking cannot be chosen knowing q\*.

## 21. Expected thesis claims

- Synthetic augmentation from a LoRA-adapted SDXL improves HAM10000 classification over real data alone (B > A; v1 test, plus the headroom control).
- ASISM's count comes from a pre-registered quantity curve: q\*, "not shown worse than N", with CI.
- At the recommended count, ASISM's 4-signal selection is or is not distinguishable from matched random selection; the CI bounds the benefit.
- Methodological findings:
  - Learned utility labels for synthetic-subset selection are noise-dominated (ICC ≤ 0.35).
  - DenseNet judges sink synthetic minority classes into nv, while DINOv2 probes have their own sinks.
  - The training floor, not evaluation size, dominates the noise.

## 22. Claims that must NOT be made

- "ASISM learns which images are useful" or "learned adaptive thresholds".
- "q\* is equivalent to N."
- "ASISM beats random selection", unless the final CI excludes 0.
- Any v1 C − B result read as a selection effect.
- That the test split is untouched.
- Adaptive per-class stopping, unless E4 = GO **and** E4b is designed, approved and run.

## 23. Stop conditions

- **SC1:** triggered; no new judges.
- **SC2:** invoke for learned utility now; for quantity only if E4 = NO.
- **E4:** one grid; technical re-runs of the same cell only.
- **Final:** one A/B/C/D grid and one Stage 5 read. No condition is added after any result.
- **Budget:** stop at $65 total.

## 24. Remaining compute and cost

About 17 GPU-hours, about $13, plus CPU work of 1–2 days, mostly decisions and docs.

## 25. Final recommended path

1. Run E4 now.
2. In parallel, take D-A, D-B, k and the SC2 note to the supervisor and freeze them in writing.
3. Apply V1/V3 mechanically.
4. Run one final A/B/C/D with k seeds and one Stage 5 read.
5. Write up a framework-plus-measurement thesis whose selection result is a bound.

No further diagnostic experiments are needed.
