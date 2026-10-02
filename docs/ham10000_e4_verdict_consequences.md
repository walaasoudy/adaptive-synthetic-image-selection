# HAM10000 E4: what each verdict does to ASISM v2. Decision protocol, written before any E4 result

Status: **APPROVED by Walaa, 2026-10-02, before any of the 60 E4 runs.** V1 and V3 are approved as
written. V2 is approved only as a condition for the future (see §3).
This is written after the E4 dry run (2026-10-02) and before any of the 60 runs. No E4 result exists.
Once one does, nothing here changes.

**This is a decision protocol only.** It changes no code. The ASISM implementation, the signals,
the ranking, E4 and Stage 4 are not modified by this document. Any code that applies these
decisions comes later, with its own plan and approval.

The document fills §10 ("Selection and quantity") and the count of condition D in §11 of
`docs/ham10000_v2_experimental_contract.md`, for each outcome of the rule in
`docs/ham10000_e4_quantity_design.md` §4.

## 1. What E4 decides, and what it does not

ASISM v2 has two jobs (contract §10):

| Job | Decided by | Touched by E4? |
|---|---|---|
| **Which** images: the safety filter, then a ranking within each class | the 4 signals that passed the gate (similarity, IQA, uncertainty, explainability; `docs/ham10000_v2_gonogo_four_signals.md`) | **No.** Under every verdict, the ranking stays as the gate admitted it. |
| **How many** images | the quantity policy below | **Yes.** |

Two facts fixed before E4 shape the decisions:

- **N = 3,168.** The safety gate removed 0 images (frozen plan, `ids_sha256 d71b3013…`), so every
  candidate is safe. A count of N makes ASISM's set identical to condition B, and leaves the
  selection comparison C vs D with nothing to compare.
- **Selection at a fixed size is below the noise** (`docs/ham10000_e4_cpu_audit.md`: same-size ICC
  0.0–0.04). C vs D is still the contract's confirmatory comparison. Written now: it may not
  resolve. If so, that is a finding about selection, not a failure of the quantity policy.

## 2. V1. Smallest size not shown to be worse than N

**Status: APPROVED as written (2026-10-02).**

**Inputs.** The 60 frozen E4 runs in `e4_runs.jsonl` and nothing else. All quantities are defined
exactly as in E4 §4:

- S = (0, 250, 500, 1000, 2000, N), with N = 3,168.
- y(r) is run r's macro AUROC (OvR) on `asism_tuning_heldout`.
- U(s) is the mean of y over the 10 runs at size s.
- Sample variances use ddof = 1.

**Test, for each s in S other than N.**

- D(s) = U(N) − U(s).
- SE(s) = √(var_N/10 + var_s/10).
- ν(s) is the Welch–Satterthwaite degrees of freedom.
- The 20 runs are treated as independent: no pairing by seed or chain.
- L(s) = D(s) − t₀.₉₇₅,ν(s) · SE(s), the lower end of the two-sided 95% Welch interval on
  U(N) − U(s).
- Size s is **shown worse than N** if and only if L(s) > 0. This is the same as rejecting
  H0: U(N) − U(s) ≤ 0 in a one-sided Welch test at α = 0.025.
- N itself is never shown worse than N.

**Rule.**

> q\* = the smallest s in S that is **not** shown worse than N.

q\* always exists, because N qualifies.

**What q\* means, and what it does not.**

- q\* is the smallest tested size for which **E4 gives no evidence, at the 95% two-sided level,
  that it is worse than using all N images.**
- It is **not** a finding that q\* is as good as N. No equivalence or non-inferiority test is done,
  and no equivalence margin is set.
- With 10 runs per size and a within-size SD of about 0.011 AUROC (E4 design D5), SE ≈ 0.0049.
  A true shortfall smaller than about 0.015 AUROC has less than an 80% chance of being detected.
- The report gives L(q\*), D(q\*) and the full interval next to the count, and calls q\* "not shown
  worse than N", never "equivalent to N".

**Fixed choices, with their reasons.**

| Choice | Reason |
|---|---|
| α = 0.025 one-sided (two-sided 95%) | It is E4's own Δall criterion, so no new number is introduced. It also makes V1 agree with the verdict by construction: Δall passes exactly when L(0) > 0. |
| No multiplicity correction across the five tests | A correction would make "shown worse" harder to reach, and so push q\* lower. Uncorrected tests are the conservative direction: they lean toward more images. |
| "Smallest", even on a curve that is not monotone | The rule is taken as written. If a smaller size is not shown worse while a larger one is, q\* is still the smaller one. The full curve is reported next to it. |

**Applied once.**

- q\* is computed once, from the complete 60-run grid, by the same analysis that computes the
  verdict.
- It is computed at full floating-point precision.
- It is not recomputed with other metrics, other α, other sizes or extra runs.
- No size is added between the tested sizes. q\* is always one of the six sizes in S.

**Consequence under each verdict.** This follows from the definitions; it is not an extra rule.

- **NO:** L(0) ≤ 0, so q\* = 0.
- **GO or COARSE:** L(0) > 0, so q\* ≥ 250.

## 3. V2. A per-class stopping rule, only after a GO

**Status: APPROVED CONDITIONALLY / FUTURE, only if E4 = GO (2026-10-02).** Only the condition is
approved. No E4b stopping rule is designed, implemented or committed now.

- No adaptive stopping rule is designed, approved or implemented now.
- If, and only if, E4 returns **GO**:
  1. E4b (per-class quantity) is designed in its own document.
  2. That design is approved as a separate step.
  3. Only then can an adaptive stopping rule be implemented.
- This document commits to no detail of that rule: not the criterion, the batch size, the number
  of runs, the per-class handling, or how E4's measurements are used to size it.
- Under GO, q\* (V1) is still computed and reported. Whether it serves as an interim or fallback
  count is decided with E4b, not here.
- Under **COARSE** or **NO**, there is no E4b and no adaptive stopping rule. The count is q\*.

## 4. V3. Splitting a total count across classes

**Status: APPROVED as written (2026-10-02), as the predeclared class-allocation policy.** It is not
tuned or replaced after any E4 result.

When a total count q is used (q = q\* under COARSE; under GO only if E4b decides a total count is
used):

- For each class c, the base share is q × n_c / N, where n_c is class c's count in the safe pool:
  df 885, vasc 736, akiec 486, bcc 400, mel 276, bkl 273, nv 112.
- Each class first gets ⌊q × n_c / N⌋ images.
- The remaining images go one at a time to the classes with the largest fractional parts, with
  ties broken by class name in alphabetical order.
- Within each class, the top images in ASISM's existing ranking are taken.
- Condition D draws at random within each class, at the same per-class counts.

Why this split: E4 measured utility at pool proportions, because its chains are uniform
permutations of the pool. This is the composition at which the count was measured.

The split is fixed now, as part of the predeclared policy. It is **not** a parameter to adjust after
E4: no other split is tried, compared, or chosen on any result.

## 5. Per verdict, in one table

| Verdict | ASISM v2 count (condition C) | Condition D | Statement in the thesis |
|---|---|---|---|
| **GO** | Not fixed here. It is decided by the E4b protocol (V2). q\* is reported. | Per E4b | The marginal utility is resolvable past the first step. A per-class rule is designed separately. |
| **COARSE** | q\* (V1), split by V3 | Random draws at C's per-class counts | Synthetic images help in total, but the marginal step cannot be resolved. The count is the smallest tested size not shown worse than N. Reported as a limitation. |
| **NO** | q\* = 0: ASISM adds no synthetic image, so C = A | Not built, because there is no count to match | At the Stage 4 recipe, adding synthetic images has no measurable effect on `asism_tuning_heldout`. ASISM's quantity answer is "none", and SC2 applies to quantity. A vs B is still run and reported. |

## 6. Not decided here

- The test-set policy (contract §12; owner: the supervisor).
- The number of Stage 4 seeds (contract §11, from a power calculation).
- The ECE bin count.
- Everything about E4b (V2).

## 7. Integrity

- No verdict is followed by an extra E4 run.
- q\* is computed only from the 60 frozen runs.
- This document is committed before the GPU run starts. Comparing the commit time with the
  start time of the first run in `e4_runs.jsonl` shows the order.
