# HAM10000 E4: what each verdict does to ASISM v2. Written before any E4 result

Status: **DRAFT. Each V row needs Walaa's approval.** This is written after the E4 dry run
(2026-10-02) and before any of the 60 runs. No E4 result exists. Once one does, nothing here changes.

This document fills §10 ("Selection and quantity") and the count of condition D in §11 of
`docs/ham10000_v2_experimental_contract.md`, for each of the three outcomes of the rule in
`docs/ham10000_e4_quantity_design.md` §4. It changes nothing in E4 itself.

## 1. What E4 decides, and what it does not

ASISM v2 has two jobs (contract §10):

| Job | Decided by | Touched by E4? |
|---|---|---|
| **Which** images: safety filter, then rank within class | the 4 signals that passed the gate (similarity, IQA, uncertainty, explainability; `docs/ham10000_v2_gonogo_four_signals.md`) | **No.** Ranking stays as the gate admitted it under every verdict. |
| **How many** images per class | the quantity rule | **Yes.** This document fixes the rule for each verdict. |

So ASISM v2 is used under every verdict: it filters and ranks under all three. E4 only decides
how the count is set.

Two facts fixed before E4 shape the choices below:

- **N = 3,168.** The safety gate removed 0 images (frozen plan, `ids_sha256 d71b3013…`), so
  "every safe image" is every candidate. Taking all of them makes ASISM's set identical to B, and
  the selection comparison C vs D has nothing to compare.
- **Selection at a fixed size is below the noise** (`docs/ham10000_e4_cpu_audit.md`: same-size ICC
  0.0–0.04). C vs D is still the contract's confirmatory comparison. The expected result, stated
  now, is that it may not resolve. That would be a finding about selection, not a failure of the
  quantity rule.

## 2. The one rule shared by COARSE and NO: smallest sufficient size

**V1 (proposed).** For any verdict that does not lead to a learned stopping rule, the count is

> q\* = the smallest s in S = (0, 250, 500, 1000, 2000, N) such that U(s) is **not** significantly
> below U(N).

Here "significantly below" means the lower end of the two-sided 95% Welch interval on
U(N) − U(s) is > 0. This is the same test, the same runs and the same metric (macro AUROC on
`asism_tuning_heldout`) as Δall in E4 §4. It is computed once, from `e4_runs.jsonl`, by the same
code.

Why this rule:

- It needs no new number. α and the interval are E4's own.
- It is consistent with the verdict by construction:
  - Under **COARSE** and **GO**, Δall passed, so s = 0 is significantly below N and q\* ≥ 250.
  - Under **NO**, Δall failed, so s = 0 is not significantly below N and q\* = 0.
- It prefers fewer images when more images cannot be shown to help. That gives C vs D something
  to compare whenever q\* < N.

Known limitation, written now: "not significantly below" is not "equivalent to". With 10 runs per
size and σ ≈ 0.011 AUROC, a difference of about 0.01 can go undetected. The report gives the
interval at q\* next to the count.

Rejected alternative: **q = N** under COARSE. It makes C identical to B (see §1) and removes the
selection comparison. It stays in the report as the B arm.

## 3. Per verdict

| Verdict | Count rule for ASISM v2 (condition C) | Condition D | What the thesis says |
|---|---|---|---|
| **GO** | **V2.** The per-class stopping rule of contract §10: add images in rank order while the lower 95% bound of the marginal utility is > 0. Its batch size, runs per step and classes come from E4b, a separate design (E4 §8) sized from E4's measured within-size SD, approved before any E4b run. Until E4b is approved and complete, q\* from §2 is the fallback, and it is the count used if E4b is not run. | random draws at C's per-class counts | Quantity is measurable past the first step; ASISM decides how many per class. |
| **COARSE** | **q\* from §2**, split by class as in V3 | random draws at C's per-class counts | Synthetic images help in total. The marginal step cannot be resolved, so the count is the smallest sufficient tested size, not a learned stopping point. Reported as a limitation. |
| **NO** | **q\* = 0** (from §2): ASISM adds no synthetic image, so C = A | not built (no count to match) | At the Stage 4 recipe, adding synthetic images has no measurable effect on `asism_tuning_heldout`. ASISM's quantity answer is "none", and SC2 applies to quantity. A vs B is still run and reported. |

**V3 (proposed). Class split of a total count q\*.** Per class c, take round(q\* × n_c / N)
images, with n_c the class's count in the safe pool, then the top images of that class in ASISM's
ranking. Rounding residue goes to the classes with the largest fractional parts, ties broken by
class name. Reason: E4 measured utility at pool proportions (its chains are uniform permutations
of the pool), so this is the composition the count was measured at. Any other split would apply
the count to a composition E4 never measured.

## 4. What is not decided here

- The test-set policy (contract §12; owner: the supervisor).
- The number of Stage 4 seeds (contract §11, from a power calculation).
- The ECE bin count.
- E4b's design. It only exists after a GO.

## 5. Integrity

- No verdict is followed by an extra E4 run. q\* is computed from the 60 frozen runs only.
- If V1–V3 are approved, they are committed before the GPU run starts. The run's start time and
  this document's commit are both recorded, so the order can be checked.
