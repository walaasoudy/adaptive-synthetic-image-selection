# HAM10000 ASISM v2: Go/No-Go on the four-signal set

Date 2026-10-02. CPU only. This follows Amendment 7 in `docs/ham10000_v2_signal_criteria.md`: J1
failed V4, so agreement is dropped and ASISM v2 runs on four signals.

## Run

- Gate: `scripts/asism/ham10000_02_gonogo.py` with the |ρ| redundancy fix (`89de88b`). The code
  and thresholds are unchanged, with redundancy at weighted within-class |ρ| < 0.90.
- Signal set: the same artifacts as the five-signal J1 run (`v3_signals/judge_j1/gonogo_signal_set`),
  with `agreement_scores.*` left out. The new set is in `ham10000_work/v3_signals/four_signal/gonogo_signal_set/`.
- Candidate manifest `all_candidates.csv`, sha256 `bf8047b5…`, the same file for every signal.
- Report: `ham10000_work/v3_signals/four_signal/gonogo_report.json`, sha256 `9ea815ea99c51376…`.
- Run at repo commit `ef6413a`.

## Result

| Signal | Source | Outcome | Max weighted \|ρ\| (with) |
|---|---|---|---|
| similarity | DINOv2 ViT-S/14 | include | 0.472 (iqa) |
| iqa | image-quality composite | include | 0.472 (similarity) |
| uncertainty | V3a aux classifier (MI) | include | 0.242 (explainability) |
| explainability | V3a aux classifier (Grad-CAM typicality) | include | 0.242 (uncertainty) |
| agreement | — | exclude, artifact not produced | — |

All four signals pass all eight checks. They are:

- technical validity
- selection eligibility
- numerical stability
- missing rate
- provenance
- directionality
- redundancy
- usefulness

The variant is **primary**, because at least three signals were included.

## What this does and does not decide

- The four signals are fixed as the ASISM v2 signal set.
- V3a is used only to produce uncertainty and explainability. It is never used as a class judge.
  Its judge failure was a different function: 84.8% of synthetic mel was read as nv.
- Limitation, to be reported: a judge that confidently mislabels synthetic mel could give those
  images a low uncertainty. Q3 passed, so the signal stays, but the risk is stated.
- This gate does **not** start utility or ranking. Whether a learned selector is trainable is the
  E4 question. The CPU audit (`docs/ham10000_e4_cpu_audit.md`) found that same-size selection
  differences are below the training noise in every existing run. The choice between SC2, E4 as
  written and a quantity-only E4 is pending with Walaa and the supervisor.
- The Stage 4 evaluation classifier stays DenseNet-121, at 512 px and 3,000 steps. It is separate
  from V3a.
