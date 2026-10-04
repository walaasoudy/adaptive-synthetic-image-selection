#!/usr/bin/env python3
"""The condition sets and comparison families, named and fixed before any result is read.

Stage 4 trains one classifier per (condition, seed); Stage 5 compares them. WHICH conditions exist
and WHICH pair carries the confirmatory claim are pre-registration decisions, not implementation
details, so they live here as named protocols rather than as literals spread across three scripts.

    v1  A / B / C          the thesis's primary result. Confirmatory: C vs B.
    v2  A / B / C2 / D2    the post-hoc follow-up. Confirmatory: C2 vs D2.

WHY v2's CONFIRMATORY PAIR IS C2 vs D2. v1 had no random control, which the limitations record
names as its largest design gap: C beating B shows that selecting beats keeping everything, but not
that THIS selector beats drawing the same number of images at random from the same safe pool. D2 is
that draw, at C2's exact per-class counts. C2 vs D2 is therefore the only comparison that isolates
the selection rule itself, and it is the only one in v2's confirmatory family.

v1 IS FROZEN. Its protocol is reproduced here exactly as the three scripts hardcoded it before this
module existed, and a test asserts that. Adding v2 must not move a single v1 number.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionManifest:
    """Where a protocol records the selection that produced its selected condition(s).

    v1 has one selected condition (C) and one manifest. v2 has two (C2 and its matched random
    control D2) and ALSO one manifest, because a single selection run produces both draws and
    documents them together; `covers` is what ties the file to the conditions it explains, which is
    why this is a descriptor rather than a per-condition path.
    """

    config: str
    """The config file whose paths.outputs_dir the manifest lives under."""
    section: str
    filename: str
    """The file inside <outputs_dir>/<namespace>/."""
    covers: tuple[str, ...]
    """The conditions whose synthetic images this manifest accounts for."""
    rebuild_command: str
    """Printed verbatim when the manifest is missing; {namespace} is substituted."""
    evidence_keys: tuple[str, ...]
    """Manifest keys copied into the Stage 5 evidence as selection_<key>."""


@dataclass(frozen=True)
class ConditionProtocol:
    """One experiment's conditions and its pre-registered comparison families."""

    name: str
    stage4_config: str
    conditions: tuple[str, ...]
    baseline: str
    """The real-only condition every per-class delta is measured against."""
    synthetic_conditions: tuple[str, ...]
    """Conditions that must have trained on synthetic images; one with none is a broken run."""
    confirmatory: tuple[tuple[str, str], ...]
    """(left, right) pairs tested on the primary metric under family-wise error control."""
    exploratory: tuple[tuple[str, str], ...]
    """Everything else: reported with FDR control and labelled exploratory wherever it appears."""
    equal_counts_suspect: tuple[tuple[str, str], ...]
    """Pairs whose synthetic image counts matching means something went wrong (e.g. C read B's
    manifest, or the selector kept every candidate)."""
    equal_counts_required: tuple[tuple[str, str], ...]
    """Pairs whose counts MUST match, from different manifests: a matched control and the arm it
    controls for. Unequal counts there would make the comparison a size comparison."""
    interpretation: str
    """Copied verbatim into the Stage 5 comparison report. v1's is reproduced exactly as that
    script wrote it, so a v1 report re-generated after this change is unchanged to the byte."""
    selection_manifest: SelectionManifest
    stage5_dirname: str
    """The directory Stage 5 writes predictions under, beside stage4's results_dir. Separate per
    protocol: a v2 evaluation sharing v1's tree could overwrite the protected result under a
    colliding run id."""
    selection_outcomes: tuple[str, ...] = ()
    """For a selector whose COUNT is its own output: the `selection_outcome` values of the selection
    manifest under which this protocol is the one to train. Empty: the manifest is not consulted
    (every protocol before asism_v2_learned)."""
    added_metrics: tuple[str, ...] = ()
    """Metrics reported at Stage 5 on top of the four v1 reports. Empty for v1, whose recorded
    comparison must re-generate unchanged: every added metric enlarges the exploratory family that
    Benjamini-Hochberg corrects over."""


# Contract §12 (decided 2026-10-04, before any v2 Stage 5 result): average precision, Brier score
# and top-label ECE with 15 equal-width bins, for every protocol after v1.
V2_ADDED_METRICS = ("macro_average_precision", "brier_score", "ece_top_label")


V1 = ConditionProtocol(
    name="v1",
    stage4_config="ham10000_stage4.yaml",
    conditions=("A", "B", "C"),
    baseline="A",
    synthetic_conditions=("B", "C"),
    confirmatory=(("C", "B"),),
    exploratory=(("C", "A"), ("B", "A")),
    equal_counts_suspect=(("B", "C"),),
    equal_counts_required=(),
    interpretation=(
        "C vs B is the confirmatory test of the contribution: same generator, same "
        "candidates, selection the only difference. C vs A shows only that synthetic data "
        "helps. Every p-value is reported with its effect size and interval; significance "
        "alone is not a result."
    ),
    selection_manifest=SelectionManifest(
        config="ham10000_stage3.yaml",
        section="ham_stage3",
        filename="asism_selection_manifest.json",
        covers=("C",),
        rebuild_command="python scripts/asism/ham10000_05_adaptive_thresholds.py --namespace {namespace}",
        evidence_keys=("threshold_policy", "n_selected"),
    ),
    stage5_dirname="stage5",
)

V2 = ConditionProtocol(
    name="v2",
    stage4_config="ham10000_v2_stage4.yaml",
    conditions=("A", "B", "C2", "D2"),
    baseline="A",
    synthetic_conditions=("B", "C2", "D2"),
    confirmatory=(("C2", "D2"),),
    exploratory=(("C2", "B"), ("D2", "B"), ("C2", "A"), ("D2", "A"), ("B", "A")),
    # C2 vs D2 is deliberately absent: they are built to the same per-class counts, so matching
    # counts there is the design holding, not a fault.
    equal_counts_suspect=(("B", "C2"), ("B", "D2")),
    equal_counts_required=(("C2", "D2"),),
    interpretation=(
        "C2 vs D2 is the confirmatory test of the selection rule: the same safe pool, the same "
        "number of images per class, the rule the only difference. C2 vs B and the comparisons "
        "against A are exploratory and say nothing about the rule on their own. Every p-value is "
        "reported with its effect size and interval; significance alone is not a result."
    ),
    # One file documents both draws: C2's selection and the D2 control built to its per-class
    # counts. Splitting it would let the two fall out of step.
    selection_manifest=SelectionManifest(
        config="ham10000_v2_selection.yaml",
        section="ham_v2_selection",
        filename="v2_selection_manifest.json",
        covers=("C2", "D2"),
        rebuild_command=(
            "python scripts/followup/ham10000_v2_select.py --namespace {namespace} "
            "--scores <stage3 scores csv>"
        ),
        evidence_keys=("n_selected_c2", "n_selected_d2"),
    ),
    stage5_dirname="stage5_v2",
    added_metrics=V2_ADDED_METRICS,
)

# ASISM v2 final (docs/ham10000_asism_v2_final_protocol.md), frozen before any E4 result. Which of
# the two applies is decided by the E4 verdict alone: COARSE -> asism_v2, NO -> asism_v2_none.
_ASISM_V2_SELECTION = SelectionManifest(
    config="ham10000_asism_v2_selection.yaml",
    section="ham_asism_v2_selection",
    filename="asism_v2_selection_manifest.json",
    covers=("C", "D"),
    rebuild_command=(
        "python -m scripts.followup.ham10000_asism_v2_select --namespace {namespace} --candidates <csv> "
        "--scores-dir <four_signal set> --gonogo-report <report> --e4-consequences <e4_consequences.json>"
    ),
    evidence_keys=("e4_verdict", "q_star", "per_class", "n_selected_c", "n_selected_d", "c_ids_sha256"),
)

ASISM_V2 = ConditionProtocol(
    name="asism_v2",
    stage4_config="ham10000_asism_v2_stage4.yaml",
    conditions=("A", "B", "C", "D"),
    baseline="A",
    synthetic_conditions=("B", "C", "D"),
    confirmatory=(("C", "D"),),
    exploratory=(("C", "B"), ("D", "B"), ("C", "A"), ("D", "A"), ("B", "A")),
    equal_counts_suspect=(("B", "C"), ("B", "D")),
    equal_counts_required=(("C", "D"),),
    interpretation=(
        "C vs D is the confirmatory test of ASISM v2's within-class ranking: the same safe pool, "
        "the same E4-derived number of images per class, the ranking the only difference. The "
        "count itself is E4's result, not tested here. Comparisons with A and B are exploratory. "
        "Every p-value is reported with its effect size and interval; significance alone is not a "
        "result."
    ),
    selection_manifest=_ASISM_V2_SELECTION,
    stage5_dirname="stage5_asism_v2",
    added_metrics=V2_ADDED_METRICS,
)

ASISM_V2_NONE = ConditionProtocol(
    name="asism_v2_none",
    stage4_config="ham10000_asism_v2_stage4.yaml",
    conditions=("A", "B"),
    baseline="A",
    synthetic_conditions=("B",),
    confirmatory=(),
    exploratory=(("B", "A"),),
    equal_counts_suspect=(),
    equal_counts_required=(),
    interpretation=(
        "E4 = NO: ASISM v2 adds no synthetic image (q* = 0), so C = A and D does not exist. There is "
        "no selection comparison to make. B vs A is reported, exploratory."
    ),
    selection_manifest=SelectionManifest(
        config="ham10000_asism_v2_selection.yaml",
        section="ham_asism_v2_selection",
        filename="asism_v2_selection_manifest.json",
        covers=(),
        rebuild_command=_ASISM_V2_SELECTION.rebuild_command,
        evidence_keys=("e4_verdict", "q_star"),
    ),
    stage5_dirname="stage5_asism_v2",
    added_metrics=V2_ADDED_METRICS,
)

# ASISM v2 with the learned ranker and the stopping rule (contract §9 and §10, approved 2026-10-03).
# The selector decides which images AND how many, so three outcomes are valid results: a proper
# subset, every candidate, or none. The first trains A/B/C/D; in the other two C is B's (or A's)
# training set by definition, so only A and B are trained and the outcome itself is reported.
_ASISM_V2_LEARNED_REBUILD = (
    "python -m scripts.followup.ham10000_asism_v2_learned_select --namespace {namespace} --phase select "
    "--candidates <csv> --scores-dir <four_signal set>"
)

ASISM_V2_LEARNED = ConditionProtocol(
    name="asism_v2_learned",
    stage4_config="ham10000_asism_v2_learned_stage4.yaml",
    conditions=("A", "B", "C", "D"),
    baseline="A",
    synthetic_conditions=("B", "C", "D"),
    confirmatory=(("C", "D"),),
    exploratory=(("C", "B"), ("D", "B"), ("C", "A"), ("D", "A"), ("B", "A")),
    equal_counts_suspect=(("B", "C"), ("B", "D")),
    equal_counts_required=(("C", "D"),),
    interpretation=(
        "C vs D is the confirmatory test of WHICH images ASISM v2 selects: the same safe pool, the "
        "same number of images per class, the learned ranking the only difference. HOW MANY is the "
        "output of ASISM's stopping rule; C vs D holds it fixed and does not test it. C vs B and the "
        "comparisons with A are exploratory. Every p-value is reported with its effect size and "
        "interval; significance alone is not a result."
    ),
    selection_manifest=SelectionManifest(
        config="ham10000_asism_v2_ranker.yaml",
        section="ham_asism_v2_ranker",
        filename="asism_v2_learned_selection_manifest.json",
        covers=("C", "D"),
        rebuild_command=_ASISM_V2_LEARNED_REBUILD,
        evidence_keys=("selection_outcome", "per_class", "n_selected_c", "n_selected_d", "c_ids_sha256"),
    ),
    stage5_dirname="stage5_asism_v2_learned",
    selection_outcomes=("subset",),
    added_metrics=V2_ADDED_METRICS,
)

ASISM_V2_LEARNED_ALL_OR_NONE = ConditionProtocol(
    name="asism_v2_learned_all_or_none",
    stage4_config="ham10000_asism_v2_learned_stage4.yaml",
    conditions=("A", "B"),
    baseline="A",
    synthetic_conditions=("B",),
    confirmatory=(),
    exploratory=(("B", "A"),),
    equal_counts_suspect=(),
    equal_counts_required=(),
    interpretation=(
        "ASISM v2's stopping rule kept every candidate (C = B) or none (C = A); the selection "
        "manifest says which. That outcome is the result of the selection. There is no separate "
        "selected set to compare, so no selection comparison is made. B vs A is reported, exploratory."
    ),
    selection_manifest=SelectionManifest(
        config="ham10000_asism_v2_ranker.yaml",
        section="ham_asism_v2_ranker",
        filename="asism_v2_learned_selection_manifest.json",
        covers=(),
        rebuild_command=_ASISM_V2_LEARNED_REBUILD,
        evidence_keys=("selection_outcome", "per_class", "n_selected_c"),
    ),
    stage5_dirname="stage5_asism_v2_learned",
    selection_outcomes=("all", "none"),
    added_metrics=V2_ADDED_METRICS,
)

PROTOCOLS = {protocol.name: protocol for protocol in (
    V1, V2, ASISM_V2, ASISM_V2_NONE, ASISM_V2_LEARNED, ASISM_V2_LEARNED_ALL_OR_NONE)}
DEFAULT_PROTOCOL = "v1"


def get_protocol(name: str | None) -> ConditionProtocol:
    key = str(name or DEFAULT_PROTOCOL)
    if key not in PROTOCOLS:
        raise KeyError(f"unknown protocol {key!r}; known: {sorted(PROTOCOLS)}")
    return PROTOCOLS[key]
