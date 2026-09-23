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
)

PROTOCOLS = {protocol.name: protocol for protocol in (V1, V2)}
DEFAULT_PROTOCOL = "v1"


def get_protocol(name: str | None) -> ConditionProtocol:
    key = str(name or DEFAULT_PROTOCOL)
    if key not in PROTOCOLS:
        raise KeyError(f"unknown protocol {key!r}; known: {sorted(PROTOCOLS)}")
    return PROTOCOLS[key]
