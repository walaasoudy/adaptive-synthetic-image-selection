"""E4 consequences: V1 (q*) and V3 (per-class split) as approved, on synthetic grids (no GPU)."""

import json

import numpy as np
import pytest

from scripts.followup import ham10000_e4_consequences as cons
from scripts.followup import ham10000_e4_quantity_curve as e4

POOL_COUNTS = {"df": 885, "vasc": 736, "akiec": 486, "bcc": 400, "mel": 276, "bkl": 273, "nv": 112}
SIZES = (0, 250, 500, 1000, 2000, 3168)


def _values(means, sd=0.004, seed=0):
    rng = np.random.default_rng(seed)
    return {size: mean + rng.normal(0, sd, e4.RUNS_PER_SIZE) for size, mean in zip(SIZES, means)}


def _plan_and_rows(means, sd=0.004, seed=0):
    cells = e4.plan_cells(SIZES)
    values = _values(means, sd, seed)
    position = {size: 0 for size in SIZES}
    rows = []
    for c in cells:
        v = float(values[c["size"]][position[c["size"]]])
        position[c["size"]] += 1
        rows.append({"cell_id": c["cell_id"], "metrics": {m: v for m in e4.METRICS}})
    plan = {"sizes": list(SIZES), "cells": cells, "safe_pool": {"per_class_counts": POOL_COUNTS}}
    return plan, rows


# ---- V1 --------------------------------------------------------------------------------------------


def test_q_star_is_the_smallest_size_whose_welch_lower_bound_is_not_above_zero():
    values = _values([0.90, 0.92, 0.93, 0.94, 0.94, 0.94], sd=0.002)
    result = cons.q_star(values, SIZES)
    assert result["q_star"] == 1000
    for s in (0, 250, 500):
        assert result["tests"][str(s)]["shown_worse_than_n"]
        assert result["tests"][str(s)]["lower_95"] > 0
    assert not result["tests"]["1000"]["shown_worse_than_n"]
    assert "Not an equivalence claim" in result["statement"]


def test_q_star_is_n_when_every_smaller_size_is_shown_worse():
    values = _values([0.80, 0.82, 0.84, 0.86, 0.88, 0.95], sd=0.001)
    assert cons.q_star(values, SIZES)["q_star"] == 3168


def test_q_star_is_zero_when_n_is_not_better_than_real_only():
    values = _values([0.93] * 6, sd=0.01)
    assert cons.q_star(values, SIZES)["q_star"] == 0


def test_q_star_takes_the_smallest_even_on_a_non_monotone_curve():
    # 250 is not shown worse, 500 is: the rule is taken as written, so q* = 250.
    values = _values([0.85, 0.95, 0.85, 0.95, 0.95, 0.95], sd=0.002)
    assert cons.q_star(values, SIZES)["q_star"] == 250


def test_q_star_uses_the_same_welch_as_the_verdict_and_no_correction():
    values = _values([0.90, 0.91, 0.92, 0.93, 0.94, 0.95], sd=0.006, seed=3)
    result = cons.q_star(values, SIZES)
    for s in SIZES[:-1]:
        expected = e4.welch(values[s], values[3168])["ci95_two_sided"][0]
        assert result["tests"][str(s)]["lower_95"] == pytest.approx(expected)


# ---- V3 --------------------------------------------------------------------------------------------


def test_allocation_sums_to_the_total_and_follows_pool_proportions():
    for total in (0, 250, 500, 1000, 2000, 3168):
        split = cons.allocate(total, POOL_COUNTS)
        assert sum(split.values()) == total
        for c, n in POOL_COUNTS.items():
            assert abs(split[c] - total * n / 3168) < 1
    assert cons.allocate(3168, POOL_COUNTS) == dict(sorted(POOL_COUNTS.items()))


def test_allocation_largest_remainder_with_alphabetical_ties():
    # 250 * n / 3168: df 69.84, vasc 58.08, akiec 38.35, bcc 31.57, mel 21.78, bkl 21.54, nv 8.84.
    # Floors sum to 246; the 4 largest fractions are df .84, nv .84, mel .78, bcc .57.
    assert cons.allocate(250, POOL_COUNTS) == {
        "akiec": 38, "bcc": 32, "bkl": 21, "df": 70, "mel": 22, "nv": 9, "vasc": 58}
    # Exact tie: two equal classes, one image left -> alphabetical first wins.
    assert cons.allocate(1, {"b": 5, "a": 5}) == {"a": 1, "b": 0}


def test_allocation_refuses_impossible_totals():
    with pytest.raises(e4.E4Error):
        cons.allocate(3169, POOL_COUNTS)
    with pytest.raises(e4.E4Error):
        cons.allocate(-1, POOL_COUNTS)


# ---- per verdict -----------------------------------------------------------------------------------


def test_coarse_splits_q_star_and_matches_d_counts():
    # Overall gain, but steps 2..5 too small to resolve after Holm -> COARSE.
    plan, rows = _plan_and_rows([0.900, 0.925, 0.927, 0.929, 0.931, 0.933], sd=0.006, seed=1)
    result = cons.consequences(plan, rows)
    assert result["verdict"] == e4.VERDICT_COARSE
    q = result["v1"]["q_star"]
    assert q >= 250
    assert result["count"]["condition_c_total"] == q
    assert result["count"]["per_class"] == cons.allocate(q, POOL_COUNTS)
    assert "pending" in result["within_class_selection"]


def test_no_gives_zero_and_no_condition_d():
    plan, rows = _plan_and_rows([0.93] * 6, sd=0.01, seed=2)
    result = cons.consequences(plan, rows)
    assert result["verdict"] == e4.VERDICT_NO
    assert result["v1"]["q_star"] == 0
    assert result["count"]["condition_c_total"] == 0
    assert "not built" in result["count"]["condition_d"]


def test_go_reports_q_star_but_makes_no_split():
    plan, rows = _plan_and_rows([0.80, 0.82, 0.85, 0.88, 0.91, 0.94], sd=0.002)
    result = cons.consequences(plan, rows)
    assert result["verdict"] == e4.VERDICT_GO
    assert result["count"]["per_class"] is None and result["count"]["condition_c_total"] is None
    assert result["v1"]["q_star"] == 3168


def test_partial_grid_and_bad_pool_counts_are_refused():
    plan, rows = _plan_and_rows([0.93] * 6)
    with pytest.raises(e4.E4Error, match="incomplete"):
        cons.consequences(plan, rows[:-1])
    plan["safe_pool"]["per_class_counts"] = {**POOL_COUNTS, "nv": 111}
    with pytest.raises(e4.E4Error, match="sum to"):
        cons.consequences(plan, rows)


def test_run_writes_the_report_from_a_real_plan_and_runs_file(tmp_path):
    pool = [f"syn_{i:05d}" for i in range(3168)]
    report = {"safety_checks_applied": {"reject_invalid_iqa": True, "reject_near_duplicates": True},
              "safety_gate_removals": {}, "candidates_before_safety_gate": 3168,
              "per_class_counts": POOL_COUNTS}
    out = tmp_path / "e4"
    e4.run_plan("ns", out, pool_loader=lambda ns: (pool, report))
    plan = json.loads((out / "e4_plan.json").read_text())
    _, rows = _plan_and_rows([0.93] * 6, sd=0.01, seed=2)
    assert [c["cell_id"] for c in plan["cells"]] == [r["cell_id"] for r in rows]
    for row, cell in zip(rows, plan["cells"]):
        row["subset_sha256"] = cell["subset_sha256"]
    path = out / "e4_runs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    result = cons.run(out)
    written = json.loads((out / cons.OUTPUT_NAME).read_text())
    assert written["verdict"] == result["verdict"] == e4.VERDICT_NO

    rows[-1]["subset_sha256"] = "0" * 64
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(e4.E4Error, match="different subset"):
        cons.run(out)
