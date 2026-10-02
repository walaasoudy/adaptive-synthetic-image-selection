"""E4 consequences: the count q* (V1) and its per-class split (V3), from the 60 frozen E4 runs.

Protocol: docs/ham10000_e4_verdict_consequences.md, approved by Walaa on 2026-10-02 before any E4
run. V1 and V3 are transcribed here; V2 (E4b, only after GO) is not designed and is not touched.

  V1  q* = the smallest tested size s that is not shown worse than N, where s is shown worse iff the
      lower end of the two-sided 95% Welch interval on U(N) - U(s) is above 0. No multiplicity
      correction. q* is "not shown worse than N", never "equivalent to N".
  V3  a total q is split across classes in proportion to the safe pool (largest remainder, ties by
      class name), applied when the count is q*: under COARSE. Under NO, q* = 0 and nothing is
      split. Under GO the count is decided by E4b, so no split is made here.

This module counts only. Which images fill each class's count is ASISM v2's within-class ranking,
defined in docs/ham10000_asism_v2_final_protocol.md section 2 and applied by
scripts/followup/ham10000_asism_v2_select.py, which reads this file. The pool's id hash is carried
over from the plan so the selector can check it fills the counts from exactly the pool E4 measured.

Reads e4_plan.json and e4_runs.jsonl, refuses an incomplete grid (the same guard as --phase analyze),
and writes e4_consequences.json next to them. It never trains anything and never reads a split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.followup import ham10000_e4_quantity_curve as e4

PROTOCOL_DOCUMENT = "docs/ham10000_e4_verdict_consequences.md"
OUTPUT_NAME = "e4_consequences.json"


def q_star(values: dict[int, np.ndarray], sizes: tuple[int, ...]) -> dict:
    """V1. One Welch interval per size below N, on U(N) - U(s), and the smallest size whose lower
    end is not above 0. N itself is never shown worse than N, so q* always exists."""
    if tuple(sorted(values)) != tuple(sizes):
        raise e4.E4Error(f"the values cover sizes {sorted(values)}, the design has {list(sizes)}")
    n = sizes[-1]
    tests = {}
    for s in sizes[:-1]:
        entry = e4.welch(values[s], values[n])          # difference = U(N) - U(s)
        entry["lower_95"] = entry["ci95_two_sided"][0]
        entry["shown_worse_than_n"] = bool(entry["lower_95"] > 0)
        tests[str(s)] = entry
    tests[str(n)] = {"shown_worse_than_n": False, "note": "N is never shown worse than N"}
    chosen = next(s for s in sizes if not tests[str(s)]["shown_worse_than_n"])
    return {
        "q_star": int(chosen),
        "tests": tests,
        "statement": f"q* = {chosen}: the smallest tested size not shown worse than N = {n} "
                     "(two-sided 95% Welch, no multiplicity correction). Not an equivalence claim.",
    }


def allocate(total: int, pool_counts: dict[str, int]) -> dict[str, int]:
    """V3. Floor of total * n_c / N per class, then one image at a time to the largest fractional
    parts, ties broken by class name. Exact integer arithmetic, so no float tie is ever invented."""
    pool_n = sum(pool_counts.values())
    if not 0 <= total <= pool_n:
        raise e4.E4Error(f"cannot split {total} images over a pool of {pool_n}")
    base = {c: total * n // pool_n for c, n in pool_counts.items()}
    remainder = {c: total * n % pool_n for c, n in pool_counts.items()}
    left = total - sum(base.values())
    for c in sorted(pool_counts, key=lambda c: (-remainder[c], c))[:left]:
        base[c] += 1
    return dict(sorted(base.items()))


def consequences(plan: dict, rows: list[dict]) -> dict:
    sizes = tuple(plan["sizes"])
    values = e4.values_by_size(rows, plan["cells"], e4.PRIMARY_METRIC)   # refuses partial grids
    verdict = e4.decide(values, sizes)["verdict"]
    v1 = q_star(values, sizes)

    pool_counts = {str(c): int(n) for c, n in plan["safe_pool"]["per_class_counts"].items()}
    if sum(pool_counts.values()) != sizes[-1]:
        raise e4.E4Error(f"the pool's class counts sum to {sum(pool_counts.values())}, N is {sizes[-1]}")
    # V1 agrees with the verdict by construction (Delta_all passes iff L(0) > 0); check it held.
    if (verdict == e4.VERDICT_NO) != (v1["q_star"] == 0):
        raise e4.E4Error(f"verdict {verdict} and q* = {v1['q_star']} disagree; the analysis is inconsistent")

    if verdict == e4.VERDICT_COARSE:
        count = {"condition_c_total": v1["q_star"],
                 "per_class": allocate(v1["q_star"], pool_counts),
                 "condition_d": "random draws within each class at the same per-class counts"}
    elif verdict == e4.VERDICT_NO:
        count = {"condition_c_total": 0, "per_class": {c: 0 for c in sorted(pool_counts)},
                 "condition_d": "not built: there is no count to match (C = A)"}
    else:
        count = {"condition_c_total": None, "per_class": None,
                 "condition_d": "per E4b",
                 "note": "GO: the count is decided by the E4b protocol (V2), which is not designed. "
                         "q* is reported only."}

    return {
        "experiment": e4.EXPERIMENT,
        "protocol_document": PROTOCOL_DOCUMENT,
        "primary_metric": e4.PRIMARY_METRIC,
        "verdict": verdict,
        "v1": v1,
        "v3_pool_counts": dict(sorted(pool_counts.items())),
        "v3_pool_ids_sha256": plan["safe_pool"].get("ids_sha256"),
        "count": count,
        "within_class_selection": "not chosen here: the ranking of docs/ham10000_asism_v2_final_protocol.md "
                                  "section 2, applied by scripts.followup.ham10000_asism_v2_select.",
    }


def run(out_dir: Path) -> dict:
    plan = e4._load_plan(out_dir)
    rows = e4._read_jsonl(out_dir / "e4_runs.jsonl")
    frozen = {c["cell_id"]: c["subset_sha256"] for c in plan["cells"]}
    for row in rows:                                    # the same guard as --phase analyze
        if row["cell_id"] in frozen and row.get("subset_sha256") != frozen[row["cell_id"]]:
            raise e4.E4Error(f"run {row['cell_id']} trained on a different subset than the plan froze")
    result = consequences(plan, rows)
    path = out_dir / OUTPUT_NAME
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"verdict {result['verdict']}; {result['v1']['statement']}")
    print(f"condition C count: {result['count']['condition_c_total']} {result['count']['per_class']}")
    return {"path": str(path), **result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True, help="the E4 directory holding e4_plan.json and e4_runs.jsonl")
    args = parser.parse_args()
    run(Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
