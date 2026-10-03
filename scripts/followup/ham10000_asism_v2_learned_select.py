#!/usr/bin/env python3
"""ASISM v2 learned selection: acceptance, fit, selection (C and D) and seed stability. CPU only.

Runs after scripts/followup/ham10000_asism_v2_utility.py has measured the labels and G1 has passed.
Every value comes from scripts/asism_v2/prereg.py.

  --phase accept     minutes. Fits without bootstrap, reads the 40 test subsets ONCE and writes the
                     verdict. If the ranker is not accepted, nothing after this phase runs: the
                     failure is the reported result.
  --phase fit        about 1.5 CPU-hours. The ranker and its 200 bootstrap models at fit seed 42,
                     saved once.
  --phase select     seconds. The stopping rule on the whole safe pool decides which images and how
                     many; writes C, the matched random D per Stage 4 seed, and the manifest.
  --phase stability  about 8 CPU-hours. The fit and the selection repeated at fit seeds 42 to 46;
                     reported, never chosen from.

Reads the candidate manifest, the four-signal artifacts and the utility measurements. Never reads
classifier_val or final_eval_heldout, and trains no classifier.

Usage:
    python -m scripts.followup.ham10000_asism_v2_learned_select --phase accept \
        --candidates <stage2/all_candidates.csv> --scores-dir <four_signal/gonogo_signal_set>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.asism_v2.contracts import fingerprint, write_new_json
from scripts.asism_v2.gates import ACCEPTANCE_NAME, acceptance, selection_stability
from scripts.asism_v2.persist import load_fitted, save_fitted
from scripts.asism_v2.pipeline import fit_ranker
from scripts.asism_v2.prereg import fit_arguments, load_prereg
from scripts.asism_v2.preserve import sha256
from scripts.asism_v2.selection_files import MANIFEST_NAME, build_selection, write_selection
from scripts.asism_v2.supervision import fit_inputs, load_signal_table
from scripts.followup import ham10000_asism_v2_utility as utility

STAGE4_CONFIG = "ham10000_asism_v2_learned_stage4.yaml"
RANKER_NAME = "ranker_fit_seed{seed}.json"
STABILITY_NAME = "selection_stability.json"


class LearnedSelectionError(RuntimeError):
    pass


def _context(namespace: str, candidates: Path, scores_dir: Path, out_dir: Path | None) -> dict:
    prereg = load_prereg()
    out_dir = Path(out_dir or utility.default_out_dir(prereg, namespace))
    plan = utility.load_plan(out_dir, prereg)
    frame, report = load_signal_table(candidates, scores_dir, prereg["safety"])
    if report["safe_pool_ids_sha256"] != plan["pool"]["safe_pool_ids_sha256"]:
        raise LearnedSelectionError("the safe pool is not the one the utility plan was drawn from")
    rows, protocol = utility.fit_measurements(out_dir)                  # refuses unless G1 passed
    return {"prereg": prereg, "out_dir": out_dir, "plan": plan, "frame": frame, "rows": rows,
            "protocol": protocol, "subsets": fit_inputs(plan), "seeds": plan["training_seeds"],
            "n_candidates": len(pd.read_csv(candidates))}


def _accepted(out_dir: Path) -> str:
    path = out_dir / ACCEPTANCE_NAME
    if not path.is_file():
        raise LearnedSelectionError("the ranker has not been through acceptance; run --phase accept")
    if not json.loads(path.read_text(encoding="utf-8"))["accepted"]:
        raise LearnedSelectionError("the ranker was NOT accepted on the held-out subsets: the learned "
                                    "ranking is not used and nothing is selected. That is the result.")
    return sha256(path)


def _provenance(ctx: dict, acceptance_sha256: str) -> dict:
    return {"prereg_sha256": ctx["prereg"]["prereg_sha256"], "protocol_sha256": fingerprint(ctx["protocol"]),
            "plan_sha256": sha256(ctx["out_dir"] / utility.PLAN_NAME), "acceptance_sha256": acceptance_sha256}


def run_accept(namespace, candidates, scores_dir, out_dir=None) -> dict:
    ctx = _context(namespace, candidates, scores_dir, out_dir)
    if (ctx["out_dir"] / ACCEPTANCE_NAME).exists():                  # before the test file is opened
        raise LearnedSelectionError("the test subsets were already read once; acceptance is not repeated")
    return acceptance(ctx["frame"], ctx["subsets"], ctx["rows"], utility.heldout_measurements(ctx["out_dir"]),
                      ctx["seeds"], ctx["protocol"], ctx["prereg"], fit_arguments(ctx["prereg"]),
                      ctx["out_dir"] / ACCEPTANCE_NAME)


def run_fit(namespace, candidates, scores_dir, out_dir=None) -> dict:
    ctx = _context(namespace, candidates, scores_dir, out_dir)
    provenance = _provenance(ctx, _accepted(ctx["out_dir"]))
    arguments = fit_arguments(ctx["prereg"])
    path = ctx["out_dir"] / RANKER_NAME.format(seed=arguments["seed"])
    if path.exists():
        raise LearnedSelectionError(f"{path} exists: the ranker is fitted once")
    fitted = fit_ranker(ctx["frame"], ctx["subsets"], ctx["rows"], ctx["seeds"], ctx["protocol"], **arguments)
    return {"ranker": str(path), "content_sha256": save_fitted(path, fitted, provenance), **fitted.history}


def run_select(namespace, candidates, scores_dir, out_dir=None) -> dict:
    from scripts.utils.config import load_named_config

    ctx = _context(namespace, candidates, scores_dir, out_dir)
    provenance = _provenance(ctx, _accepted(ctx["out_dir"]))
    path = ctx["out_dir"] / RANKER_NAME.format(seed=fit_arguments(ctx["prereg"])["seed"])
    if not path.is_file():
        raise LearnedSelectionError(f"no fitted ranker at {path}; run --phase fit")
    fitted, saved = load_fitted(path, provenance)
    seeds = [int(s) for s in load_named_config(STAGE4_CONFIG, "ham_stage4").seeds]
    selection = build_selection(fitted, ctx["frame"], seeds, ctx["n_candidates"])
    return write_selection(ctx["out_dir"], selection, ctx["frame"], namespace, seeds,
                           {**provenance, "ranker_sha256": saved["content_sha256"],
                            "candidates_csv_sha256": ctx["plan"]["pool"]["candidates_csv_sha256"],
                            "safe_pool_ids_sha256": ctx["plan"]["pool"]["safe_pool_ids_sha256"]})


def run_stability(namespace, candidates, scores_dir, out_dir=None) -> dict:
    ctx = _context(namespace, candidates, scores_dir, out_dir)
    provenance = _provenance(ctx, _accepted(ctx["out_dir"]))
    if (ctx["out_dir"] / STABILITY_NAME).exists():
        raise LearnedSelectionError("the stability report already exists")
    report = selection_stability(ctx["frame"], ctx["subsets"], ctx["rows"], ctx["seeds"], ctx["protocol"],
                                 ctx["frame"], ctx["prereg"], fit_arguments(ctx["prereg"]))
    report = json.loads(json.dumps({**report, **provenance}))        # integer seeds become JSON keys
    write_new_json(ctx["out_dir"] / STABILITY_NAME, report)
    return report


PHASES = {"accept": run_accept, "fit": run_fit, "select": run_select, "stability": run_stability}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", default="ham-stratified-v1")
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--scores-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    result = PHASES[args.phase](args.namespace, args.candidates, args.scores_dir, args.out_dir)
    result = {k: v for k, v in result.items() if k not in ("stops", "models")}
    print(json.dumps(result, indent=2, default=str))
    print(f"(manifest: {MANIFEST_NAME})" if args.phase == "select" else "", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
