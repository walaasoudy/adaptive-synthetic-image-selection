#!/usr/bin/env python3
"""J1, the agreement judge of Amendment 6 (docs/ham10000_v2_signal_criteria.md).

The P1/P1b probe, unchanged, trained on gen_train together with classifier_train (5,227 real images),
validated on classifier_val, and read on the 3,168 synthetic candidates. It is accepted only if all
six criteria hold:

    V1  classifier_val balanced accuracy > 0.478 and no class with zero recall (criteria A, B)
    V2  Q1 sensible, normalised by J1's classifier_val recall
    V3  Q2 not influential
    V4  no single class receives >= 50% of all synthetic mismatches
    V5  agreement is not a within-class restatement of similarity, uncertainty or explainability
        (|rho| >= 0.7 in >= 4 classes makes a pair redundant)
    V6  agreement is "include" under the corrected Go/No-Go, with the five-signal set

If it is accepted, its agreement artifact replaces V3a's in ASISM v2. If any criterion fails,
agreement is left out of ASISM v2 and no other judge is tried on this pool. CPU only; no earlier
artifact is written.

Usage (approved 2026-10-02, CPU):
    python scripts/asism/ham10000_agreement_judge.py \
        --embeddings-dir <separability dir> --real-embeddings-dir <embed-real dir> \
        --metadata <HAM10000_metadata.csv> --candidates <all_candidates.csv> \
        --similarity-dir <dir with similarity + iqa scores> --v3-signals-dir <V3a signals dir> \
        --output-dir <new dir>
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.asism import ham10000_02_gonogo as gonogo  # noqa: E402
from scripts.asism import ham10000_independent_probe as probe  # noqa: E402
from scripts.asism import ham10000_v2_signal_diagnostics as diag  # noqa: E402
from scripts.asism.ham10000_signals import compute_agreement_scores  # noqa: E402
from scripts.asism.ham10000_synthetic_separability import l2_normalise, load_embeddings  # noqa: E402
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS, normalize_diagnosis  # noqa: E402
from scripts.utils.manifest import get_git_commit_hash, read_json, sha256_file, write_json  # noqa: E402

CLASSES: list[str] = list(CLASSIFIER_TARGET_LABELS)
TRAIN_SPLITS = ("gen_train", "classifier_train")
VALIDATION_SPLIT = "classifier_val"
SINK_MAX_SHARE = 0.50                      # V4: Q2's 50%, for every class
V5_MIN_ABS_RHO = diag.CRITERIA["q5_min_abs_spearman"]           # 0.7
V5_MIN_CLASSES = diag.CRITERIA["q5_min_redundant_classes"]      # 4
V5_PARTNERS = {
    "similarity": "similarity_knn_mean",
    "uncertainty": "uncertainty_mutual_information",
    "explainability": "explainability_calibrated_typicality",
}


# ==============================================================================================
# Inputs
# ==============================================================================================


def training_data(separability_dir: Path, real_dir: Path, metadata: pd.DataFrame):
    """(x_train, y_train, x_val, y_val, x_syn, syn_ids, syn_dx, info), lesion-disjointness checked."""
    x, y, source = load_embeddings(separability_dir)
    ids = np.load(separability_dir / "embeddings.npz", allow_pickle=False)["image_id"].astype(str)
    x_real, real_ids, real_dx, split = probe.load_real_embeddings(real_dir, separability_dir)
    gen = source == "real"
    syn = source == "synthetic"
    cls_train, val = split == "classifier_train", split == VALIDATION_SPLIT

    lesion = dict(zip(metadata["image_id"].astype(str), metadata["lesion_id"].astype(str)))
    groups = {"gen_train": set(ids[gen]), "classifier_train": set(real_ids[cls_train]),
              VALIDATION_SPLIT: set(real_ids[val])}
    missing = sorted(i for g in groups.values() for i in g if i not in lesion)
    if missing:
        raise SystemExit(f"{len(missing)} real image(s) are not in the metadata (e.g. {missing[:3]}).")
    names = list(groups)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            shared = {lesion[i] for i in groups[names[a]]} & {lesion[i] for i in groups[names[b]]}
            if shared:
                raise SystemExit(f"{names[a]} and {names[b]} share {len(shared)} lesion(s); refusing.")

    x_train = l2_normalise(np.concatenate([x[gen], x_real[cls_train]]))
    y_train = np.array([CLASSES.index(n) for n in [*y[gen], *real_dx[cls_train]]])
    info = {"n_gen_train": int(gen.sum()), "n_classifier_train": int(cls_train.sum()),
            "n_train": int(len(y_train)), "n_validation": int(val.sum()), "n_synthetic": int(syn.sum())}
    return (x_train, y_train, l2_normalise(x_real[val]), np.array([CLASSES.index(n) for n in real_dx[val]]),
            l2_normalise(x[syn]), ids[syn], y[syn], info)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix.tolist()


# ==============================================================================================
# Criteria V4, V5
# ==============================================================================================


def v4_sink(frame: pd.DataFrame) -> dict:
    mismatched = frame[~frame["agreement_is_argmax_match"].astype(bool)]
    counts = mismatched["agreement_predicted_diagnosis"].value_counts().reindex(CLASSES, fill_value=0)
    total = int(counts.sum())
    shares = {k: (int(v) / total if total else 0.0) for k, v in counts.items()}
    worst = max(shares, key=shares.get)
    return {"n_mismatches": total, "shares": shares, "largest": worst, "largest_share": shares[worst],
            "max_share_exclusive": SINK_MAX_SHARE, "passed": bool(shares[worst] < SINK_MAX_SHARE)}


def v5_not_a_restatement(agreement: pd.DataFrame, partners: dict[str, pd.DataFrame]) -> dict:
    pairs = {}
    for name, frame in partners.items():
        column = V5_PARTNERS[name]
        joined = agreement[["image_id", "dx", "agreement_score"]].merge(frame[["image_id", column]], on="image_id")
        within = {c: diag.spearman(joined.loc[joined["dx"] == c, "agreement_score"],
                                   joined.loc[joined["dx"] == c, column]) for c in CLASSES}
        n_strong = sum(1 for r in within.values() if not np.isnan(r) and abs(r) >= V5_MIN_ABS_RHO)
        pairs[f"agreement~{name}"] = {"within_class_spearman": within, "n_classes_abs_rho_at_least": n_strong,
                                      "redundant": bool(n_strong >= V5_MIN_CLASSES)}
    return {"pairs": pairs, "min_abs_rho": V5_MIN_ABS_RHO, "min_classes": V5_MIN_CLASSES,
            "passed": not any(p["redundant"] for p in pairs.values())}


# ==============================================================================================
# The run
# ==============================================================================================


def judge_id(weights: np.ndarray, bias: np.ndarray, inputs: dict) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(weights, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(bias, dtype=np.float64).tobytes())
    digest.update(repr(sorted(inputs.items())).encode())
    return "ham10000-judge-j1:" + digest.hexdigest()


def run(separability_dir: Path, real_dir: Path, metadata_path: Path, candidates_path: Path,
        similarity_dir: Path, v3_dir: Path, output_dir: Path, expected_n: int = diag.EXPECTED_CANDIDATES) -> dict:
    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    metadata = pd.read_csv(metadata_path)
    x_train, y_train, x_val, y_val, x_syn, syn_ids, syn_dx, info = training_data(separability_dir, real_dir, metadata)

    weights, bias = probe.fit_probe(x_train, y_train, probe._class_weights(y_train))
    val_pred = probe.predict(x_val, weights, bias)
    val_counts = probe.recall_counts(y_val, val_pred)
    v1 = probe.gate(val_counts)

    candidates = pd.read_csv(candidates_path)
    pool_dx = dict(zip(candidates["image_id"].astype(str), candidates["dx"].map(normalize_diagnosis)))
    if set(syn_ids) != set(pool_dx) or len(pool_dx) != expected_n:
        raise SystemExit("the synthetic embeddings and all_candidates.csv do not hold the same images.")
    if any(pool_dx[i] != d for i, d in zip(syn_ids, syn_dx)):
        raise SystemExit("the embeddings' dx disagrees with all_candidates.csv.")
    probabilities = softmax(np.asarray(x_syn, dtype=np.float64) @ weights + bias)
    scores = compute_agreement_scores(probabilities, list(syn_dx), stage3, labels=CLASSES)
    scores.insert(0, "image_id", syn_ids)
    frame = scores.assign(dx=list(syn_dx))

    q1 = diag.q1_agreement(frame, val_counts)
    q2 = diag.q2_mel_to_nv(frame)
    v4 = v4_sink(frame)
    v3 = diag.load_version(v3_dir, "v3", expected_n)
    similarity = pd.read_parquet(similarity_dir / "similarity_scores.parquet")
    v5 = v5_not_a_restatement(frame, {"similarity": similarity, "uncertainty": v3, "explainability": v3})

    # --- the agreement artifact, in the signal pipeline's own schema, in a new directory ---
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "agreement_scores.parquet"
    scores.to_parquet(artifact, index=False)
    inputs = {"embeddings_sha256": sha256_file(separability_dir / "embeddings.npz"),
              "real_embeddings_sha256": sha256_file(real_dir / "real_embeddings.npz"),
              "metadata_sha256": sha256_file(metadata_path),
              "candidates_csv_sha256": sha256_file(candidates_path)}
    jid = judge_id(weights, bias, inputs)
    write_json(output_dir / "agreement_scores.provenance.json", {
        "schema_version": 2, "signal": "agreement", "dataset": "ham10000", "split_namespace": "ham-stratified-v1",
        "n_rows": int(len(scores)), "n_candidates_scored": int(len(scores)),
        "candidates_csv_sha256": inputs["candidates_csv_sha256"], "git_commit_hash": get_git_commit_hash(),
        "judge_id": jid, "judge": "J1 (Amendment 6): DINOv2 linear probe",
        "judge_trained_on_splits": list(TRAIN_SPLITS), "judge_validated_on_split": VALIDATION_SPLIT,
        "rival_confidence_threshold": float(stage3.signals.agreement.rival_confidence_threshold),
        "penalty_weight": float(stage3.signals.agreement.penalty_weight),
        "probabilities": "softmax of the logistic-regression logits", "columns": list(scores.columns),
        "parquet_sha256": sha256_file(artifact),
    })

    # --- V6: the corrected Go/No-Go on the five-signal set, in a scratch copy under output_dir ---
    gate_dir = output_dir / "gonogo_signal_set"
    if gate_dir.exists():
        shutil.rmtree(gate_dir)
    gate_dir.mkdir()
    for signal, folder in (("similarity", similarity_dir), ("iqa", similarity_dir), ("uncertainty", v3_dir),
                           ("explainability", v3_dir), ("agreement", output_dir)):
        for suffix in ("parquet", "provenance.json"):
            shutil.copy2(folder / f"{signal}_scores.{suffix}", gate_dir / f"{signal}_scores.{suffix}")
    frames, absent = gonogo.load_artifacts(gate_dir)
    pool_hash = gonogo.assert_one_candidate_pool(gate_dir, frames)
    diagnoses = pd.Series(list(candidates["dx"].map(normalize_diagnosis)), index=candidates["image_id"].astype(str))
    per_signal = gonogo.evaluate(frames, diagnoses, gate_dir, stage3)
    gate_report = gonogo.build_report(per_signal, absent, gate_dir, "ham-stratified-v1", pool_hash)
    v6 = {"outcome": per_signal["agreement"]["outcome"], "reason": per_signal["agreement"]["reason"],
          "surviving_signals": gate_report["surviving_signals"],
          "passed": per_signal["agreement"]["outcome"] == gonogo.OUTCOME_INCLUDE}
    write_json(output_dir / "gonogo_report.json", gate_report)

    criteria = {"V1_real_validity": v1["passed"], "V2_q1_sensible": q1["sensible"],
                "V3_q2_not_influential": not q2["influential"], "V4_no_class_sink": v4["passed"],
                "V5_not_a_restatement": v5["passed"], "V6_gonogo_include": v6["passed"]}
    accepted = all(criteria.values())
    report = {
        "judge": "J1", "judge_id": jid, "criteria_document": "docs/ham10000_v2_signal_criteria.md, Amendment 6",
        "inputs": {**inputs, **info}, "git_commit_hash": get_git_commit_hash(),
        "settings": {"l2_penalty": probe.L2_PENALTY, "lbfgs_tolerance": probe.LBFGS_TOLERANCE,
                     "lbfgs_max_iterations": probe.LBFGS_MAX_ITERATIONS,
                     "class_weights": [float(w) for w in probe._class_weights(y_train)]},
        "criteria": criteria, "accepted": accepted,
        "decision": ("J1's agreement replaces the V3a agreement in ASISM v2" if accepted else
                     "agreement is left out of ASISM v2; no other judge is tried on this pool"),
        "v1_real_validity": v1, "validation_recall_counts": {k: list(v) for k, v in val_counts.items()},
        "q1": q1, "q2": q2, "v4_sink": v4, "v5_restatement": v5, "v6_gonogo": v6,
        "confusion_validation": confusion(y_val, val_pred),
        "confusion_synthetic": confusion(np.array([CLASSES.index(d) for d in syn_dx]),
                                         np.array([CLASSES.index(d) for d in frame["agreement_predicted_diagnosis"]])),
        "classes": CLASSES,
    }
    write_json(output_dir / "judge_j1.json", report)
    (output_dir / "judge_j1.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: dict) -> str:
    c, q1, q2, v4, v5, v6 = (report[k] for k in ("criteria", "q1", "q2", "v4_sink", "v5_restatement", "v6_gonogo"))
    lines = ["# HAM10000 agreement judge J1", "",
             "Criteria: `docs/ham10000_v2_signal_criteria.md`, Amendment 6 (applied as written).", "",
             f"**{'ACCEPTED' if report['accepted'] else 'NOT ACCEPTED'}**: {report['decision']}.", "",
             "| Criterion | Passed |", "|---|---|"]
    lines += [f"| {name} | {passed} |" for name, passed in c.items()]
    v1 = report["v1_real_validity"]
    lines += ["", f"- V1: classifier_val balanced accuracy {v1['balanced_accuracy']:.3f}; zero-recall classes "
              f"{v1['zero_recall_classes'] or 'none'}",
              "", "| Class | classifier_val recall | synthetic match | normalised |", "|---|---|---|---|"]
    for name in CLASSES:
        k, n = report["validation_recall_counts"][name]
        e = q1["per_class"][name]
        lines.append(f"| {name} | {k}/{n} ({k / n:.1%}) | {diag._ci(e)} | {diag._num(e['normalised_match_rate'], 2)} |")
    lines += ["", f"- V2 (Q1): classes at or below chance {q1['n_classes_at_or_below_chance']}; Spearman "
              f"{diag._num(q1['spearman_match_vs_real_recall'])}; sensible = {q1['sensible']}",
              f"- V3 (Q2): mel predicted nv {diag._ci(q2['mel_predicted_nv'])}; nv share of mismatches "
              f"{diag._ci(q2['nv_share_of_mismatches'])}; influential = {q2['influential']}",
              f"- V4: largest sink {v4['largest']} at {v4['largest_share']:.1%} of {v4['n_mismatches']} mismatches; "
              f"shares {({k: round(v, 3) for k, v in v4['shares'].items()})}"]
    for pair, entry in v5["pairs"].items():
        rhos = {k: round(v, 2) for k, v in entry["within_class_spearman"].items()}
        lines.append(f"- V5 {pair}: {entry['n_classes_abs_rho_at_least']} classes with |rho| >= {v5['min_abs_rho']}; "
                     f"redundant = {entry['redundant']}; {rhos}")
    lines.append(f"- V6: agreement {v6['outcome']} ({v6['reason']}); surviving {v6['surviving_signals']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for name in ("embeddings-dir", "real-embeddings-dir", "metadata", "candidates", "similarity-dir",
                 "v3-signals-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    a = parser.parse_args()
    report = run(a.embeddings_dir, a.real_embeddings_dir, a.metadata, a.candidates, a.similarity_dir,
                 a.v3_signals_dir, a.output_dir)
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
