#!/usr/bin/env python3
"""P1, the independent probe of Amendment 3 (docs/ham10000_v2_signal_criteria.md).

Does a classifier trained on real images only, from DINOv2 features instead of DenseNet's, also read
synthetic mel as nv? Everything below is fixed by the amendment, before the probe existed:

    features   the separability check's DINOv2 embeddings (sha256-checked), L2-normalised
    model      multinomial logistic regression, inverse-frequency class weights (as v2 and v3),
               weighted mean cross-entropy + 1e-4 * ||W||^2 (bias not penalised), full-batch
               L-BFGS from zeros, tolerance 1e-9, at most 1,000 iterations
    real CV    five lesion-grouped folds on real gen_train (lesion_grouped_folds, seed 42)
    gate       out-of-fold balanced accuracy > 0.478 and no class with zero recall (criteria A, B)
    synthetic  refit on all real images, predict the 3,168 candidates, then Q1 and Q2 exactly as
               the v3 diagnostic computes them, Q1 normalised by the probe's own out-of-fold recall

The verdict is read on Q2 alone. D1 (near or far misses of v3) and D2 (does v3 agree with DINOv2)
are descriptive and decide nothing. Nothing is written outside the output directory; no signal,
threshold, ranking, selection or earlier artifact is touched. CPU only.

Usage (after approval only):
    python scripts/asism/ham10000_independent_probe.py \
        --embeddings-dir <dir with embeddings.npz + provenance.json> \
        --metadata <HAM10000_metadata.csv> \
        --v3-signals-dir <dir with the v3 *_scores.parquet + provenance> \
        --candidates <all_candidates.csv> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.asism import ham10000_v2_signal_diagnostics as diag  # noqa: E402
from scripts.asism.ham10000_synthetic_separability import l2_normalise, load_embeddings  # noqa: E402
from scripts.classify.ham10000_train_auxiliary_classifier import lesion_grouped_folds  # noqa: E402
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS, normalize_diagnosis  # noqa: E402
from scripts.utils.ham10000_classifier import class_weights_from_records  # noqa: E402
from scripts.utils.manifest import get_git_commit_hash, sha256_file, write_json  # noqa: E402

CLASSES: list[str] = list(CLASSIFIER_TARGET_LABELS)
SEED = 42
N_FOLDS = 5
N_BOOTSTRAP = 1000
L2_PENALTY = 1e-4          # the weight decay of the v2 and v3 recipe; one value, not tuned
LBFGS_TOLERANCE = 1e-9
LBFGS_MAX_ITERATIONS = 1000
# Acceptance criteria A and B of the auxiliary classifier (configs/ham10000_stage3.yaml,
# min_balanced_accuracy_exclusive), applied to the probe's out-of-fold predictions.
GATE_MIN_BALANCED_ACCURACY_EXCLUSIVE = 0.478


# ==============================================================================================
# The probe
# ==============================================================================================


def fit_probe(x: np.ndarray, y_index: np.ndarray, class_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(W, b) of the weighted, L2-penalised multinomial logistic regression, float64, from zeros."""
    import torch

    features = torch.as_tensor(x, dtype=torch.float64)
    targets = torch.as_tensor(y_index, dtype=torch.long)
    weights = torch.zeros((features.shape[1], len(CLASSES)), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(len(CLASSES), dtype=torch.float64, requires_grad=True)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.as_tensor(class_weights, dtype=torch.float64))
    optimiser = torch.optim.LBFGS(
        [weights, bias], lr=1.0, max_iter=LBFGS_MAX_ITERATIONS, max_eval=LBFGS_MAX_ITERATIONS * 2,
        tolerance_grad=0.0, tolerance_change=LBFGS_TOLERANCE, history_size=20,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimiser.zero_grad()
        loss = loss_fn(features @ weights + bias, targets) + L2_PENALTY * (weights * weights).sum()
        loss.backward()
        return loss

    optimiser.step(closure)
    return weights.detach().numpy(), bias.detach().numpy()


def predict(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Argmax class index."""
    return np.argmax(np.asarray(x, dtype=np.float64) @ weights + bias, axis=1)


def _class_weights(y_index: np.ndarray) -> np.ndarray:
    return class_weights_from_records([{"class_index": int(c)} for c in y_index], n_classes=len(CLASSES))


def recall_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, tuple[int, int]]:
    return {name: (int(((y_true == i) & (y_pred == i)).sum()), int((y_true == i).sum()))
            for i, name in enumerate(CLASSES)}


def cross_validate(x: np.ndarray, y_index: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Out-of-fold predicted class index for every real image."""
    predicted = np.full(len(y_index), -1, dtype=int)
    for fold in range(N_FOLDS):
        held = folds == fold
        weights, bias = fit_probe(x[~held], y_index[~held], _class_weights(y_index[~held]))
        predicted[held] = predict(x[held], weights, bias)
    return predicted


def gate(counts: dict[str, tuple[int, int]]) -> dict:
    recalls = {name: correct / support for name, (correct, support) in counts.items()}
    balanced = float(np.mean(list(recalls.values())))
    zero = sorted(name for name, value in recalls.items() if value == 0)
    passed = bool(balanced > GATE_MIN_BALANCED_ACCURACY_EXCLUSIVE and not zero)
    return {"balanced_accuracy": balanced, "per_class_recall": recalls, "zero_recall_classes": zero,
            "min_balanced_accuracy_exclusive": GATE_MIN_BALANCED_ACCURACY_EXCLUSIVE, "passed": passed}


def decide_p1(gate_result: dict, q2: dict) -> dict:
    """The P1 decision table of Amendment 3, read on Q2 alone."""
    if not gate_result["passed"]:
        return {"verdict": "not_informative",
                "next": "recorded as it is; Walaa decides; nothing is retried with other settings"}
    if not q2["influential"]:
        return {"verdict": "classifier_specific",
                "next": "a classifier-side change, chosen and written in the criteria before it is tried"}
    held = [name for name, met in (
        ("mel predicted nv", q2["mel_predicted_nv"]["rate"] >= diag.CRITERIA["q2_max_mel_to_nv_share"]),
        ("nv share of mismatches", q2["nv_share_of_mismatches"]["rate"] >= diag.CRITERIA["q2_max_nv_share_of_mismatches"]),
    ) if met]
    return {"verdict": "class_fidelity_likelier", "conditions_held": held,
            "rests_on_one_condition": len(held) == 1,
            "next": "a generator-side step, chosen and written in the criteria before it is tried"}


# ==============================================================================================
# Descriptive reads (decide nothing)
# ==============================================================================================


def d1_near_or_far(v3: pd.DataFrame) -> dict:
    rows = v3[(v3["dx"] == "mel") & (v3["agreement_predicted_diagnosis"] == "nv")]
    if (rows["agreement_best_rival_diagnosis"] != "nv").any():
        raise SystemExit("D1: a mel image predicted nv has a best rival other than nv.")
    p_mel, p_nv = rows["agreement_intended_prob"].to_numpy(float), rows["agreement_best_rival_prob"].to_numpy(float)
    runner_up = p_mel > 1.0 - p_nv - p_mel
    quartiles = np.percentile(p_mel, [25, 50, 75]) if len(rows) else [float("nan")] * 3
    return {"n": int(len(rows)),
            "p_mel_quartiles": [float(q) for q in quartiles],
            "p_nv_median": float(np.median(p_nv)) if len(rows) else float("nan"),
            "margin_median": float(rows["agreement_margin"].median()) if len(rows) else float("nan"),
            "mel_certainly_runner_up": diag.proportion(int(runner_up.sum()), int(len(rows)))}


def d2_dinov2_agreement(x_real, y_real, x_syn, syn_ids, v3: pd.DataFrame, rng: np.random.Generator) -> dict:
    centroids = {name: l2_normalise(x_real[y_real == name].mean(axis=0, keepdims=True))[0] for name in ("mel", "nv")}
    score = dict(zip(syn_ids, x_syn @ centroids["mel"] - x_syn @ centroids["nv"]))
    mel = v3[v3["dx"] == "mel"]
    matched = np.array([score[i] for i in mel.loc[mel["agreement_is_argmax_match"].astype(bool), "image_id"]])
    as_nv = np.array([score[i] for i in mel.loc[mel["agreement_predicted_diagnosis"] == "nv", "image_id"]])
    out = {"n_read_mel": int(len(matched)), "n_read_nv": int(len(as_nv)),
           "median_read_mel": float(np.median(matched)) if len(matched) else float("nan"),
           "median_read_nv": float(np.median(as_nv)) if len(as_nv) else float("nan")}
    if len(matched) and len(as_nv):
        draws = [np.median(rng.choice(matched, len(matched))) - np.median(rng.choice(as_nv, len(as_nv)))
                 for _ in range(N_BOOTSTRAP)]
        out["difference"] = out["median_read_mel"] - out["median_read_nv"]
        out["difference_ci95"] = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    return out


# ==============================================================================================
# Inputs and the run
# ==============================================================================================


def real_records(image_ids: np.ndarray, dx: np.ndarray, metadata: pd.DataFrame) -> list[dict]:
    meta = metadata.assign(image_id=metadata["image_id"].astype(str)).set_index("image_id")
    missing = sorted(set(image_ids) - set(meta.index))
    if missing:
        raise SystemExit(f"{len(missing)} real image(s) are not in the metadata (e.g. {missing[:3]}).")
    records = []
    for image_id, name in zip(image_ids, dx):
        if normalize_diagnosis(meta.at[image_id, "dx"]) != name:
            raise SystemExit(f"{image_id}: metadata dx disagrees with the embeddings' dx.")
        records.append({"image_id": image_id, "class_index": CLASSES.index(name),
                        "lesion_id": meta.at[image_id, "lesion_id"]})
    return records


def run(embeddings_dir: Path, metadata_path: Path, v3_dir: Path, candidates_path: Path,
        expected_n: int = diag.EXPECTED_CANDIDATES) -> dict:
    x, y, source = load_embeddings(embeddings_dir)
    ids = np.load(embeddings_dir / "embeddings.npz", allow_pickle=False)["image_id"].astype(str)
    x = l2_normalise(x)
    real, syn = source == "real", source == "synthetic"
    if pd.Series(ids[real]).duplicated().any() or pd.Series(ids[syn]).duplicated().any():
        raise SystemExit("duplicate image ids in the embeddings.")

    records = real_records(ids[real], y[real], pd.read_csv(metadata_path))
    y_real = np.array([r["class_index"] for r in records])
    folds = lesion_grouped_folds(records, N_FOLDS, SEED)
    oof = cross_validate(x[real], y_real, folds)
    oof_counts = recall_counts(y_real, oof)
    gate_result = gate(oof_counts)

    weights, bias = fit_probe(x[real], y_real, _class_weights(y_real))
    predicted = np.array(CLASSES, dtype=object)[predict(x[syn], weights, bias)]
    candidates = pd.read_csv(candidates_path)
    pool_dx = dict(zip(candidates["image_id"].astype(str), candidates["dx"].map(normalize_diagnosis)))
    if set(ids[syn]) != set(pool_dx) or len(pool_dx) != expected_n:
        raise SystemExit("the synthetic embeddings and all_candidates.csv do not hold the same images.")
    probe_frame = pd.DataFrame({"image_id": ids[syn], "dx": y[syn], "agreement_predicted_diagnosis": predicted})
    if (probe_frame["dx"] != probe_frame["image_id"].map(pool_dx)).any():
        raise SystemExit("the embeddings' dx disagrees with all_candidates.csv.")
    probe_frame["agreement_is_argmax_match"] = probe_frame["agreement_predicted_diagnosis"] == probe_frame["dx"]

    q1 = diag.q1_agreement(probe_frame, oof_counts)
    q2 = diag.q2_mel_to_nv(probe_frame)

    v3 = diag.attach_class(diag.load_version(v3_dir, "v3", expected_n), candidates, "v3")
    rng = np.random.default_rng(SEED)
    return {
        "inputs": {"embeddings_sha256": sha256_file(embeddings_dir / "embeddings.npz"),
                   "metadata_sha256": sha256_file(metadata_path),
                   "candidates_sha256": sha256_file(candidates_path),
                   "v3_agreement_sha256": sha256_file(v3_dir / "agreement_scores.parquet"),
                   "n_real": int(real.sum()), "n_synthetic": int(syn.sum())},
        "settings": {"seed": SEED, "folds": N_FOLDS, "l2_penalty": L2_PENALTY,
                     "lbfgs_tolerance": LBFGS_TOLERANCE, "lbfgs_max_iterations": LBFGS_MAX_ITERATIONS,
                     "class_weights": [float(w) for w in _class_weights(y_real)]},
        "git_commit_hash": get_git_commit_hash(),
        "gate": gate_result,
        "oof_recall_counts": {k: list(v) for k, v in oof_counts.items()},
        "q1": q1, "q2": q2,
        "decision": decide_p1(gate_result, q2),
        "descriptive": {"d1": d1_near_or_far(v3),
                        "d2": d2_dinov2_agreement(x[real], y[real], x[syn], ids[syn], v3, rng)},
    }


def render_markdown(report: dict) -> str:
    g, q1, q2, d = report["gate"], report["q1"], report["q2"], report["decision"]
    lines = ["# HAM10000 P1: independent DINOv2 probe", "",
             "Criteria: `docs/ham10000_v2_signal_criteria.md`, Amendment 3 (applied as written).", "",
             f"**Decision: {d['verdict']}**. {d['next']}.", ""]
    if d.get("conditions_held"):
        lines += [f"- Q2 conditions held: {', '.join(d['conditions_held'])}"
                  + (" (rests on one condition only)" if d["rests_on_one_condition"] else ""), ""]
    lines += ["## Gate (out-of-fold, real gen_train)", "",
              f"- balanced accuracy {g['balanced_accuracy']:.3f} (must be > {g['min_balanced_accuracy_exclusive']}); "
              f"zero-recall classes {g['zero_recall_classes'] or 'none'}; **passed = {g['passed']}**", "",
              "| Class | real recall (OOF) | synthetic match | normalised |", "|---|---|---|---|"]
    for name in CLASSES:
        c, s = report["oof_recall_counts"][name]
        e = q1["per_class"][name]
        lines.append(f"| {name} | {c}/{s} ({c / s:.1%}) | {diag._ci(e)} | {diag._num(e['normalised_match_rate'], 2)} |")
    lines += ["", f"- Q1 (reported, not decisive): classes at or below chance {q1['n_classes_at_or_below_chance']}; "
              f"Spearman {diag._num(q1['spearman_match_vs_real_recall'])}; sensible = {q1['sensible']}",
              f"- Q2: mel predicted nv {diag._ci(q2['mel_predicted_nv'])}; nv share of mismatches "
              f"{diag._ci(q2['nv_share_of_mismatches'])}; mismatches predicted as {q2['mismatches_predicted_as']}; "
              f"**influential = {q2['influential']}**", "",
              "## Descriptive (decide nothing)", ""]
    d1, d2 = report["descriptive"]["d1"], report["descriptive"]["d2"]
    lines.append(f"- D1, v3's {d1['n']} synthetic mel predicted nv: p(mel) quartiles "
                 f"{', '.join(diag._num(q) for q in d1['p_mel_quartiles'])}; median p(nv) {diag._num(d1['p_nv_median'])}; "
                 f"mel certainly runner-up {diag._ci(d1['mel_certainly_runner_up'])}")
    if "difference" in d2:
        lines.append(f"- D2, cos(mel centroid) - cos(nv centroid): v3 read mel (n={d2['n_read_mel']}) median "
                     f"{diag._num(d2['median_read_mel'], 4)}; read nv (n={d2['n_read_nv']}) median "
                     f"{diag._num(d2['median_read_nv'], 4)}; difference {diag._num(d2['difference'], 4)} "
                     f"[{diag._num(d2['difference_ci95'][0], 4)}, {diag._num(d2['difference_ci95'][1], 4)}]")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--v3-signals-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.embeddings_dir, args.metadata, args.v3_signals_dir, args.candidates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "probe.json", report)
    (args.output_dir / "probe.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
