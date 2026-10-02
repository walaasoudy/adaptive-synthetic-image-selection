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

P1b (Amendment 4) is the same probe on the images v3 saw: trained on all of classifier_train, gated
and Q1-normalised on classifier_val, read on the same synthetic embeddings. Its real embeddings come
from `embed-real` (GPU pod), written to outputs/ham10000/diagnostics/independent_probe/<namespace>/.

Usage (after approval only):
    python scripts/asism/ham10000_independent_probe.py p1 \
        --embeddings-dir <dir with embeddings.npz + provenance.json> \
        --metadata <HAM10000_metadata.csv> \
        --v3-signals-dir <dir with the v3 *_scores.parquet + provenance> \
        --candidates <all_candidates.csv> --output-dir <dir>
    python scripts/asism/ham10000_independent_probe.py embed-real --namespace ham-stratified-v1
    python scripts/asism/ham10000_independent_probe.py p1b \
        --embeddings-dir <separability dir> --real-embeddings-dir <embed-real dir> \
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
# Amendment 4 (P1b): the v3 classifier's own training and selection splits.
P1B_TRAIN_SPLIT = "classifier_train"
P1B_GATE_SPLIT = "classifier_val"
PROBE_OUTPUT_SUBDIR = Path("outputs/ham10000/diagnostics/independent_probe")
ENCODER_FIELDS = ("encoder", "weights_repo_id", "weights_revision", "weights_sha256")
P1B_READING = {
    "classifier_specific": "the caveat is closed; the nv reading belongs to the DenseNet classifier; "
                           "a classifier-side change is chosen and written in an Amendment 5 before it is tried",
    "class_fidelity_likelier": "P1's result came from training on the generator's own images; the reading "
                               "is mixed; written as it is, and Walaa decides",
    "not_informative": "P1b says nothing; P1 stands, with the gen_train caveat; nothing is retried",
}


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


def synthetic_frame(ids: np.ndarray, dx: np.ndarray, predicted_index: np.ndarray,
                    candidates: pd.DataFrame, expected_n: int) -> pd.DataFrame:
    """The probe's reading of the pool, in the columns Q1 and Q2 read, checked against the pool."""
    pool_dx = dict(zip(candidates["image_id"].astype(str), candidates["dx"].map(normalize_diagnosis)))
    if set(ids) != set(pool_dx) or len(pool_dx) != expected_n:
        raise SystemExit("the synthetic embeddings and all_candidates.csv do not hold the same images.")
    frame = pd.DataFrame({"image_id": ids, "dx": dx,
                          "agreement_predicted_diagnosis": np.array(CLASSES, dtype=object)[predicted_index]})
    if (frame["dx"] != frame["image_id"].map(pool_dx)).any():
        raise SystemExit("the embeddings' dx disagrees with all_candidates.csv.")
    frame["agreement_is_argmax_match"] = frame["agreement_predicted_diagnosis"] == frame["dx"]
    return frame


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
    candidates = pd.read_csv(candidates_path)
    probe_frame = synthetic_frame(ids[syn], y[syn], predict(x[syn], weights, bias), candidates, expected_n)

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


# ==============================================================================================
# P1b (Amendment 4)
# ==============================================================================================


def embed_real(namespace: str, device: str | None = None, limit: int | None = None) -> dict:
    """DINOv2 embeddings of classifier_train and classifier_val, with the similarity signal's pinned
    encoder and embedding code, from the same preprocessed images the v3 classifier was trained and
    accepted on (records_from_split, which skips absent images exactly as v3's training did)."""
    from scripts.asism import ham10000_01_compute_signals as signals
    from scripts.utils.config import load_named_config
    from scripts.utils.ham10000_classifier import records_from_split

    stage1 = load_named_config("ham10000_stage1.yaml", "ham_stage1")
    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    splits_cfg = load_named_config("splits_ham10000.yaml", "ham_splits")
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    split_dir = Path(splits_cfg.paths.splits_root) / namespace
    images_root = Path(stage1.paths.images_dir) / namespace
    final_eval = signals.load_final_eval_image_ids(split_dir)

    rows, counts = [], {}
    for split in (P1B_TRAIN_SPLIT, P1B_GATE_SPLIT):
        frame = pd.read_csv(split_dir / f"{split}.csv")
        if limit:
            frame = frame.groupby("dx", group_keys=False).head(max(1, int(limit) // len(CLASSES)))
        records = records_from_split(frame, images_root / split)
        counts[split] = {"in_split": int(len(frame)), "embedded": int(len(records))}
        rows += [{**record, "split": split} for record in records]
    ids = [row["image_id"] for row in rows]
    leaked = sorted(set(ids) & final_eval)
    if leaked:
        raise SystemExit(f"{len(leaked)} image(s) are in final_eval_heldout (e.g. {leaked[:3]}).")
    if len(set(ids)) != len(ids):
        raise SystemExit("an image appears in both classifier_train and classifier_val.")

    similarity_cfg = stage3.signals.similarity
    encoder, transform, weights_path = signals.load_encoder(similarity_cfg, device)
    print(f"embedding {len(rows)} real images ({counts}) on {device}", flush=True)
    embeddings = np.asarray(signals.embed_images(
        encoder, transform, [Path(row["image_path"]) for row in rows], int(similarity_cfg.batch_size), device,
    ), dtype=np.float32)
    out_dir = Path(stage3.paths.project_root) / PROBE_OUTPUT_SUBDIR / namespace
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "real_embeddings.npz", embeddings=embeddings, image_id=np.array(ids),
                        dx=np.array([CLASSES[row["class_index"]] for row in rows]),
                        split=np.array([row["split"] for row in rows]))
    provenance = {
        "namespace": namespace,
        "encoder": str(similarity_cfg.encoder),
        "weights_repo_id": str(similarity_cfg.weights_repo_id),
        "weights_revision": str(similarity_cfg.weights_revision),
        "weights_sha256": sha256_file(weights_path),
        "splits": counts,
        "limit": limit,
        "embeddings_sha256": sha256_file(out_dir / "real_embeddings.npz"),
        "git_commit_hash": get_git_commit_hash(),
    }
    write_json(out_dir / "provenance.json", provenance)
    return {"out_dir": str(out_dir), **provenance}


def load_real_embeddings(real_dir: Path, separability_dir: Path):
    """(embeddings, image_id, dx, split) of embed-real, after the hash and same-encoder checks."""
    from scripts.utils.manifest import read_json

    provenance = read_json(real_dir / "provenance.json")
    if provenance.get("embeddings_sha256") != sha256_file(real_dir / "real_embeddings.npz"):
        raise SystemExit("real_embeddings.npz does not match its provenance hash.")
    if provenance.get("limit"):
        raise SystemExit("the real embeddings are a smoke run (limit set); P1b reads full runs only.")
    other = read_json(separability_dir / "provenance.json")
    differing = [field for field in ENCODER_FIELDS if provenance.get(field) != other.get(field)]
    if differing:
        raise SystemExit(f"the two embedding files differ in {differing}; refusing to compare them.")
    data = np.load(real_dir / "real_embeddings.npz", allow_pickle=False)
    return data["embeddings"], data["image_id"].astype(str), data["dx"].astype(str), data["split"].astype(str)


def run_p1b(separability_dir: Path, real_dir: Path, candidates_path: Path,
            expected_n: int = diag.EXPECTED_CANDIDATES) -> dict:
    x, y, source = load_embeddings(separability_dir)
    ids = np.load(separability_dir / "embeddings.npz", allow_pickle=False)["image_id"].astype(str)
    syn = source == "synthetic"
    x_real, _, real_dx, split = load_real_embeddings(real_dir, separability_dir)
    x_real = l2_normalise(x_real)
    y_real = np.array([CLASSES.index(name) for name in real_dx])
    train, held = split == P1B_TRAIN_SPLIT, split == P1B_GATE_SPLIT
    if not train.any() or not held.any() or not (train | held).all():
        raise SystemExit(f"the real embeddings must hold {P1B_TRAIN_SPLIT} and {P1B_GATE_SPLIT} only.")

    weights, bias = fit_probe(x_real[train], y_real[train], _class_weights(y_real[train]))
    gate_counts = recall_counts(y_real[held], predict(x_real[held], weights, bias))
    gate_result = gate(gate_counts)
    frame = synthetic_frame(ids[syn], y[syn], predict(l2_normalise(x[syn]), weights, bias),
                            pd.read_csv(candidates_path), expected_n)
    q1 = diag.q1_agreement(frame, gate_counts)
    q2 = diag.q2_mel_to_nv(frame)
    decision = decide_p1(gate_result, q2)
    return {
        "probe": "P1b",
        "inputs": {"embeddings_sha256": sha256_file(separability_dir / "embeddings.npz"),
                   "real_embeddings_sha256": sha256_file(real_dir / "real_embeddings.npz"),
                   "candidates_sha256": sha256_file(candidates_path),
                   "n_train": int(train.sum()), "n_gate": int(held.sum()), "n_synthetic": int(syn.sum())},
        "settings": {"train_split": P1B_TRAIN_SPLIT, "gate_split": P1B_GATE_SPLIT, "l2_penalty": L2_PENALTY,
                     "lbfgs_tolerance": LBFGS_TOLERANCE, "lbfgs_max_iterations": LBFGS_MAX_ITERATIONS,
                     "class_weights": [float(w) for w in _class_weights(y_real[train])]},
        "git_commit_hash": get_git_commit_hash(),
        "gate": gate_result,
        "oof_recall_counts": {k: list(v) for k, v in gate_counts.items()},
        "q1": q1, "q2": q2,
        "decision": decision,
        "reading_with_p1": P1B_READING[decision["verdict"]],
    }


def render_markdown(report: dict) -> str:
    g, q1, q2, d = report["gate"], report["q1"], report["q2"], report["decision"]
    p1b = report.get("probe") == "P1b"
    lines = [f"# HAM10000 {'P1b' if p1b else 'P1'}: independent DINOv2 probe", "",
             f"Criteria: `docs/ham10000_v2_signal_criteria.md`, Amendment {4 if p1b else 3} (applied as written).", "",
             f"**Decision: {d['verdict']}**. {d['next']}.", ""]
    if p1b:
        lines += [f"Read with P1: {report['reading_with_p1']}.", ""]
    if d.get("conditions_held"):
        lines += [f"- Q2 conditions held: {', '.join(d['conditions_held'])}"
                  + (" (rests on one condition only)" if d["rests_on_one_condition"] else ""), ""]
    gate_label = "classifier_val, trained on classifier_train" if p1b else "out-of-fold, real gen_train"
    real_label = "real recall (classifier_val)" if p1b else "real recall (OOF)"
    lines += [f"## Gate ({gate_label})", "",
              f"- balanced accuracy {g['balanced_accuracy']:.3f} (must be > {g['min_balanced_accuracy_exclusive']}); "
              f"zero-recall classes {g['zero_recall_classes'] or 'none'}; **passed = {g['passed']}**", "",
              f"| Class | {real_label} | synthetic match | normalised |", "|---|---|---|---|"]
    for name in CLASSES:
        c, s = report["oof_recall_counts"][name]
        e = q1["per_class"][name]
        lines.append(f"| {name} | {c}/{s} ({c / s:.1%}) | {diag._ci(e)} | {diag._num(e['normalised_match_rate'], 2)} |")
    lines += ["", f"- Q1 (reported, not decisive): classes at or below chance {q1['n_classes_at_or_below_chance']}; "
              f"Spearman {diag._num(q1['spearman_match_vs_real_recall'])}; sensible = {q1['sensible']}",
              f"- Q2: mel predicted nv {diag._ci(q2['mel_predicted_nv'])}; nv share of mismatches "
              f"{diag._ci(q2['nv_share_of_mismatches'])}; mismatches predicted as {q2['mismatches_predicted_as']}; "
              f"**influential = {q2['influential']}**", ""]
    if "descriptive" not in report:
        return "\n".join(lines) + "\n"
    lines += ["## Descriptive (decide nothing)", ""]
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
    sub = parser.add_subparsers(dest="step", required=True)
    p1 = sub.add_parser("p1", help="Amendment 3: trained on real gen_train (CPU)")
    p1.add_argument("--embeddings-dir", type=Path, required=True)
    p1.add_argument("--metadata", type=Path, required=True)
    p1.add_argument("--v3-signals-dir", type=Path, required=True)
    p1.add_argument("--candidates", type=Path, required=True)
    p1.add_argument("--output-dir", type=Path, required=True)
    e = sub.add_parser("embed-real", help="Amendment 4: classifier_train + classifier_val embeddings (GPU pod)")
    e.add_argument("--namespace", required=True)
    e.add_argument("--device", default=None)
    e.add_argument("--limit", type=int, default=None, help="smoke runs only; P1b refuses them")
    b = sub.add_parser("p1b", help="Amendment 4: trained on classifier_train, gated on classifier_val (CPU)")
    b.add_argument("--embeddings-dir", type=Path, required=True)
    b.add_argument("--real-embeddings-dir", type=Path, required=True)
    b.add_argument("--candidates", type=Path, required=True)
    b.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.step == "embed-real":
        import json

        print(json.dumps(embed_real(args.namespace, args.device, args.limit), indent=2))
        return 0
    if args.step == "p1":
        report, name = run(args.embeddings_dir, args.metadata, args.v3_signals_dir, args.candidates), "probe"
    else:
        report, name = run_p1b(args.embeddings_dir, args.real_embeddings_dir, args.candidates), "probe_p1b"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / f"{name}.json", report)
    (args.output_dir / f"{name}.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
