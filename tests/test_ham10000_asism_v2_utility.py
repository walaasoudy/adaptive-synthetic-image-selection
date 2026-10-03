"""The utility measurement runner: refusal without approval, resumable grid, G1 written once.

The classifier is replaced by a function of the subset's signals, so nothing is trained and no
HAM10000 image or split is read.
"""
import json
import zlib

import numpy as np
import pandas as pd
import pytest

from scripts.asism_v2.contracts import fingerprint
from scripts.asism_v2.features import SIGNALS
from scripts.asism_v2.prereg import load_prereg
from scripts.asism_v2.supervision import build_plan, freeze_plan, ids_sha256
from scripts.followup import ham10000_asism_v2_utility as utility

NAMESPACE = "ham-stratified-v1"
POOL_COUNTS = {"akiec": 486, "bcc": 400, "bkl": 273, "df": 885, "mel": 276, "nv": 112, "vasc": 736}
HASHES = {"outcome_split": "o" * 64, "real_train_split": "r" * 64, "measurement_code": "c" * 64}


def _pool():
    rng = np.random.default_rng(5)
    return pd.DataFrame([{"image_id": f"{dx}-{j:04d}", "dx": dx, **dict(zip(SIGNALS, rng.normal(0, 1, 4)))}
                         for dx, n in POOL_COUNTS.items() for j in range(n)])


def _frozen(tmp_path):
    pool = _pool()
    report = {"safe_pool": len(pool), "safe_pool_ids_sha256": ids_sha256(pool.image_id),
              "candidates_csv_sha256": "a" * 64, "signal_artifact_sha256": {"similarity": "s" * 64}}
    freeze_plan(tmp_path, build_plan(pool, load_prereg(), report))
    return pool


def _fake(pool, noise, calls=None, stop_after=None):
    """A stand-in for the proxy classifier: utility is a function of the subset, plus seed noise."""
    signal = pool.set_index("image_id")[SIGNALS[0]]

    def measure(train, tuning, proxy, seed, device):
        if stop_after is not None and calls is not None and len(calls) >= stop_after:
            raise KeyboardInterrupt
        if calls is not None:
            calls.append(seed)
        assert proxy.resolution == 224 and proxy.max_steps == 300
        ids = [record["image_id"] for record in train]
        rng = np.random.default_rng([seed, zlib.crc32("".join(ids).encode())])
        value = 0.80 + 0.02 * np.log1p(len(ids)) + 0.02 * float(signal.loc[ids].mean()) + rng.normal(0, noise)
        return {"macro_auroc_ovr": float(np.clip(value, 0, 1))}

    inputs = {"real": [], "tuning": [], "hashes": HASHES,
              "candidates": {i: {"image_id": i} for i in pool.image_id}}
    return measure, inputs


def test_the_gpu_measurement_is_refused_until_it_is_approved(tmp_path):
    pool = _frozen(tmp_path)
    measure, inputs = _fake(pool, 0.002)
    assert utility.MEASUREMENT_APPROVED is False
    with pytest.raises(utility.UtilityError, match="NOT been approved"):
        utility.run_measure(NAMESPACE, None, True, out_dir=tmp_path, measure_fn=measure, inputs=inputs)
    assert not (tmp_path / utility.FIT_RUNS).exists() and not (tmp_path / utility.INPUTS_NAME).exists()


def test_approval_alone_is_not_enough_without_the_flag(tmp_path, monkeypatch):
    pool = _frozen(tmp_path)
    measure, inputs = _fake(pool, 0.002)
    monkeypatch.setattr(utility, "MEASUREMENT_APPROVED", True)
    with pytest.raises(utility.UtilityError, match="i-understand-this-trains-real-models"):
        utility.run_measure(NAMESPACE, None, False, out_dir=tmp_path, measure_fn=measure, inputs=inputs)


def test_the_grid_is_measured_resumed_and_gated_once(tmp_path, monkeypatch):
    pool = _frozen(tmp_path)
    monkeypatch.setattr(utility, "MEASUREMENT_APPROVED", True)
    calls = []
    measure, inputs = _fake(pool, 0.002, calls, stop_after=137)
    with pytest.raises(KeyboardInterrupt):                               # the pod dies mid-grid
        utility.run_measure(NAMESPACE, None, True, out_dir=tmp_path, measure_fn=measure, inputs=inputs)
    measure, inputs = _fake(pool, 0.002, calls)
    utility.run_measure(NAMESPACE, None, True, out_dir=tmp_path, measure_fn=measure, inputs=inputs)
    assert len(calls) == 1000                                            # no cell measured twice
    fit = [json.loads(line) for line in (tmp_path / utility.FIT_RUNS).read_text().splitlines()]
    held = [json.loads(line) for line in (tmp_path / utility.TEST_RUNS).read_text().splitlines()]
    assert len(fit) == 800 and len(held) == 200
    assert {row["role"] for row in fit} == {"train", "validation"} and {row["role"] for row in held} == {"test"}
    assert sorted({row["seed"] for row in fit}) == [42, 43, 44, 45, 46]
    assert utility.run_measure(NAMESPACE, None, True, out_dir=tmp_path, measure_fn=measure,
                               inputs=inputs)["completed"] == 1000 and len(calls) == 1000

    (tmp_path / utility.TEST_RUNS).rename(tmp_path / "hidden")           # G1 must not need the test file
    gate = utility.run_gate(NAMESPACE, tmp_path)
    assert gate["passed"] and gate["subsets"] == 160 and gate["repeats"] == 5 and gate["test_subsets_read"] is False
    assert set(gate["within_size_not_gating"]) == {"125", "250", "500", "1000"}
    with pytest.raises(FileExistsError):
        utility.run_gate(NAMESPACE, tmp_path)
    (tmp_path / "hidden").rename(tmp_path / utility.TEST_RUNS)

    rows, protocol = utility.fit_measurements(tmp_path)
    assert protocol["instrument_decision"] == "accepted" and len(protocol["instrument_gate_sha256"]) == 64
    assert {row["protocol_sha256"] for row in rows} == {fingerprint(protocol)}
    assert len(utility.heldout_measurements(tmp_path)) == 200


def test_changed_inputs_are_not_mixed_into_a_started_grid(tmp_path, monkeypatch):
    pool = _frozen(tmp_path)
    monkeypatch.setattr(utility, "MEASUREMENT_APPROVED", True)
    calls = []
    measure, inputs = _fake(pool, 0.002, calls, stop_after=3)
    with pytest.raises(KeyboardInterrupt):
        utility.run_measure(NAMESPACE, None, True, out_dir=tmp_path, measure_fn=measure, inputs=inputs)
    changed = {**inputs, "hashes": {**HASHES, "outcome_split": "x" * 64}}
    with pytest.raises(RuntimeError, match="changed since the first run"):
        utility.run_measure(NAMESPACE, None, True, out_dir=tmp_path, measure_fn=measure, inputs=changed)


def test_unrepeatable_labels_fail_g1_and_no_fit_follows(tmp_path, monkeypatch):
    pool = _frozen(tmp_path)
    monkeypatch.setattr(utility, "MEASUREMENT_APPROVED", True)
    measure, inputs = _fake(pool, 0.5)
    utility.run_measure(NAMESPACE, None, True, out_dir=tmp_path, measure_fn=measure, inputs=inputs)
    assert not utility.run_gate(NAMESPACE, tmp_path)["passed"]
    with pytest.raises(utility.UtilityError, match="G1 failed"):
        utility.fit_measurements(tmp_path)


def test_an_incomplete_grid_is_not_gated(tmp_path, monkeypatch):
    pool = _frozen(tmp_path)
    monkeypatch.setattr(utility, "MEASUREMENT_APPROVED", True)
    measure, inputs = _fake(pool, 0.002, [], stop_after=400)
    with pytest.raises(KeyboardInterrupt):
        utility.run_measure(NAMESPACE, None, True, out_dir=tmp_path, measure_fn=measure, inputs=inputs)
    with pytest.raises(ValueError, match="Incomplete"):
        utility.run_gate(NAMESPACE, tmp_path)
    assert not (tmp_path / utility.GATE_NAME).exists()
