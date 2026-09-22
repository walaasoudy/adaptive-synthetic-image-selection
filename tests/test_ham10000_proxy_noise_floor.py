"""The noise-floor diagnostic's statistics and its freeze-the-plan guard (no GPU, no images)."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.followup import ham10000_proxy_noise_floor as nf


def _v1_results(n=80):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "subset_id": [f"s{i:03d}" for i in range(n)],
        "size": np.resize([60, 120, 180], n),
        "real_only_balanced_accuracy": 0.5,
        "augmented_balanced_accuracy": 0.5 + rng.normal(0, 0.03, n),
    })


def _write_v1(tmp_path):
    v1_dir, out_dir = tmp_path / "v1", tmp_path / "out"
    v1_dir.mkdir()
    (v1_dir / "utility_results.jsonl").write_text("\n".join(json.dumps(r) for r in _v1_results().to_dict("records")))
    (v1_dir / "utility_subsets.jsonl").write_text(json.dumps({"subset_id": "s000", "image_ids": []}))
    return v1_dir, out_dir


def test_pick_subsets_takes_one_per_rank_bin_and_is_deterministic():
    v1 = _v1_results()
    chosen = nf.pick_subsets(v1)
    assert chosen == nf.pick_subsets(v1)
    assert len(set(chosen)) == nf.N_SUBSETS == 12
    ranked = v1.sort_values("augmented_balanced_accuracy")["subset_id"].to_numpy()
    for sid, bin_ids in zip(chosen, np.array_split(ranked, 12)):
        assert sid in bin_ids


def test_variance_components_recover_known_signal_and_noise():
    rng = np.random.default_rng(0)
    true_means = rng.normal(0, 0.04, 400)                        # between SD 0.04
    table = true_means[:, None] + rng.normal(0, 0.02, (400, 4))  # within SD 0.02
    comp = nf.variance_components(table)
    assert comp["between_sd"] == pytest.approx(0.04, rel=0.1)
    assert comp["within_sd"] == pytest.approx(0.02, rel=0.1)
    assert comp["icc_single_run"] == pytest.approx(0.8, abs=0.05)


def test_grouping_by_size_removes_a_pure_size_effect():
    # Subsets differ ONLY by size; within a size they are identical up to noise.
    rng = np.random.default_rng(2)
    sizes = np.resize([60, 120, 180], 300)
    table = (sizes / 1000.0)[:, None] + rng.normal(0, 0.01, (300, 4))
    assert nf.variance_components(table)["icc_single_run"] > 0.8
    assert nf.variance_components(table, groups=sizes)["icc_single_run"] < 0.1


def test_pure_noise_gives_low_icc():
    table = np.random.default_rng(3).normal(0.5, 0.05, (12, 4))
    result = nf.analyse_table(table)
    if result["between_variance"] <= 0:
        assert result["verdict"] == "no_detectable_signal"
    assert result["icc_single_run"] < 0.5


def test_repeats_and_ceiling_follow_spearman_brown():
    # ICC of one run = 0.5 -> R runs give R/(R+1): 0.8 is first reached at R = 4
    assert nf.repeats_needed(1.0, 1.0) == 4
    assert nf.correlation_ceiling(1.0, 1.0) == pytest.approx(np.sqrt(0.5))
    assert nf.correlation_ceiling(1.0, 1.0, repeats=4) == pytest.approx(np.sqrt(0.8))
    assert nf.verdict(1.0, 4) == "repeat"
    assert nf.repeats_needed(1.0, 9.0) is None                  # ICC 0.1 -> needs 36 runs
    assert nf.verdict(1.0, None) == "lengthen_proxy"
    assert nf.verdict(0.0, None) == "no_detectable_signal"


def test_v2_metric_rule():
    blocks = {"balanced_accuracy": {"verdict": "lengthen_proxy", "icc_single_run": 0.2, "repeats_needed": None},
              "macro_auroc_ovr": {"verdict": "repeat", "icc_single_run": 0.6, "repeats_needed": 3},
              "macro_f1": {"verdict": "repeat", "icc_single_run": 0.5, "repeats_needed": 4}}
    assert nf.choose_v2_metric(blocks) == {"metric": "macro_auroc_ovr", "repeats": 3,
                                           "reason": "highest single-run ICC among metrics whose verdict is 'repeat'"}
    for block in blocks.values():
        block["verdict"] = "lengthen_proxy"
    assert nf.choose_v2_metric(blocks)["metric"] == "balanced_accuracy"


def test_plan_is_frozen_and_cannot_be_redrawn(tmp_path):
    v1_dir, out_dir = _write_v1(tmp_path)
    assert nf.run_plan(v1_dir, out_dir)["status"] == "frozen"
    assert nf.run_plan(v1_dir, out_dir)["status"] == "already frozen, identical"
    plan = json.loads((out_dir / "noise_floor_plan.json").read_text())
    assert len(plan["chosen_subset_sizes"]) == 12
    plan["chosen_subsets"][0] = "tampered"
    (out_dir / "noise_floor_plan.json").write_text(json.dumps(plan))
    with pytest.raises(nf.NoiseFloorError):
        nf.run_plan(v1_dir, out_dir)


def test_analyze_uses_only_complete_seeds_and_every_metric(tmp_path):
    v1_dir, out_dir = _write_v1(tmp_path)
    chosen = nf.run_plan(v1_dir, out_dir)["chosen_subsets"]
    rng = np.random.default_rng(5)

    def row(sid, seed):
        return {"subset_id": sid, "seed": seed, "seconds": 60.0, "gpu": "test",
                "metrics": {m: float(0.5 + rng.normal(0, 0.02)) for m in nf.METRICS}}

    rows = [row(sid, seed) for seed in (42, 43, 44) for sid in [nf.BASELINE_ID, *chosen]]
    rows.append(row(chosen[0], 45))                               # seed 45 unfinished
    (out_dir / "noise_floor_runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    nf.run_analyze(v1_dir, out_dir)
    report = json.loads((out_dir / "noise_floor_report.json").read_text())
    assert report["complete_seeds"] == [42, 43, 44]
    assert report["post_hoc"] is True
    assert set(report["v1_style_label"]) == set(nf.METRICS)
    assert "same_size_only" in report["v1_style_label"]["balanced_accuracy"]
    assert report["timing"]["mean_seconds"] == 60.0


def test_proxy_config_must_equal_v1():
    nf.check_proxy_config({**nf.V1_PROXY, "seed": 42})           # identical: passes
    nf.check_proxy_config({**nf.V1_PROXY, "learning_rate": "1e-4"})  # same value, other spelling
    for key, other in (("max_steps", 1000), ("resolution", 512), ("pretrained_source", "random")):
        with pytest.raises(nf.NoiseFloorError, match=key):
            nf.check_proxy_config({**nf.V1_PROXY, key: other})


def test_plan_records_proxy_and_selection_basis(tmp_path):
    v1_dir, out_dir = _write_v1(tmp_path)
    nf.run_plan(v1_dir, out_dir)
    plan = json.loads((out_dir / "noise_floor_plan.json").read_text())
    assert plan["proxy"] == {**nf.V1_PROXY, "architecture": "densenet121"}
    assert "only to stratify coverage" in plan["selection_basis"]


def test_measure_inputs_refuse_to_mix_runs(tmp_path):
    path = tmp_path / "measure_inputs.json"
    inputs = {"git_commit_hash": "abc", "asism_tuning_heldout_csv_sha256": "111"}
    assert nf.freeze_measure_inputs(path, inputs) == "frozen"
    assert nf.freeze_measure_inputs(path, dict(inputs)) == "identical"
    with pytest.raises(nf.NoiseFloorError, match="asism_tuning_heldout_csv_sha256"):
        nf.freeze_measure_inputs(path, {**inputs, "asism_tuning_heldout_csv_sha256": "222"})


def test_report_labels_ceiling_and_same_size_as_diagnostics(tmp_path):
    v1_dir, out_dir = _write_v1(tmp_path)
    chosen = nf.run_plan(v1_dir, out_dir)["chosen_subsets"]
    rng = np.random.default_rng(7)
    rows = [{"subset_id": sid, "seed": seed, "metrics": {m: float(0.5 + rng.normal(0, 0.02)) for m in nf.METRICS}}
            for seed in nf.SEEDS for sid in [nf.BASELINE_ID, *chosen]]
    (out_dir / "noise_floor_runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    nf.run_analyze(v1_dir, out_dir)
    report = json.loads((out_dir / "noise_floor_report.json").read_text())
    assert len(rows) == 52
    assert "not a strict upper bound" in report["interpretation"]["correlation_ceiling"]
    same = report["v1_style_label"]["balanced_accuracy"]["same_size_only"]
    assert same["role"] == "sensitivity"
    assert sum(same["subsets_per_size"].values()) == 12


def test_v1_plan_is_unchanged_by_the_variant_option(tmp_path):
    v1_dir, out_dir = _write_v1(tmp_path)
    nf.run_plan(v1_dir, out_dir)                                  # default variant
    plan = json.loads((out_dir / "noise_floor_plan.json").read_text())
    assert "proxy_variant" not in plan                           # the round-1 plan stays byte-comparable
    assert nf.run_plan(v1_dir, out_dir, "v1")["status"] == "already frozen, identical"


def test_steps1500_plan_changes_only_the_step_budget(tmp_path):
    v1_dir, v1_out = _write_v1(tmp_path)
    long_out = tmp_path / "long"
    nf.run_plan(v1_dir, v1_out)
    nf.run_plan(v1_dir, long_out, "steps1500")
    short = json.loads((v1_out / "noise_floor_plan.json").read_text())
    long = json.loads((long_out / "noise_floor_plan.json").read_text())
    assert long["chosen_subsets"] == short["chosen_subsets"] and long["seeds"] == short["seeds"]
    assert long["proxy_variant"] == "steps1500"
    assert long["proxy"] == {**short["proxy"], "max_steps": 1500}
    assert sorted(k for k in set(long) | set(short) if long.get(k) != short.get(k)) == ["proxy", "proxy_variant"]
    with pytest.raises(nf.NoiseFloorError):                       # a frozen round-2 plan cannot become round 1
        nf.run_plan(v1_dir, long_out, "v1")


def test_unknown_variant_and_default_dirs():
    with pytest.raises(nf.NoiseFloorError):
        nf.variant_proxy("steps999")
    assert nf.default_out_name("v1") == "proxy_noise_floor"
    assert nf.default_out_name("steps1500") == "proxy_noise_floor_steps1500"


def test_stage4size_changes_only_resolution_and_steps(tmp_path):
    v1_dir, v1_out = _write_v1(tmp_path)
    nf.run_plan(v1_dir, v1_out)
    nf.run_plan(v1_dir, tmp_path / "s4", "stage4size")
    short = json.loads((v1_out / "noise_floor_plan.json").read_text())
    s4 = json.loads((tmp_path / "s4" / "noise_floor_plan.json").read_text())
    assert s4["chosen_subsets"] == short["chosen_subsets"]
    assert s4["proxy"] == {**short["proxy"], "resolution": 512, "max_steps": 3000}
    assert nf.default_out_name("stage4size") == "proxy_noise_floor_stage4size"


def test_stage4size_matches_the_stage4_config():
    from omegaconf import OmegaConf
    stage4 = OmegaConf.load("configs/ham10000_stage4.yaml")
    proxy = nf.variant_proxy("stage4size")
    assert proxy["resolution"] == stage4.model.resolution
    assert proxy["max_steps"] == stage4.training.max_steps
