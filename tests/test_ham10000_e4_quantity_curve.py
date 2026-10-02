"""E4 quantity curve: the frozen design, the section 4 rule and the phase guards (no GPU, no images)."""

import json

import numpy as np
import pytest
from scipy import stats

from scripts.followup import ham10000_e4_quantity_curve as e4

LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
POOL = [f"syn_{i:05d}" for i in range(2100)]
SIZES = e4.sizes_for_pool(len(POOL))


def _safe_report(**overrides):
    applied = {"reject_invalid_iqa": True, "reject_near_duplicates": True, **overrides}
    return {"safety_checks_applied": applied, "safety_gate_removals": {},
            "candidates_before_safety_gate": len(POOL), "per_class_counts": {"df": len(POOL)}}


def _values(means, sd=0.004, seed=0):
    rng = np.random.default_rng(seed)
    return {size: mean + rng.normal(0, sd, e4.RUNS_PER_SIZE) for size, mean in zip(SIZES, means)}


# ---- the design ------------------------------------------------------------------------------------


def test_sizes_are_the_approved_six_and_need_a_pool_above_2000():
    assert SIZES == (0, 250, 500, 1000, 2000, 2100)
    with pytest.raises(e4.E4Error):
        e4.sizes_for_pool(2000)


def test_chains_are_frozen_permutations_independent_of_input_order():
    chains = e4.build_chains(POOL)
    assert chains == e4.build_chains(list(reversed(POOL)))
    assert set(chains) == {"1", "2"}
    for chain in chains.values():
        assert sorted(chain) == sorted(POOL)
    assert chains["1"] != chains["2"]
    with pytest.raises(e4.E4Error):
        e4.build_chains(POOL + POOL[:1])


def test_subsets_are_nested_within_a_chain():
    chains = e4.build_chains(POOL)
    previous = []
    for size in SIZES[1:]:
        ids = e4.subset_ids({"size": size, "chain": 1}, chains)
        assert len(ids) == size and ids[: len(previous)] == previous
        previous = ids
    assert e4.subset_ids({"size": 0, "chain": None}, chains) == []


def test_cells_are_sixty_with_ten_runs_per_size():
    cells = e4.plan_cells(SIZES)
    assert len(cells) == 60
    assert len({c["cell_id"] for c in cells}) == 60
    base = [c for c in cells if c["size"] == 0]
    assert sorted(c["seed"] for c in base) == list(range(42, 52))
    for size in SIZES[1:]:
        at = [c for c in cells if c["size"] == size]
        assert len(at) == 10
        assert sorted((c["chain"], c["seed"]) for c in at) == [(ch, s) for ch in (1, 2) for s in range(42, 47)]


def test_job_order_is_seed_major():
    order = e4.job_order(e4.plan_cells(SIZES))
    seeds = [c["seed"] for c in order]
    assert seeds == sorted(seeds)


def test_stage4_recipe_check_passes_on_the_real_config_and_refuses_any_difference():
    from omegaconf import OmegaConf

    from scripts.utils.config import load_named_config

    stage4 = load_named_config("ham10000_stage4.yaml", "ham_stage4")
    proxy = e4.variant_proxy(e4.PROXY_VARIANT)
    e4.check_stage4_recipe(stage4, proxy)
    with pytest.raises(e4.E4Error):
        e4.check_stage4_recipe(stage4, {**proxy, "max_steps": 1500})
    weighted = OmegaConf.merge(stage4, {"training": {"class_weighting": "inverse_frequency"}})
    with pytest.raises(e4.E4Error):
        e4.check_stage4_recipe(weighted, proxy)


# ---- section 4 -------------------------------------------------------------------------------------


def test_welch_matches_scipy():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0.90, 0.01, 10), rng.normal(0.91, 0.02, 10)
    result = e4.welch(a, b)
    reference = stats.ttest_ind(b, a, equal_var=False, alternative="greater")
    assert np.isclose(result["p_one_sided_greater"], reference.pvalue)
    low, high = result["ci95_two_sided"]
    assert low < result["difference"] < high


def test_rising_curve_is_go():
    values = _values([0.900, 0.910, 0.925, 0.940, 0.955, 0.970])
    result = e4.decide(values, SIZES)
    assert result["overall"]["passes"]
    assert result["resolved_steps"]
    assert result["verdict"] == e4.VERDICT_GO


def test_a_jump_at_the_first_step_only_is_coarse():
    # Step 1 is not tested: a curve that rises only from 0 to 250 cannot support a stopping rule.
    values = _values([0.900, 0.940, 0.940, 0.940, 0.940, 0.940])
    result = e4.decide(values, SIZES)
    assert result["overall"]["passes"]
    assert result["resolved_steps"] == []
    assert result["steps"]["1"]["tested"] is False
    assert result["verdict"] == e4.VERDICT_COARSE


def test_a_flat_curve_is_no_whatever_the_steps_show():
    values = _values([0.920, 0.920, 0.920, 0.920, 0.920, 0.920])
    assert e4.decide(values, SIZES)["verdict"] == e4.VERDICT_NO


def test_overall_failure_overrides_a_resolved_step():
    # A rise from 250 to 500 cannot make GO when U(N) - U(0) is not above zero.
    values = _values([0.940, 0.900, 0.940, 0.940, 0.940, 0.940], sd=0.002)
    result = e4.decide(values, SIZES)
    assert not result["overall"]["passes"]
    assert 2 in result["resolved_steps"]
    assert result["verdict"] == e4.VERDICT_NO


def test_holm_is_applied_over_the_four_tested_steps():
    # One step whose raw one-sided p is below 0.05 but above 0.05/4 is NOT resolved.
    base = np.full(10, 0.900) + np.linspace(-0.01, 0.01, 10)
    values = {size: base + 0.05 for size in SIZES}
    values[0] = base
    values[250] = base + 0.05
    shift = 0.0
    for candidate in np.linspace(0.0, 0.02, 2001):
        p = e4.welch(values[250], base + 0.05 + candidate)["p_one_sided_greater"]
        if 0.0125 < p < 0.05:
            shift = candidate
            break
    assert shift > 0
    for size in (500, 1000, 2000, 2100):
        values[size] = base + 0.05 + shift
    result = e4.decide(values, SIZES)
    step2 = result["steps"]["2"]
    assert 0.0125 < step2["p_one_sided_greater"] < 0.05
    assert step2["holm_adjusted_p"] > 0.05 and not step2["resolved"]
    assert result["verdict"] == e4.VERDICT_COARSE


def test_decide_refuses_values_for_a_different_set_of_sizes():
    values = _values([0.9] * 6)
    values.pop(2100)
    with pytest.raises(e4.E4Error):
        e4.decide(values, SIZES)


# ---- grid integrity -------------------------------------------------------------------------------


def _rows(cells, value=0.9):
    return [{"cell_id": c["cell_id"], "metrics": {m: value for m in e4.METRICS}} for c in cells]


def test_values_by_size_refuses_missing_duplicated_and_unknown_cells():
    cells = e4.plan_cells(SIZES)
    rows = _rows(cells)
    assert {k: len(v) for k, v in e4.values_by_size(rows, cells, e4.PRIMARY_METRIC).items()} == {s: 10 for s in SIZES}
    with pytest.raises(e4.E4Error, match="incomplete"):
        e4.values_by_size(rows[:-1], cells, e4.PRIMARY_METRIC)
    with pytest.raises(e4.E4Error, match="duplicated"):
        e4.values_by_size(rows + rows[:1], cells, e4.PRIMARY_METRIC)
    with pytest.raises(e4.E4Error, match="does not define"):
        e4.values_by_size(rows + [{"cell_id": "s9999_c3_seed1", "metrics": {}}], cells, e4.PRIMARY_METRIC)


# ---- phases, end to end on the CPU ---------------------------------------------------------------


def _fake_inputs():
    real = [{"image_id": f"real_{i}", "dx": "nv"} for i in range(40)]
    tuning = [{"image_id": f"tune_{i}", "dx": "nv"} for i in range(20)]
    candidates = {i: {"image_id": i, "dx": "df"} for i in POOL}
    return {"real": real, "tuning": tuning, "candidates": candidates}


def _fake_measure(train, tuning, proxy, seed, device):
    n_synthetic = sum(1 for r in train if str(r["image_id"]).startswith("syn_"))
    assert proxy.resolution == 512 and proxy.max_steps == 3000
    rng = np.random.default_rng(seed * 10007 + n_synthetic)
    auroc = 0.90 + 0.012 * np.log2(1 + n_synthetic / 250) + rng.normal(0, 0.003)
    return {**{m: float(auroc) for m in e4.METRICS},
            "per_class": {label: {"recall": 0.5} for label in LABELS}}


def test_plan_measure_analyze_end_to_end(tmp_path):
    out = tmp_path / "e4"
    plan = e4.run_plan("ns", out, pool_loader=lambda ns: (POOL, _safe_report()))
    assert plan["runs_planned"] == 60 and plan["safe_pool_size"] == len(POOL)
    with pytest.raises(e4.E4Error, match="never re-drawn"):
        e4.run_plan("ns", out, pool_loader=lambda ns: (POOL, _safe_report()))

    e4.run_measure("ns", out, "cpu", measure_fn=_fake_measure, inputs=_fake_inputs())
    rows = [json.loads(line) for line in (out / "e4_runs.jsonl").read_text().splitlines()]
    assert len(rows) == 60
    assert {r["size"]: r["n_synthetic"] for r in rows} == {s: s for s in SIZES}
    assert all(r["effective_dataset_size"] == 40 + r["n_synthetic"] for r in rows)

    # Resuming after completion trains nothing.
    e4.run_measure("ns", out, "cpu", measure_fn=lambda *a: pytest.fail("re-ran a done cell"),
                   inputs=_fake_inputs())

    result = e4.run_analyze(out)
    report = json.loads((out / "e4_report.json").read_text())
    assert result["verdict"] == report["verdict"] == e4.VERDICT_GO
    assert report["primary_metric"] == "macro_auroc_ovr"
    assert set(report["reported_not_deciding"]["per_chain_curve"]) == {"1", "2"}
    assert report["real_only_baseline"]["n"] == 10


def test_measure_resumes_only_the_missing_cells(tmp_path):
    out = tmp_path / "e4"
    e4.run_plan("ns", out, pool_loader=lambda ns: (POOL, _safe_report()))
    calls = []

    def counting(*args):
        calls.append(args[3])
        if len(calls) == 7:
            raise RuntimeError("pod lost")
        return _fake_measure(*args)

    with pytest.raises(RuntimeError):
        e4.run_measure("ns", out, "cpu", measure_fn=counting, inputs=_fake_inputs())
    assert len((out / "e4_runs.jsonl").read_text().splitlines()) == 6
    with pytest.raises(e4.E4Error, match="incomplete"):
        e4.run_analyze(out)
    e4.run_measure("ns", out, "cpu", measure_fn=_fake_measure, inputs=_fake_inputs())
    assert len((out / "e4_runs.jsonl").read_text().splitlines()) == 60


def test_analyze_refuses_a_run_on_a_different_subset(tmp_path):
    out = tmp_path / "e4"
    e4.run_plan("ns", out, pool_loader=lambda ns: (POOL, _safe_report()))
    e4.run_measure("ns", out, "cpu", measure_fn=_fake_measure, inputs=_fake_inputs())
    path = out / "e4_runs.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["subset_sha256"] = "0" * 64
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(e4.E4Error, match="different subset"):
        e4.run_analyze(out)


def test_a_tampered_plan_is_refused(tmp_path):
    out = tmp_path / "e4"
    e4.run_plan("ns", out, pool_loader=lambda ns: (POOL, _safe_report()))
    path = out / "e4_plan.json"
    plan = json.loads(path.read_text())
    # The hash covers membership, so swap an image inside every subset with one outside the 2000.
    chain = plan["chains"]["1"]
    chain[0], chain[-1] = chain[-1], chain[0]
    path.write_text(json.dumps(plan))
    with pytest.raises(e4.E4Error, match="frozen subset hash"):
        e4.run_measure("ns", out, "cpu", measure_fn=_fake_measure, inputs=_fake_inputs())


def test_plan_refuses_a_pool_built_without_both_safety_checks(tmp_path):
    with pytest.raises(e4.E4Error, match="safety"):
        e4.run_plan("ns", tmp_path / "e4", pool_loader=lambda ns: (POOL, _safe_report(reject_near_duplicates=False)))


def test_measure_refuses_a_subset_image_that_is_not_a_candidate(tmp_path):
    out = tmp_path / "e4"
    e4.run_plan("ns", out, pool_loader=lambda ns: (POOL, _safe_report()))
    inputs = _fake_inputs()
    inputs["candidates"].pop(e4._load_plan(out)["chains"]["1"][0])
    with pytest.raises(e4.E4Error, match="not candidates"):
        e4.run_measure("ns", out, "cpu", measure_fn=_fake_measure, inputs=inputs)
