"""Producer/consumer and refusal contracts added during the code-completeness pass."""
from __future__ import annotations

import importlib.util
import json
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.asism.signals import aggregate_pathology_overlaps
from scripts.utils.artifact_contracts import (
    ArtifactContractError, require_manifest_fields, require_score_artifact, stage2_paths,
    stage3_paths, validate_namespace_run_id,
)
from scripts.utils.manifest import sha256_file
from scripts.utils.splits import FinalEvalAccessViolation, load_split, validate_final_eval_context
from fixture_workspace import fixture_workspace


def load_script(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_versioned_namespace_contract_refuses_mutable_aliases():
    for value, kind in (("dev", "dev"), ("production", "production"), ("dev-v1", "production")):
        try:
            validate_namespace_run_id(value, kind)
        except ArtifactContractError:
            pass
        else:
            raise AssertionError(f"accepted mutable/mismatched namespace {value}")
    validate_namespace_run_id("dev-smoke-v1", "dev")
    validate_namespace_run_id("production-thesis-v1", "production")


def test_stage2_and_stage3_paths_are_namespace_disjoint():
    cfg = SimpleNamespace(paths=SimpleNamespace(synthetic_root="synthetic"))
    assert stage2_paths(cfg, "dev-v1")["manifest_path"] != stage2_paths(cfg, "production-v1")["manifest_path"]
    assert stage3_paths(cfg, "dev-v1")["scores_dir"] != stage3_paths(cfg, "production-v1")["scores_dir"]


def test_manifest_contract_rejects_stale_field():
    with fixture_workspace("stale-manifest") as workspace:
        path = workspace / "manifest.json"
        path.write_text(json.dumps({"split_manifest_hash": "old"}), encoding="utf-8")
        try:
            require_manifest_fields(path, {"split_manifest_hash": "new"}, "fixture")
        except ArtifactContractError as exc:
            assert "Stale/incompatible" in str(exc)
        else:
            raise AssertionError("stale manifest was accepted")


def test_score_sidecar_hash_is_enforced():
    with fixture_workspace("score-refusal") as workspace:
        path = workspace / "similarity_scores.parquet"
        path.write_bytes(b"not-a-valid-completed-score")
        try:
            require_score_artifact(path, "similarity", {"split_namespace": "production-v1"})
        except ArtifactContractError:
            pass
        else:
            raise AssertionError("score without compatible sidecar was accepted")


def test_stale_unnamespaced_stage2_recipes_are_explicitly_rejected():
    module = load_script("scripts/generate/02_generate_synthetic_images.py", "stage2_generation_test")
    with fixture_workspace("legacy-recipes") as root:
        (root / "label_recipes.csv").write_text("recipe_id\nlegacy\n", encoding="utf-8")
        cfg = SimpleNamespace(paths=SimpleNamespace(synthetic_root=str(root)))
        try:
            module.validate_generation_inputs(cfg, "dev-v1", stage2_paths(cfg, "dev-v1"))
        except ArtifactContractError as exc:
            assert "stale unnamespaced" in str(exc)
        else:
            raise AssertionError("legacy recipes were accepted")


def test_final_eval_metadata_purpose_never_returns_rows():
    try:
        load_split("final_eval_heldout", "production-v1", purpose="schema_validation", caller="test")
    except FinalEvalAccessViolation as exc:
        assert "split_metadata" in str(exc)
    else:
        raise AssertionError("metadata purpose exposed final rows")


def test_arbitrary_final_eval_context_is_refused():
    with fixture_workspace("bad-final-context") as workspace:
        path = workspace / "context.json"
        path.write_text(json.dumps({"final_eval_run_id": "anything", "namespace": "production-v1", "status": "registered"}), encoding="utf-8")
        try:
            validate_final_eval_context(path, "anything", "production-v1")
        except FinalEvalAccessViolation:
            pass
        else:
            raise AssertionError("arbitrary run ID unlocked final data")


def test_full_sha256_is_not_truncated():
    assert len(sha256_file(REPO / "scripts/utils/classifier.py")) == 64


def test_real_data_smoke_stage5_marker_is_bound_to_dev_namespace_and_split():
    module = load_script("scripts/eval/stage5_evaluate.py", "stage5_real_smoke_authorization_test")
    manifest = {"namespace_class": "dev", "manifest_hash": "split-hash"}
    with fixture_workspace("real-smoke-stage5-marker") as workspace:
        marker = {
            "schema_version": 1,
            "purpose": "real-data-engineering-smoke-only",
            "namespace": "dev-real-smoke-v2",
            "split_manifest_hash": "split-hash",
            "not_scientific_evidence": True,
            "upstream_code_identity_sha256": "a" * 64,
        }
        (workspace / "REAL_DATA_SMOKE_ONLY.json").write_text(json.dumps(marker), encoding="utf-8")
        authorization = module.require_smoke_final_eval_authorization(
            "dev-real-smoke-v2", manifest, workspace
        )
        assert authorization == {
            "kind": "real_data", "upstream_code_identity_sha256": "a" * 64
        }

        marker["split_manifest_hash"] = "different-split"
        (workspace / "REAL_DATA_SMOKE_ONLY.json").write_text(json.dumps(marker), encoding="utf-8")
        try:
            module.require_smoke_final_eval_authorization("dev-real-smoke-v2", manifest, workspace)
        except SystemExit as exc:
            assert "marker mismatch" in str(exc)
        else:
            raise AssertionError("real-data smoke marker was accepted for the wrong split")

        try:
            module.require_smoke_final_eval_authorization(
                "production-thesis-v1",
                {"namespace_class": "production", "manifest_hash": "split-hash"},
                workspace,
            )
        except SystemExit as exc:
            assert "dev-class" in str(exc)
        else:
            raise AssertionError("real-data smoke marker unlocked a production split")


def test_multilabel_gradcam_aggregation_uses_every_positive():
    mean, minimum = aggregate_pathology_overlaps({"Edema": 0.9, "Pneumonia": 0.1})
    assert mean == 0.5 and minimum == 0.1


def test_asism_grid_contains_equal_and_every_individual_signal_baseline():
    module = load_script("scripts/asism/03_tune_freeze_select.py", "asism_tuning_test")
    signals = ["similarity", "iqa", "explainability", "agreement"]
    grid = module.candidate_weight_grid([*signals, "uncertainty"], 12, 42)
    assert any(len({row[name] for name in signals}) == 1 and row[signals[0]] > 0 for row in grid)
    for signal in signals:
        assert any(row[signal] == 1.0 and sum(row.values()) == 1.0 for row in grid), signal
    assert any(row.get("__uncertainty_band_only__") == 1.0 for row in grid)


def test_split_builder_initializes_out_dir_before_use():
    source = (REPO / "scripts/data/02b_build_sixway_splits.py").read_text(encoding="utf-8")
    assignment = source.index("out_dir = resolve_splits_dir")
    first_use = source.index("if out_dir.exists()")
    terminal = source.index("raise SystemExit(main())")
    assert assignment < first_use < terminal


if __name__ == "__main__":
    import traceback
    tests = [(name, value) for name, value in sorted(globals().items()) if name.startswith("test_")]
    passed = failed = 0
    for name, function in tests:
        try:
            function(); print(f"  PASS  {name}", flush=True); passed += 1
        except Exception:
            print(f"  FAIL  {name}", flush=True); traceback.print_exc(); failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    raise SystemExit(1 if failed else 0)
