"""Stage 5 under a named protocol, and the guarantee that v1's evaluation did not move.

Stage 5 is the only place final_eval_heldout is read for an outcome, and v1's read already happened.
So the first class of tests is the one that matters: every string and every manifest key v1's Stage 5
produced is asserted against the literal the script held BEFORE protocols existed, not against the
protocol object, which would agree with itself.

Nothing here reads final_eval_heldout or runs an evaluation. The fixtures are empty Stage 4 run
directories carrying only the manifest fields the preconditions inspect.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval import ham10000_stage5_evaluate as stage5
from scripts.utils.ham10000_conditions import PROTOCOLS, get_protocol

NAMESPACE = "ham-stratified-v1"
SEEDS = [42, 43, 44]


# --------------------------------------------------------------------------------------------
# fixtures: a Stage 4 tree and a selection manifest, on disk, with nothing real in them
# --------------------------------------------------------------------------------------------
def _write_stage4_tree(results_dir: Path, counts: dict[str, int], manifests: dict[str, str],
                       seeds=SEEDS, skip: set[tuple[str, int]] | None = None) -> None:
    skip = skip or set()
    for condition, count in counts.items():
        for seed in seeds:
            if (condition, seed) in skip:
                continue
            run_dir = results_dir / NAMESPACE / condition / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_manifest.json").write_text(json.dumps({
                "final_eval_heldout_read": False,
                "data": {
                    "real_train_split": "classifier_train",
                    "selection_split": "classifier_val",
                    "real_train_images": 4565,
                    "synthetic_images": count,
                    "synthetic_manifest": manifests[condition],
                },
            }), encoding="utf-8")
            (run_dir / "selection_metrics.json").write_text(json.dumps({"balanced_accuracy": 0.6}),
                                                            encoding="utf-8")


def _write_selection_manifest(outputs_dir: Path, filename: str, payload: dict) -> Path:
    directory = outputs_dir / NAMESPACE
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _configs(tmp_path: Path):
    return (
        SimpleNamespace(paths=SimpleNamespace(outputs_dir=str(tmp_path / "selection"))),
        SimpleNamespace(paths=SimpleNamespace(results_dir=str(tmp_path / "stage4")), seeds=SEEDS),
    )


def _v1_fixture(tmp_path: Path, namespace_in_manifest: str = NAMESPACE):
    selection_cfg, stage4 = _configs(tmp_path)
    _write_stage4_tree(
        Path(stage4.paths.results_dir),
        {"A": 0, "B": 3168, "C": 1100},
        {"A": "None", "B": "/all_candidates.csv", "C": "/asism_selected.csv"},
    )
    _write_selection_manifest(
        Path(selection_cfg.paths.outputs_dir), "asism_selection_manifest.json",
        {"namespace": namespace_in_manifest, "threshold_policy": "learned", "n_selected": 1100},
    )
    return selection_cfg, stage4


def _v2_fixture(tmp_path: Path, namespace_in_manifest: str = NAMESPACE):
    selection_cfg, stage4 = _configs(tmp_path)
    _write_stage4_tree(
        Path(stage4.paths.results_dir),
        {"A": 0, "B": 3168, "C2": 1100, "D2": 1100},
        {"A": "None", "B": "/all_candidates.csv", "C2": "/c2_selected.csv", "D2": "/d2_selected.csv"},
    )
    _write_selection_manifest(
        Path(selection_cfg.paths.outputs_dir), "v2_selection_manifest.json",
        {"namespace": namespace_in_manifest, "n_selected_c2": 1100, "n_selected_d2": 1100},
    )
    return selection_cfg, stage4


# --------------------------------------------------------------------------------------------
class TestV1IsUnchanged:
    """Asserted against the literals scripts/eval/ham10000_stage5_evaluate.py held before this
    change. v1's Stage 5 has already been run; these are what it produced."""

    def test_selection_manifest_location(self):
        spec = get_protocol("v1").selection_manifest
        assert spec.config == "ham10000_stage3.yaml"
        assert spec.section == "ham_stage3"
        assert spec.filename == "asism_selection_manifest.json"
        assert spec.covers == ("C",)

    def test_selection_manifest_is_read_from_stage3(self):
        """The path was Path(stage3.paths.outputs_dir) / namespace / <filename>."""
        from scripts.utils.config import load_named_config

        cfg = load_named_config(*[get_protocol("v1").selection_manifest.config,
                                  get_protocol("v1").selection_manifest.section])
        assert str(cfg.paths.outputs_dir).replace("\\", "/").endswith("outputs/ham10000/stage3")

    def test_output_directory_name(self):
        assert get_protocol("v1").stage5_dirname == "stage5"

    def test_conditions_phrase_is_condition_c(self):
        assert stage5.conditions_phrase(get_protocol("v1").selection_manifest.covers) == "condition C"

    def test_missing_manifest_message_is_byte_identical(self, tmp_path):
        selection_cfg, stage4 = _v1_fixture(tmp_path)
        path = Path(selection_cfg.paths.outputs_dir) / NAMESPACE / "asism_selection_manifest.json"
        path.unlink()
        with pytest.raises(stage5.PreconditionFailed) as excinfo:
            stage5.enforce_preconditions(NAMESPACE, "run-x", selection_cfg, stage4, get_protocol("v1"))
        assert str(excinfo.value) == (
            f"condition C's selection manifest is missing at {path}.\n"
            "Run: python scripts/asism/ham10000_05_adaptive_thresholds.py --namespace " + NAMESPACE
        )

    def test_namespace_mismatch_message_is_byte_identical(self, tmp_path):
        selection_cfg, stage4 = _v1_fixture(tmp_path, namespace_in_manifest="other-ns")
        with pytest.raises(stage5.PreconditionFailed) as excinfo:
            stage5.enforce_preconditions(NAMESPACE, "run-x", selection_cfg, stage4, get_protocol("v1"))
        assert str(excinfo.value) == (
            "the selection manifest was produced for namespace 'other-ns', not "
            f"{NAMESPACE!r}; condition C would be evaluated against another experiment's selection"
        )

    def test_incomplete_grid_message_still_says_a_b_c(self, tmp_path):
        selection_cfg, stage4 = _v1_fixture(tmp_path)
        (Path(stage4.paths.results_dir) / NAMESPACE / "C" / "seed44" / "run_manifest.json").unlink()
        with pytest.raises(stage5.PreconditionFailed) as excinfo:
            stage5.enforce_preconditions(NAMESPACE, "run-x", selection_cfg, stage4, get_protocol("v1"))
        assert "Stage 5 evaluates the complete A/B/C grid or nothing." in str(excinfo.value)
        assert "C/seed44" in str(excinfo.value)

    def test_evidence_keys_are_exactly_the_old_ones(self, tmp_path):
        selection_cfg, stage4 = _v1_fixture(tmp_path)
        evidence = stage5.enforce_preconditions(
            NAMESPACE, "run-x", selection_cfg, stage4, get_protocol("v1")
        )
        evidence.pop("runs")
        assert sorted(evidence) == sorted([
            "final_eval_run_id", "namespace", "shared_protocol", "training_data_per_condition",
            "selection_manifest", "selection_manifest_sha256",
            "selection_threshold_policy", "selection_n_selected",
        ])
        assert evidence["selection_threshold_policy"] == "learned"
        assert evidence["selection_n_selected"] == 1100

    def test_run_still_defaults_to_v1(self):
        """Every Stage 5 command written before protocols existed omits --protocol."""
        import inspect

        assert inspect.signature(stage5.run).parameters["protocol_name"].default == "v1"


# --------------------------------------------------------------------------------------------
class TestV2:
    def test_one_manifest_covers_both_c2_and_d2(self):
        spec = get_protocol("v2").selection_manifest
        assert spec.filename == "v2_selection_manifest.json"
        assert spec.covers == ("C2", "D2")

    def test_conditions_phrase_names_both(self):
        assert stage5.conditions_phrase(get_protocol("v2").selection_manifest.covers) == (
            "conditions C2 and D2"
        )

    def test_selection_config_points_at_stage3_v2(self):
        from scripts.utils.config import load_named_config

        spec = get_protocol("v2").selection_manifest
        cfg = load_named_config(spec.config, spec.section)
        assert str(cfg.paths.outputs_dir).replace("\\", "/").endswith("outputs/ham10000/stage3_v2")

    def test_preconditions_pass_and_record_both_counts(self, tmp_path):
        selection_cfg, stage4 = _v2_fixture(tmp_path)
        evidence = stage5.enforce_preconditions(
            NAMESPACE, "run-v2", selection_cfg, stage4, get_protocol("v2")
        )
        assert evidence["selection_n_selected_c2"] == 1100
        assert evidence["selection_n_selected_d2"] == 1100
        assert set(evidence["training_data_per_condition"]) == {"A", "B", "C2", "D2"}

    def test_missing_manifest_names_both_conditions_and_the_v2_command(self, tmp_path):
        selection_cfg, stage4 = _v2_fixture(tmp_path)
        (Path(selection_cfg.paths.outputs_dir) / NAMESPACE / "v2_selection_manifest.json").unlink()
        with pytest.raises(stage5.PreconditionFailed) as excinfo:
            stage5.enforce_preconditions(NAMESPACE, "run-v2", selection_cfg, stage4, get_protocol("v2"))
        message = str(excinfo.value)
        assert message.startswith("conditions C2 and D2's selection manifest is missing at ")
        assert "scripts/followup/ham10000_v2_select.py --namespace " + NAMESPACE in message

    def test_incomplete_grid_names_the_v2_conditions(self, tmp_path):
        selection_cfg, stage4 = _v2_fixture(tmp_path)
        (Path(stage4.paths.results_dir) / NAMESPACE / "D2" / "seed42" / "run_manifest.json").unlink()
        with pytest.raises(stage5.PreconditionFailed) as excinfo:
            stage5.enforce_preconditions(NAMESPACE, "run-v2", selection_cfg, stage4, get_protocol("v2"))
        assert "complete A/B/C2/D2 grid" in str(excinfo.value)

    def test_size_mismatch_between_c2_and_d2_is_refused_here_too(self, tmp_path):
        """The aggregator's control check has to hold at the protected evaluation as well: a D2 of a
        different size would make the confirmatory comparison a size comparison."""
        selection_cfg, stage4 = _configs(tmp_path)
        _write_stage4_tree(
            Path(stage4.paths.results_dir),
            {"A": 0, "B": 3168, "C2": 1100, "D2": 900},
            {"A": "None", "B": "/all.csv", "C2": "/c2.csv", "D2": "/d2.csv"},
        )
        _write_selection_manifest(
            Path(selection_cfg.paths.outputs_dir), "v2_selection_manifest.json",
            {"namespace": NAMESPACE, "n_selected_c2": 1100, "n_selected_d2": 900},
        )
        from scripts.classify.ham10000_aggregate_conditions import AggregationError

        with pytest.raises(AggregationError, match="size-matched control"):
            stage5.enforce_preconditions(NAMESPACE, "run-v2", selection_cfg, stage4, get_protocol("v2"))


# --------------------------------------------------------------------------------------------
class TestV2CannotOverwriteV1:
    def test_output_directory_names_differ(self):
        assert get_protocol("v1").stage5_dirname != get_protocol("v2").stage5_dirname

    def test_the_separation_is_load_bearing(self):
        """Stage 5 builds its output directory as results_dir.parent / stage5_dirname. Both
        protocols share that parent, so the directory NAME is the only thing keeping a v2 run under
        a colliding run id off the protected v1 result."""
        from scripts.utils.config import load_named_config

        v1 = load_named_config(get_protocol("v1").stage4_config, "ham_stage4")
        v2 = load_named_config(get_protocol("v2").stage4_config, "ham_stage4")
        assert Path(str(v1.paths.results_dir)).parent == Path(str(v2.paths.results_dir)).parent

    @pytest.mark.parametrize("name", sorted(PROTOCOLS))
    def test_every_covered_condition_exists_and_carries_synthetic_images(self, name):
        protocol = get_protocol(name)
        for condition in protocol.selection_manifest.covers:
            assert condition in protocol.conditions
            assert condition in protocol.synthetic_conditions

    @pytest.mark.parametrize("name", sorted(PROTOCOLS))
    def test_rebuild_command_is_substitutable(self, name):
        command = get_protocol(name).selection_manifest.rebuild_command
        assert "{namespace}" in command
        assert NAMESPACE in command.format(namespace=NAMESPACE)

    @pytest.mark.parametrize("name", sorted(PROTOCOLS))
    def test_stage5_dirname_is_not_a_path(self, name):
        dirname = get_protocol(name).stage5_dirname
        assert dirname and "/" not in dirname and "\\" not in dirname
