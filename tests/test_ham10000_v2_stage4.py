"""Stage 4 v2: the A/B/C2/D2 protocol, and the guarantee that v1 did not move.

The first class of tests is the important one. Making the condition set configurable touched the
three scripts the v1 primary result was produced by, so every v1 constant is asserted against the
literal values those scripts hardcoded before this change — not against the protocol object, which
would happily agree with itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.classify import ham10000_aggregate_conditions as aggregate
from scripts.eval import ham10000_compare_conditions as compare
from scripts.utils.ham10000_conditions import PROTOCOLS, get_protocol


class TestV1IsUnchanged:
    """Written against the literals the scripts held before protocols existed."""

    def test_condition_tuple(self):
        assert get_protocol("v1").conditions == ("A", "B", "C")
        assert aggregate.CONDITIONS == ("A", "B", "C")
        assert compare.CONDITIONS == ("A", "B", "C")

    def test_comparison_families(self):
        assert compare.CONFIRMATORY_COMPARISONS == (("C", "B"),)
        assert compare.EXPLORATORY_COMPARISONS == (("C", "A"), ("B", "A"))

    def test_primary_metric(self):
        assert compare.PRIMARY_METRIC == "balanced_accuracy"

    def test_config_file(self):
        assert get_protocol("v1").stage4_config == "ham10000_stage4.yaml"

    def test_default_protocol_is_v1(self):
        """Every existing v1 command omits --protocol, so the default has to stay v1."""
        assert get_protocol(None).name == "v1"

    def test_interpretation_text_is_byte_identical(self):
        """This string is written into v1's stage5_comparison.json; regenerating that report after
        the protocol refactor must not change a character of it."""
        assert get_protocol("v1").interpretation == (
            "C vs B is the confirmatory test of the contribution: same generator, same "
            "candidates, selection the only difference. C vs A shows only that synthetic data "
            "helps. Every p-value is reported with its effect size and interval; significance "
            "alone is not a result."
        )

    def test_v1_baseline_delta_key_is_unchanged(self):
        """v1 reports wrote per_class_recall_delta_vs_A; the key is built from the baseline now."""
        assert f"per_class_recall_delta_vs_{get_protocol('v1').baseline}" == "per_class_recall_delta_vs_A"


class TestV2Protocol:
    def test_conditions(self):
        assert get_protocol("v2").conditions == ("A", "B", "C2", "D2")

    def test_confirmatory_is_c2_vs_d2(self):
        """Decided before any v2 run: the only comparison that isolates the selection rule."""
        assert get_protocol("v2").confirmatory == (("C2", "D2"),)

    def test_c2_vs_d2_is_not_also_exploratory(self):
        protocol = get_protocol("v2")
        assert ("C2", "D2") not in protocol.exploratory
        assert ("D2", "C2") not in protocol.exploratory

    def test_separate_config_file(self):
        assert get_protocol("v2").stage4_config == "ham10000_v2_stage4.yaml"
        assert get_protocol("v2").stage4_config != get_protocol("v1").stage4_config

    def test_unknown_protocol_refused(self):
        with pytest.raises(KeyError):
            get_protocol("v3")


class TestProtocolsAreWellFormed:
    @pytest.mark.parametrize("name", sorted(PROTOCOLS))
    def test_every_compared_condition_exists(self, name):
        protocol = get_protocol(name)
        for left, right in protocol.confirmatory + protocol.exploratory:
            assert left in protocol.conditions
            assert right in protocol.conditions

    @pytest.mark.parametrize("name", sorted(PROTOCOLS))
    def test_baseline_carries_no_synthetic_images(self, name):
        protocol = get_protocol(name)
        assert protocol.baseline in protocol.conditions
        assert protocol.baseline not in protocol.synthetic_conditions

    @pytest.mark.parametrize("name", sorted(PROTOCOLS))
    def test_families_do_not_overlap(self, name):
        """A pair in both families would be tested twice under two different corrections."""
        protocol = get_protocol(name)
        assert not set(protocol.confirmatory) & set(protocol.exploratory)

    @pytest.mark.parametrize("name", sorted(PROTOCOLS))
    def test_required_and_suspect_equal_counts_do_not_contradict(self, name):
        protocol = get_protocol(name)
        suspect = {frozenset(pair) for pair in protocol.equal_counts_suspect}
        required = {frozenset(pair) for pair in protocol.equal_counts_required}
        assert not suspect & required


def _runs(counts: dict[str, int], manifests: dict[str, str] | None = None) -> dict:
    """One seed per condition, carrying only the manifest fields the data check reads."""
    manifests = manifests or {name: f"/manifests/{name}.csv" for name in counts}
    return {
        (condition, 42): {
            "manifest": {
                "data": {
                    "synthetic_images": count,
                    "synthetic_manifest": manifests[condition],
                    "real_train_images": 4565,
                }
            }
        }
        for condition, count in counts.items()
    }


class TestConditionDataChecks:
    def test_v2_accepts_c2_and_d2_at_equal_counts(self):
        """The whole point of D2: same size as C2, different draw. This must not be flagged."""
        runs = _runs({"A": 0, "B": 3168, "C2": 1100, "D2": 1100})
        result = aggregate.check_training_data_differs(runs, get_protocol("v2"))
        assert result["C2"]["synthetic_images"] == [1100]
        assert result["D2"]["synthetic_images"] == [1100]

    def test_v2_refuses_c2_and_d2_at_different_counts(self):
        runs = _runs({"A": 0, "B": 3168, "C2": 1100, "D2": 900})
        with pytest.raises(aggregate.AggregationError, match="size-matched control"):
            aggregate.check_training_data_differs(runs, get_protocol("v2"))

    def test_v2_refuses_c2_and_d2_sharing_a_manifest(self):
        runs = _runs(
            {"A": 0, "B": 3168, "C2": 1100, "D2": 1100},
            {"A": "None", "B": "/b.csv", "C2": "/c2.csv", "D2": "/c2.csv"},
        )
        with pytest.raises(aggregate.AggregationError, match="same manifest"):
            aggregate.check_training_data_differs(runs, get_protocol("v2"))

    def test_v2_refuses_b_matching_c2(self):
        """B == C2 still means the selector kept everything, exactly as in v1."""
        runs = _runs({"A": 0, "B": 3168, "C2": 3168, "D2": 3168})
        with pytest.raises(aggregate.AggregationError, match="same number"):
            aggregate.check_training_data_differs(runs, get_protocol("v2"))

    def test_v1_still_refuses_b_matching_c(self):
        runs = _runs({"A": 0, "B": 3168, "C": 3168})
        with pytest.raises(aggregate.AggregationError, match="same number"):
            aggregate.check_training_data_differs(runs, get_protocol("v1"))

    def test_v1_still_refuses_a_with_synthetic_images(self):
        runs = _runs({"A": 5, "B": 3168, "C": 1100})
        with pytest.raises(aggregate.AggregationError, match="real-only baseline"):
            aggregate.check_training_data_differs(runs, get_protocol("v1"))

    def test_v1_default_matches_explicit_v1(self):
        runs = _runs({"A": 0, "B": 3168, "C": 1100})
        assert aggregate.check_training_data_differs(runs) == aggregate.check_training_data_differs(
            runs, get_protocol("v1")
        )

    @pytest.mark.parametrize("condition", ["B", "C2", "D2"])
    def test_v2_refuses_an_empty_synthetic_condition(self, condition):
        counts = {"A": 0, "B": 3168, "C2": 1100, "D2": 1100}
        counts[condition] = 0
        with pytest.raises(aggregate.AggregationError, match="no synthetic images"):
            aggregate.check_training_data_differs(counts and _runs(counts), get_protocol("v2"))


class TestV2Config:
    def test_conditions_and_results_dir(self):
        import os

        from scripts.utils.config import load_named_config

        os.environ.setdefault("PROJECT_ROOT", ".")
        config = load_named_config("ham10000_v2_stage4.yaml", "ham_stage4")
        assert list(config.conditions) == ["A", "B", "C2", "D2"]
        assert config.conditions.A.synthetic_manifest is None
        assert "stage3_v2" in str(config.conditions.C2.synthetic_manifest)
        assert "d2_selected.csv" in str(config.conditions.D2.synthetic_manifest)

    def test_writes_somewhere_v1_cannot_be_overwritten(self):
        from scripts.utils.config import load_named_config

        v1 = load_named_config("ham10000_stage4.yaml", "ham_stage4")
        v2 = load_named_config("ham10000_v2_stage4.yaml", "ham_stage4")
        assert str(v1.paths.results_dir) != str(v2.paths.results_dir)

    def test_training_budget_is_identical_to_v1(self):
        """Only the DATA may differ between conditions, in v2 as in v1."""
        from scripts.utils.config import load_named_config

        v1 = load_named_config("ham10000_stage4.yaml", "ham_stage4")
        v2 = load_named_config("ham10000_v2_stage4.yaml", "ham_stage4")
        for key in ("max_steps", "batch_size", "learning_rate", "weight_decay", "class_weighting"):
            assert v1.training[key] == v2.training[key], key
        assert dict(v1.model) == dict(v2.model)
        assert list(v1.seeds) == list(v2.seeds)
        assert str(v1.real_train_split) == str(v2.real_train_split)
        assert str(v1.selection_split) == str(v2.selection_split)
