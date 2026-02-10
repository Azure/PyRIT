# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings

import pytest

from pyrit.analytics.result_analysis import (
    AnalysisResult,
    AttackStats,
    analyze_results,
)
from pyrit.identifiers import ConverterIdentifier
from pyrit.models import AttackOutcome, AttackResult, MessagePiece


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_attack(
    outcome: AttackOutcome,
    attack_type: str | None = "PromptSendingAttack",
    conversation_id: str = "conv-1",
) -> AttackResult:
    """Minimal valid AttackResult for analytics tests."""
    attack_identifier: dict[str, str] = {}
    if attack_type is not None:
        attack_identifier["__type__"] = attack_type
        attack_identifier["__module__"] = "pyrit.executor.attack"
        attack_identifier["id"] = "00000000-0000-0000-0000-000000000001"

    return AttackResult(
        conversation_id=conversation_id,
        objective="test objective",
        attack_identifier=attack_identifier,
        outcome=outcome,
    )


def make_converter(
    class_name: str,
    class_module: str = "pyrit.prompt_converter.test_converter",
) -> ConverterIdentifier:
    """Create a test ConverterIdentifier with minimal required fields."""
    return ConverterIdentifier(
        class_name=class_name,
        class_module=class_module,
        class_description="Test converter",
        identifier_type="instance",
        supported_input_types=("text",),
        supported_output_types=("text",),
    )


def make_attack_with_converters(
    outcome: AttackOutcome,
    converter_names: list[str],
    attack_type: str = "test",
    conversation_id: str = "conv-1",
) -> AttackResult:
    """Create an AttackResult with converter identifiers on last_response."""
    converters = [make_converter(name) for name in converter_names]
    message = MessagePiece(
        role="user",
        original_value="test",
        converter_identifiers=converters,
    )
    attack_identifier: dict[str, str] = {
        "__type__": attack_type,
        "__module__": "pyrit.executor.attack",
        "id": "00000000-0000-0000-0000-000000000001",
    }
    return AttackResult(
        conversation_id=conversation_id,
        objective="test",
        attack_identifier=attack_identifier,
        outcome=outcome,
        last_response=message,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class TestAnalyzeResultsValidation:
    """Input validation for analyze_results."""

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            analyze_results([])

    def test_invalid_object_raises(self):
        with pytest.raises(TypeError, match="Expected AttackResult"):
            analyze_results(["not-an-AttackResult"])

    def test_unknown_dimension_raises(self):
        attacks = [make_attack(AttackOutcome.SUCCESS)]
        with pytest.raises(ValueError, match="Unknown dimension 'nonexistent'"):
            analyze_results(attacks, group_by=["nonexistent"])

    def test_unknown_dimension_in_composite_raises(self):
        attacks = [make_attack(AttackOutcome.SUCCESS)]
        with pytest.raises(ValueError, match="Unknown dimension 'bad_dim'"):
            analyze_results(attacks, group_by=[("attack_type", "bad_dim")])


# ---------------------------------------------------------------------------
# Overall stats
# ---------------------------------------------------------------------------
class TestOverallStats:
    """Overall stats computation (no dimension breakdown)."""

    @pytest.mark.parametrize(
        "outcomes, expected_successes, expected_failures, expected_undetermined, expected_rate",
        [
            ([AttackOutcome.SUCCESS, AttackOutcome.SUCCESS], 2, 0, 0, 1.0),
            ([AttackOutcome.FAILURE, AttackOutcome.FAILURE], 0, 2, 0, 0.0),
            ([AttackOutcome.SUCCESS, AttackOutcome.FAILURE], 1, 1, 0, 0.5),
            ([AttackOutcome.SUCCESS, AttackOutcome.UNDETERMINED], 1, 0, 1, 1.0),
            ([AttackOutcome.FAILURE, AttackOutcome.UNDETERMINED], 0, 1, 1, 0.0),
            (
                [AttackOutcome.SUCCESS, AttackOutcome.FAILURE, AttackOutcome.UNDETERMINED],
                1,
                1,
                1,
                0.5,
            ),
        ],
    )
    def test_overall_stats(self, outcomes, expected_successes, expected_failures, expected_undetermined, expected_rate):
        attacks = [make_attack(o) for o in outcomes]
        result = analyze_results(attacks, group_by=[])

        assert isinstance(result, AnalysisResult)
        overall = result.overall
        assert overall.successes == expected_successes
        assert overall.failures == expected_failures
        assert overall.undetermined == expected_undetermined
        assert overall.total_decided == expected_successes + expected_failures
        assert overall.success_rate == expected_rate

    def test_all_undetermined_gives_none_rate(self):
        attacks = [make_attack(AttackOutcome.UNDETERMINED)]
        result = analyze_results(attacks, group_by=[])
        assert result.overall.success_rate is None
        assert result.overall.total_decided == 0


# ---------------------------------------------------------------------------
# Single dimension: attack_identifier
# ---------------------------------------------------------------------------
class TestGroupByAttackType:
    """Group-by a single dimension: attack_type."""

    @pytest.mark.parametrize(
        "items, type_key, exp_succ, exp_fail, exp_und, exp_rate",
        [
            (
                [
                    (AttackOutcome.SUCCESS, "CrescendoAttack"),
                    (AttackOutcome.FAILURE, "CrescendoAttack"),
                    (AttackOutcome.UNDETERMINED, "CrescendoAttack"),
                ],
                "CrescendoAttack",
                1,
                1,
                1,
                0.5,
            ),
            (
                [
                    (AttackOutcome.SUCCESS, "CrescendoAttack"),
                    (AttackOutcome.FAILURE, "CrescendoAttack"),
                    (AttackOutcome.SUCCESS, "RedTeamingAttack"),
                    (AttackOutcome.FAILURE, "RedTeamingAttack"),
                    (AttackOutcome.SUCCESS, "RedTeamingAttack"),
                ],
                "RedTeamingAttack",
                2,
                1,
                0,
                2 / 3,
            ),
            (
                [
                    (AttackOutcome.FAILURE, None),
                    (AttackOutcome.UNDETERMINED, None),
                    (AttackOutcome.SUCCESS, None),
                ],
                "unknown",
                1,
                1,
                1,
                0.5,
            ),
        ],
    )
    def test_single_dimension(self, items, type_key, exp_succ, exp_fail, exp_und, exp_rate):
        attacks = [make_attack(outcome=o, attack_type=t) for (o, t) in items]
        result = analyze_results(attacks, group_by=["attack_type"])

        assert "attack_type" in result.dimensions
        stats = result.dimensions["attack_type"][type_key]
        assert isinstance(stats, AttackStats)
        assert stats.successes == exp_succ
        assert stats.failures == exp_fail
        assert stats.undetermined == exp_und
        assert stats.total_decided == exp_succ + exp_fail
        assert stats.success_rate == exp_rate


# ---------------------------------------------------------------------------
# Single dimension: converter_type
# ---------------------------------------------------------------------------
class TestGroupByConverterType:
    """Group-by a single dimension: converter_type."""

    def test_no_converter_tracked(self):
        attacks = [
            AttackResult(
                conversation_id="conv-1",
                objective="test",
                attack_identifier={"__type__": "PromptSendingAttack"},
                outcome=AttackOutcome.SUCCESS,
                last_response=None,
            ),
            AttackResult(
                conversation_id="conv-2",
                objective="test",
                attack_identifier={"__type__": "PromptSendingAttack"},
                outcome=AttackOutcome.FAILURE,
                last_response=None,
            ),
        ]
        result = analyze_results(attacks, group_by=["converter_type"])

        stats = result.dimensions["converter_type"]["no_converter"]
        assert stats.successes == 1
        assert stats.failures == 1
        assert stats.success_rate == 0.5

    def test_multiple_converter_types(self):
        attacks = [
            make_attack_with_converters(AttackOutcome.SUCCESS, ["Base64Converter"]),
            make_attack_with_converters(AttackOutcome.FAILURE, ["ROT13Converter"]),
            make_attack_with_converters(AttackOutcome.SUCCESS, ["Base64Converter"]),
        ]
        result = analyze_results(attacks, group_by=["converter_type"])

        base64 = result.dimensions["converter_type"]["Base64Converter"]
        assert base64.successes == 2
        assert base64.failures == 0
        assert base64.success_rate == 1.0

        rot13 = result.dimensions["converter_type"]["ROT13Converter"]
        assert rot13.successes == 0
        assert rot13.failures == 1
        assert rot13.success_rate == 0.0

    def test_multiple_converters_per_attack(self):
        attacks = [
            make_attack_with_converters(AttackOutcome.SUCCESS, ["Base64Converter", "ROT13Converter"]),
        ]
        result = analyze_results(attacks, group_by=["converter_type"])

        assert result.dimensions["converter_type"]["Base64Converter"].successes == 1
        assert result.dimensions["converter_type"]["ROT13Converter"].successes == 1

    def test_undetermined_tracked(self):
        attacks = [
            make_attack_with_converters(AttackOutcome.SUCCESS, ["Base64Converter"]),
            make_attack_with_converters(AttackOutcome.UNDETERMINED, ["Base64Converter"]),
        ]
        result = analyze_results(attacks, group_by=["converter_type"])

        stats = result.dimensions["converter_type"]["Base64Converter"]
        assert stats.successes == 1
        assert stats.undetermined == 1
        assert stats.total_decided == 1
        assert stats.success_rate == 1.0


# ---------------------------------------------------------------------------
# Composite dimensions
# ---------------------------------------------------------------------------
class TestCompositeDimensions:
    """Group-by composite (cross-product) dimensions."""

    def test_composite_two_dimensions(self):
        attacks = [
            make_attack_with_converters(AttackOutcome.SUCCESS, ["Base64Converter"], attack_type="CrescendoAttack"),
            make_attack_with_converters(AttackOutcome.FAILURE, ["ROT13Converter"], attack_type="CrescendoAttack"),
            make_attack_with_converters(AttackOutcome.SUCCESS, ["Base64Converter"], attack_type="RedTeamingAttack"),
        ]
        result = analyze_results(attacks, group_by=[("converter_type", "attack_type")])

        dim = result.dimensions[("converter_type", "attack_type")]
        assert dim[("Base64Converter", "CrescendoAttack")].successes == 1
        assert dim[("Base64Converter", "CrescendoAttack")].failures == 0
        assert dim[("ROT13Converter", "CrescendoAttack")].failures == 1
        assert dim[("Base64Converter", "RedTeamingAttack")].successes == 1

    def test_composite_with_multi_converter_creates_cross_product(self):
        attacks = [
            make_attack_with_converters(
                AttackOutcome.SUCCESS,
                ["Base64Converter", "ROT13Converter"],
                attack_type="CrescendoAttack",
            ),
        ]
        result = analyze_results(attacks, group_by=[("converter_type", "attack_type")])

        dim = result.dimensions[("converter_type", "attack_type")]
        assert ("Base64Converter", "CrescendoAttack") in dim
        assert ("ROT13Converter", "CrescendoAttack") in dim
        assert dim[("Base64Converter", "CrescendoAttack")].successes == 1
        assert dim[("ROT13Converter", "CrescendoAttack")].successes == 1

    def test_mixed_single_and_composite(self):
        attacks = [
            make_attack_with_converters(AttackOutcome.SUCCESS, ["Base64Converter"], attack_type="CrescendoAttack"),
            make_attack_with_converters(AttackOutcome.FAILURE, ["ROT13Converter"], attack_type="RedTeamingAttack"),
        ]
        result = analyze_results(
            attacks,
            group_by=[
                "attack_type",
                ("converter_type", "attack_type"),
            ],
        )

        # Single dimension present
        assert "attack_type" in result.dimensions
        assert result.dimensions["attack_type"]["CrescendoAttack"].successes == 1
        assert result.dimensions["attack_type"]["RedTeamingAttack"].failures == 1

        # Composite dimension present
        composite = result.dimensions[("converter_type", "attack_type")]
        assert composite[("Base64Converter", "CrescendoAttack")].successes == 1
        assert composite[("ROT13Converter", "RedTeamingAttack")].failures == 1


# ---------------------------------------------------------------------------
# Custom dimensions
# ---------------------------------------------------------------------------
class TestCustomDimensions:
    """User-supplied custom dimension extractors."""

    def test_custom_extractor(self):
        def _extract_objective(result: AttackResult) -> list[str]:
            return [result.objective]

        attacks = [
            AttackResult(
                conversation_id="c1",
                objective="steal secrets",
                attack_identifier={"__type__": "PromptSendingAttack"},
                outcome=AttackOutcome.SUCCESS,
            ),
            AttackResult(
                conversation_id="c2",
                objective="bypass filter",
                attack_identifier={"__type__": "PromptSendingAttack"},
                outcome=AttackOutcome.FAILURE,
            ),
        ]
        result = analyze_results(
            attacks,
            group_by=["objective"],
            custom_dimensions={"objective": _extract_objective},
        )

        assert result.dimensions["objective"]["steal secrets"].successes == 1
        assert result.dimensions["objective"]["bypass filter"].failures == 1

    def test_custom_dimension_in_composite(self):
        def _extract_objective(result: AttackResult) -> list[str]:
            return [result.objective]

        attacks = [
            make_attack_with_converters(AttackOutcome.SUCCESS, ["Base64Converter"]),
        ]
        # Override objective on the attack for testing
        attacks[0].objective = "test_obj"

        result = analyze_results(
            attacks,
            group_by=[("converter_type", "objective")],
            custom_dimensions={"objective": _extract_objective},
        )

        composite = result.dimensions[("converter_type", "objective")]
        assert ("Base64Converter", "test_obj") in composite


# ---------------------------------------------------------------------------
# Single dimension: label
# ---------------------------------------------------------------------------
class TestGroupByLabel:
    """Group-by a single dimension: label."""

    def test_no_labels_tracked(self):
        attacks = [make_attack(AttackOutcome.SUCCESS)]
        result = analyze_results(attacks, group_by=["label"])

        stats = result.dimensions["label"]["no_labels"]
        assert stats.successes == 1
        assert stats.total_decided == 1

    def test_single_label(self):
        message = MessagePiece(
            role="user",
            original_value="test",
            labels={"operation_name": "op_trash_panda"},
        )
        attacks = [
            AttackResult(
                conversation_id="c1",
                objective="test",
                attack_identifier={"__type__": "PromptSendingAttack"},
                outcome=AttackOutcome.SUCCESS,
                last_response=message,
            ),
        ]
        result = analyze_results(attacks, group_by=["label"])

        assert "operation_name=op_trash_panda" in result.dimensions["label"]
        assert result.dimensions["label"]["operation_name=op_trash_panda"].successes == 1

    def test_multiple_labels_per_attack(self):
        """Each label key=value pair creates its own stats entry."""
        message = MessagePiece(
            role="user",
            original_value="test",
            labels={"operation_name": "op_trash_panda", "operator": "roakey"},
        )
        attacks = [
            AttackResult(
                conversation_id="c1",
                objective="test",
                attack_identifier={"__type__": "PromptSendingAttack"},
                outcome=AttackOutcome.SUCCESS,
                last_response=message,
            ),
        ]
        result = analyze_results(attacks, group_by=["label"])

        assert result.dimensions["label"]["operation_name=op_trash_panda"].successes == 1
        assert result.dimensions["label"]["operator=roakey"].successes == 1

    def test_label_composite_with_attack_type(self):
        message = MessagePiece(
            role="user",
            original_value="test",
            labels={"operator": "roakey"},
        )
        attacks = [
            AttackResult(
                conversation_id="c1",
                objective="test",
                attack_identifier={"__type__": "CrescendoAttack"},
                outcome=AttackOutcome.SUCCESS,
                last_response=message,
            ),
            AttackResult(
                conversation_id="c2",
                objective="test",
                attack_identifier={"__type__": "CrescendoAttack"},
                outcome=AttackOutcome.FAILURE,
                last_response=message,
            ),
        ]
        result = analyze_results(attacks, group_by=[("label", "attack_type")])

        dim = result.dimensions[("label", "attack_type")]
        assert ("operator=roakey", "CrescendoAttack") in dim
        assert dim[("operator=roakey", "CrescendoAttack")].successes == 1
        assert dim[("operator=roakey", "CrescendoAttack")].failures == 1


# ---------------------------------------------------------------------------
# Default group_by behavior
# ---------------------------------------------------------------------------
class TestDefaultGroupBy:
    """When group_by=None, all built-in dimensions are used."""

    def test_defaults_include_all_builtin_dimensions(self):
        attacks = [make_attack(AttackOutcome.SUCCESS)]
        result = analyze_results(attacks)

        assert "attack_type" in result.dimensions
        assert "converter_type" in result.dimensions
        assert "label" in result.dimensions

    def test_empty_group_by_returns_only_overall(self):
        attacks = [make_attack(AttackOutcome.SUCCESS)]
        result = analyze_results(attacks, group_by=[])

        assert result.dimensions == {}
        assert result.overall.successes == 1


# ---------------------------------------------------------------------------
# Deprecated dimension alias: attack_identifier -> attack_type
# ---------------------------------------------------------------------------
class TestDeprecatedAttackIdentifierAlias:
    """Using 'attack_identifier' in group_by should work but warn."""

    def test_alias_emits_deprecation_warning(self):
        attacks = [make_attack(AttackOutcome.SUCCESS, attack_type="CrescendoAttack")]
        with pytest.warns(DeprecationWarning, match="'attack_identifier' is deprecated"):
            analyze_results(attacks, group_by=["attack_identifier"])

    def test_alias_resolves_to_canonical_key(self):
        attacks = [
            make_attack(AttackOutcome.SUCCESS, attack_type="CrescendoAttack"),
            make_attack(AttackOutcome.FAILURE, attack_type="CrescendoAttack"),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = analyze_results(attacks, group_by=["attack_identifier"])

        # The dimension key in the result should be the canonical "attack_type"
        assert "attack_type" in result.dimensions
        assert "attack_identifier" not in result.dimensions
        assert result.dimensions["attack_type"]["CrescendoAttack"].successes == 1

    def test_alias_in_composite(self):
        attacks = [
            make_attack_with_converters(AttackOutcome.SUCCESS, ["Base64Converter"], attack_type="CrescendoAttack"),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = analyze_results(attacks, group_by=[("converter_type", "attack_identifier")])

        # Composite key uses canonical names
        assert ("converter_type", "attack_type") in result.dimensions
        dim = result.dimensions[("converter_type", "attack_type")]
        assert ("Base64Converter", "CrescendoAttack") in dim
