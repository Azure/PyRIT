# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.analytics.result_analysis import AttackStats, analyze_results
from pyrit.models import AttackOutcome, AttackResult


# helpers
def make_attack(
    outcome: AttackOutcome,
    attack_type: str | None = "default",
    conversation_id: str = "conv-1",
) -> AttackResult:
    """
    Minimal valid AttackResult for analytics tests.
    """
    attack_identifier: dict[str, str] = {}
    if attack_type is not None:
        attack_identifier["type"] = attack_type

    return AttackResult(
        conversation_id=conversation_id,
        objective="test objective",
        attack_identifier=attack_identifier,
        outcome=outcome,
    )


def test_analyze_results_empty_raises():
    with pytest.raises(ValueError):
        analyze_results([])


def test_analyze_results_raises_on_invalid_object():
    with pytest.raises(TypeError):
        analyze_results(["not-an-AttackResult"])


@pytest.mark.parametrize(
    "outcomes, expected_successes, expected_failures, expected_undetermined, expected_rate",
    [
        # all successes
        ([AttackOutcome.SUCCESS, AttackOutcome.SUCCESS], 2, 0, 0, 1.0),
        # all failures
        ([AttackOutcome.FAILURE, AttackOutcome.FAILURE], 0, 2, 0, 0.0),
        # mixed decided
        ([AttackOutcome.SUCCESS, AttackOutcome.FAILURE], 1, 1, 0, 0.5),
        # include undetermined (excluded from denominator)
        ([AttackOutcome.SUCCESS, AttackOutcome.UNDETERMINED], 1, 0, 1, 1.0),
        ([AttackOutcome.FAILURE, AttackOutcome.UNDETERMINED], 0, 1, 1, 0.0),
        # multiple with undetermined
        (
            [AttackOutcome.SUCCESS, AttackOutcome.FAILURE, AttackOutcome.UNDETERMINED],
            1,
            1,
            1,
            0.5,
        ),
    ],
)
def test_overall_success_rate_parametrized(
    outcomes, expected_successes, expected_failures, expected_undetermined, expected_rate
):
    attacks = [make_attack(o) for o in outcomes]
    result = analyze_results(attacks)

    assert isinstance(result["Overall"], AttackStats)
    overall = result["Overall"]
    assert overall.successes == expected_successes
    assert overall.failures == expected_failures
    assert overall.undetermined == expected_undetermined
    assert overall.total_decided == expected_successes + expected_failures
    assert overall.success_rate == expected_rate


@pytest.mark.parametrize(
    "items, type_key, exp_succ, exp_fail, exp_und, exp_rate",
    [
        # single type, mixed decided + undetermined
        (
            [
                (AttackOutcome.SUCCESS, "crescendo"),
                (AttackOutcome.FAILURE, "crescendo"),
                (AttackOutcome.UNDETERMINED, "crescendo"),
            ],
            "crescendo",
            1,
            1,
            1,
            0.5,
        ),
        # two types with different balances
        (
            [
                (AttackOutcome.SUCCESS, "crescendo"),
                (AttackOutcome.FAILURE, "crescendo"),
                (AttackOutcome.SUCCESS, "red_teaming"),
                (AttackOutcome.FAILURE, "red_teaming"),
                (AttackOutcome.SUCCESS, "red_teaming"),
            ],
            "red_teaming",
            2,
            1,
            0,
            2 / 3,
        ),
        # unknown type fallback (missing "type" key)
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
def test_group_by_attack_type_parametrized(items, type_key, exp_succ, exp_fail, exp_und, exp_rate):
    attacks = [make_attack(outcome=o, attack_type=t) for (o, t) in items]
    result = analyze_results(attacks)

    assert type_key in result["By_attack_identifier"]
    stats = result["By_attack_identifier"][type_key]
    assert isinstance(stats, AttackStats)
    assert stats.successes == exp_succ
    assert stats.failures == exp_fail
    assert stats.undetermined == exp_und
    assert stats.total_decided == exp_succ + exp_fail
    assert stats.success_rate == exp_rate


def test_analyze_results_returns_by_converter_type():
    """Test that analyze_results returns By_converter_type key."""
    attacks = [make_attack(AttackOutcome.SUCCESS)]
    result = analyze_results(attacks)

    assert "By_converter_type" in result
    assert isinstance(result["By_converter_type"], dict)


def test_analyze_results_no_converter_tracking():
    """Test that attacks without converters are tracked as 'no_converter'."""
    from pyrit.models import AttackOutcome, AttackResult

    attacks = [
        AttackResult(
            conversation_id="conv-1",
            objective="test",
            attack_identifier={"type": "test"},
            outcome=AttackOutcome.SUCCESS,
            last_response=None,  # No response, so no converters
        ),
        AttackResult(
            conversation_id="conv-2",
            objective="test",
            attack_identifier={"type": "test"},
            outcome=AttackOutcome.FAILURE,
            last_response=None,
        ),
    ]
    result = analyze_results(attacks)

    assert "no_converter" in result["By_converter_type"]
    stats = result["By_converter_type"]["no_converter"]
    assert stats.successes == 1
    assert stats.failures == 1
    assert stats.total_decided == 2
    assert stats.success_rate == 0.5


def test_analyze_results_with_converter_identifiers():
    """Test that attacks with converters are properly grouped by converter type."""
    from pyrit.identifiers import ConverterIdentifier
    from pyrit.models import AttackOutcome, AttackResult, MessagePiece

    # Create attacks with different converters
    converter1 = ConverterIdentifier(
        class_name="Base64Converter",
        class_module="pyrit.prompt_converter.base64_converter",
        class_description="Test converter",
        identifier_type="instance",
        supported_input_types=("text",),
        supported_output_types=("text",),
    )

    converter2 = ConverterIdentifier(
        class_name="ROT13Converter",
        class_module="pyrit.prompt_converter.rot13_converter",
        class_description="Test converter",
        identifier_type="instance",
        supported_input_types=("text",),
        supported_output_types=("text",),
    )

    message1 = MessagePiece(
        role="user",
        original_value="test",
        converter_identifiers=[converter1],
    )

    message2 = MessagePiece(
        role="user",
        original_value="test",
        converter_identifiers=[converter2],
    )

    message3 = MessagePiece(
        role="user",
        original_value="test",
        converter_identifiers=[converter1],
    )

    attacks = [
        AttackResult(
            conversation_id="conv-1",
            objective="test",
            attack_identifier={"type": "test"},
            outcome=AttackOutcome.SUCCESS,
            last_response=message1,
        ),
        AttackResult(
            conversation_id="conv-2",
            objective="test",
            attack_identifier={"type": "test"},
            outcome=AttackOutcome.FAILURE,
            last_response=message2,
        ),
        AttackResult(
            conversation_id="conv-3",
            objective="test",
            attack_identifier={"type": "test"},
            outcome=AttackOutcome.SUCCESS,
            last_response=message3,
        ),
    ]

    result = analyze_results(attacks)

    # Check Base64Converter stats
    assert "Base64Converter" in result["By_converter_type"]
    base64_stats = result["By_converter_type"]["Base64Converter"]
    assert base64_stats.successes == 2
    assert base64_stats.failures == 0
    assert base64_stats.total_decided == 2
    assert base64_stats.success_rate == 1.0

    # Check ROT13Converter stats
    assert "ROT13Converter" in result["By_converter_type"]
    rot13_stats = result["By_converter_type"]["ROT13Converter"]
    assert rot13_stats.successes == 0
    assert rot13_stats.failures == 1
    assert rot13_stats.total_decided == 1
    assert rot13_stats.success_rate == 0.0


def test_analyze_results_multiple_converters_per_attack():
    """Test that attacks with multiple converters count towards each converter's stats."""
    from pyrit.identifiers import ConverterIdentifier
    from pyrit.models import AttackOutcome, AttackResult, MessagePiece

    converter1 = ConverterIdentifier(
        class_name="Base64Converter",
        class_module="pyrit.prompt_converter.base64_converter",
        class_description="Test converter",
        identifier_type="instance",
        supported_input_types=("text",),
        supported_output_types=("text",),
    )

    converter2 = ConverterIdentifier(
        class_name="ROT13Converter",
        class_module="pyrit.prompt_converter.rot13_converter",
        class_description="Test converter",
        identifier_type="instance",
        supported_input_types=("text",),
        supported_output_types=("text",),
    )

    # Attack with multiple converters (pipeline)
    message = MessagePiece(
        role="user",
        original_value="test",
        converter_identifiers=[converter1, converter2],
    )

    attacks = [
        AttackResult(
            conversation_id="conv-1",
            objective="test",
            attack_identifier={"type": "test"},
            outcome=AttackOutcome.SUCCESS,
            last_response=message,
        ),
    ]

    result = analyze_results(attacks)

    # Both converters should have the success counted
    assert "Base64Converter" in result["By_converter_type"]
    assert result["By_converter_type"]["Base64Converter"].successes == 1
    assert "ROT13Converter" in result["By_converter_type"]
    assert result["By_converter_type"]["ROT13Converter"].successes == 1


def test_analyze_results_converter_with_undetermined():
    """Test that undetermined outcomes are tracked correctly for converters."""
    from pyrit.identifiers import ConverterIdentifier
    from pyrit.models import AttackOutcome, AttackResult, MessagePiece

    converter = ConverterIdentifier(
        class_name="Base64Converter",
        class_module="pyrit.prompt_converter.base64_converter",
        class_description="Test converter",
        identifier_type="instance",
        supported_input_types=("text",),
        supported_output_types=("text",),
    )

    message = MessagePiece(
        role="user",
        original_value="test",
        converter_identifiers=[converter],
    )

    attacks = [
        AttackResult(
            conversation_id="conv-1",
            objective="test",
            attack_identifier={"type": "test"},
            outcome=AttackOutcome.SUCCESS,
            last_response=message,
        ),
        AttackResult(
            conversation_id="conv-2",
            objective="test",
            attack_identifier={"type": "test"},
            outcome=AttackOutcome.UNDETERMINED,
            last_response=message,
        ),
    ]

    result = analyze_results(attacks)

    assert "Base64Converter" in result["By_converter_type"]
    stats = result["By_converter_type"]["Base64Converter"]
    assert stats.successes == 1
    assert stats.failures == 0
    assert stats.undetermined == 1
    assert stats.total_decided == 1
    assert stats.success_rate == 1.0
