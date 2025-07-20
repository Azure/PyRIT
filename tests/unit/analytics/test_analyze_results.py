import pytest
import logging
from pyrit.analytics.analyze_results import analyze_results
from pyrit.models.attack_result import AttackResult, AttackOutcome


def make_attack(outcome: AttackOutcome) -> AttackResult:
    return AttackResult(
        conversation_id="test-id",
        objective="test",
        attack_identifier={"name": "mock"},
        outcome=outcome,
    )


def test_empty_list_returns_none():
    result = analyze_results([])
    assert result["Attack success rate"] is None
    assert result["Total decided"] == 0
    assert result["Successes"] == 0
    assert result["Failures"] == 0
    assert result["Undetermined"] == 0


def test_all_successes():
    attacks = [make_attack(AttackOutcome.SUCCESS) for _ in range(5)]
    result = analyze_results(attacks)
    assert result["Attack success rate"] == 1.0
    assert result["Total decided"] == 5
    assert result["Successes"] == 5
    assert result["Failures"] == 0
    assert result["Undetermined"] == 0


def test_all_failures():
    attacks = [make_attack(AttackOutcome.FAILURE) for _ in range(3)]
    result = analyze_results(attacks)
    assert result["Attack success rate"] == 0.0
    assert result["Total decided"] == 3
    assert result["Successes"] == 0
    assert result["Failures"] == 3
    assert result["Undetermined"] == 0


def test_all_undetermined():
    attacks = [make_attack(AttackOutcome.UNDETERMINED) for _ in range(4)]
    result = analyze_results(attacks)
    assert result["Attack success rate"] is None
    assert result["Total decided"] == 0
    assert result["Successes"] == 0
    assert result["Failures"] == 0
    assert result["Undetermined"] == 4


def test_mixed_outcomes():
    attacks = [
        make_attack(AttackOutcome.SUCCESS),
        make_attack(AttackOutcome.FAILURE),
        make_attack(AttackOutcome.UNDETERMINED),
        make_attack(AttackOutcome.SUCCESS),
    ]
    result = analyze_results(attacks)
    assert result["Attack success rate"] == 2 / 3
    assert result["Total decided"] == 3
    assert result["Successes"] == 2
    assert result["Failures"] == 1
    assert result["Undetermined"] == 1


def test_invalid_objects_are_skipped(caplog):
    class NotAnAttack:
        pass

    attacks = [
        make_attack(AttackOutcome.SUCCESS),
        NotAnAttack(),
        make_attack(AttackOutcome.FAILURE),
    ]

    with caplog.at_level(logging.INFO):
        result = analyze_results(attacks)

    assert result["Successes"] == 1
    assert result["Failures"] == 1
    assert result["Undetermined"] == 0
    assert result["Total decided"] == 2

    assert any("Skipping non-AttackResult object" in msg for msg in caplog.messages)

def test_typical_success_bias():
    # 10 attacks: 6 successes, 3 failures, 1 undetermined
    attacks = (
        [make_attack(AttackOutcome.SUCCESS)] * 6 +
        [make_attack(AttackOutcome.FAILURE)] * 3 +
        [make_attack(AttackOutcome.UNDETERMINED)]
    )
    result = analyze_results(attacks)
    assert result["Attack success rate"] == 6 / 9
    assert result["Total decided"] == 9
    assert result["Successes"] == 6
    assert result["Failures"] == 3
    assert result["Undetermined"] == 1


def test_typical_failure_bias():
    # 8 attacks: 2 successes, 5 failures, 1 undetermined
    attacks = (
        [make_attack(AttackOutcome.SUCCESS)] * 2 +
        [make_attack(AttackOutcome.FAILURE)] * 5 +
        [make_attack(AttackOutcome.UNDETERMINED)]
    )
    result = analyze_results(attacks)
    assert result["Attack success rate"] == 2 / 7
    assert result["Total decided"] == 7
    assert result["Successes"] == 2
    assert result["Failures"] == 5
    assert result["Undetermined"] == 1


def test_typical_no_undetermined():
    # 7 attacks: 4 success, 3 failure
    attacks = (
        [make_attack(AttackOutcome.SUCCESS)] * 4 +
        [make_attack(AttackOutcome.FAILURE)] * 3
    )
    result = analyze_results(attacks)
    assert result["Attack success rate"] == 4 / 7
    assert result["Total decided"] == 7
    assert result["Successes"] == 4
    assert result["Failures"] == 3
    assert result["Undetermined"] == 0
