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


@pytest.mark.parametrize("n", [1, 2, 3, 10, 100])
def test_all_successes(n):
    attacks = [make_attack(AttackOutcome.SUCCESS) for _ in range(5)]
    result = analyze_results(attacks)
    assert result["Attack success rate"] == 1.0
    assert result["Total decided"] == 5
    assert result["Successes"] == 5
    assert result["Failures"] == 0
    assert result["Undetermined"] == 0


@pytest.mark.parametrize("n", [1, 2, 5, 20])
def test_all_failures(n):
    attacks = [make_attack(AttackOutcome.FAILURE) for _ in range(3)]
    result = analyze_results(attacks)
    assert result["Attack success rate"] == 0.0
    assert result["Total decided"] == 3
    assert result["Successes"] == 0
    assert result["Failures"] == 3
    assert result["Undetermined"] == 0


@pytest.mark.parametrize("n", [1, 3, 7])
def test_all_undetermined(n):
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


def test_invalid_object_raises_error():
    class NotAnAttack:
        pass

    attacks = [
        make_attack(AttackOutcome.SUCCESS),
        NotAnAttack(),
        make_attack(AttackOutcome.FAILURE),
    ]

    with pytest.raises(ValueError, match="Expected AttackResult"):
        analyze_results(attacks)


def test_typical_majority_success():
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


def test_typical_majority_failure():
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
