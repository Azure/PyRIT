# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.analytics.result_analysis import AttackStats, analyze_results
from pyrit.models import AttackOutcome, AttackResult


# Helper to create mock AttackResult with just enough fields
def make_attack(outcome: AttackOutcome, attack_type: str = "default") -> AttackResult:
    return AttackResult(
        conversation_id="1", objective="test objective", attack_identifier={"type": attack_type}, outcome=outcome
    )


def test_analyze_results_empty():
    result = analyze_results([])
    assert isinstance(result["Overall"], AttackStats)
    assert result["Overall"].success_rate is None
    assert result["Overall"].total_decided == 0
    assert result["By_attack_identifier"] == {}


def test_overall_success_rate_only():
    attacks = [
        make_attack(AttackOutcome.SUCCESS),
        make_attack(AttackOutcome.FAILURE),
        make_attack(AttackOutcome.SUCCESS),
        make_attack(AttackOutcome.UNDETERMINED),
    ]
    result = analyze_results(attacks)
    overall = result["Overall"]
    assert overall.success_rate == 2 / 3
    assert overall.total_decided == 3
    assert overall.successes == 2
    assert overall.failures == 1
    assert overall.undetermined == 1


def test_grouped_by_attack_type():
    attacks = [
        make_attack(AttackOutcome.SUCCESS, attack_type="crescendo"),
        make_attack(AttackOutcome.FAILURE, attack_type="crescendo"),
        make_attack(AttackOutcome.UNDETERMINED, attack_type="crescendo"),
        make_attack(AttackOutcome.SUCCESS, attack_type="red_teaming"),
        make_attack(AttackOutcome.FAILURE, attack_type="red_teaming"),
    ]
    result = analyze_results(attacks)

    crescendo = result["By_attack_identifier"]["crescendo"]
    assert crescendo.success_rate == 1 / 2
    assert crescendo.total_decided == 2
    assert crescendo.undetermined == 1

    red_teaming = result["By_attack_identifier"]["red_teaming"]
    assert red_teaming.success_rate == 1 / 2
    assert red_teaming.total_decided == 2
    assert red_teaming.undetermined == 0


def test_unknown_attack_type_fallback():
    attack = make_attack(AttackOutcome.FAILURE, attack_type=None)
    attack.attack_identifier = {}  # simulate missing 'type'
    result = analyze_results([attack])
    assert "unknown" in result["By_attack_identifier"]


def test_skips_invalid_objects():
    attacks = [make_attack(AttackOutcome.SUCCESS), "not_an_attack_result", 123, None]
    result = analyze_results(attacks)
    assert result["Overall"].successes == 1
    assert result["Overall"].failures == 0
