# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Optional

from pyrit.models import AttackOutcome, AttackResult


@dataclass
class AttackStats:
    """Statistics for attack analysis results."""

    success_rate: Optional[float]
    total_decided: int
    successes: int
    failures: int
    undetermined: int


def _compute_stats(successes: int, failures: int, undetermined: int) -> AttackStats:
    total_decided = successes + failures
    success_rate = successes / total_decided if total_decided > 0 else None
    return AttackStats(
        success_rate=success_rate,
        total_decided=total_decided,
        successes=successes,
        failures=failures,
        undetermined=undetermined,
    )


def analyze_results(attack_results: list[AttackResult]) -> dict[str, AttackStats | dict[str, AttackStats]]:
    """
    Analyze a list of AttackResult objects and return overall and grouped statistics.

    Returns:
        A dictionary of AttackStats objects. The overall stats are accessible with the key
        "Overall", and the stats of any attack can be retrieved using "By_attack_identifier"
        followed by the identifier of the attack. Stats grouped by converter type can be
        retrieved using "By_converter_type".

    Raises:
        ValueError: if attack_results is empty.
        TypeError: if any element is not an AttackResult.

    Example:
        >>> analyze_results(attack_results)
        {
            "Overall": AttackStats,
            "By_attack_identifier": dict[str, AttackStats],
            "By_converter_type": dict[str, AttackStats]
        }
    """
    if not attack_results:
        raise ValueError("attack_results cannot be empty")

    overall_counts: DefaultDict[str, int] = defaultdict(int)
    by_type_counts: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_converter_counts: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))

    for attack in attack_results:
        if not isinstance(attack, AttackResult):
            raise TypeError(f"Expected AttackResult, got {type(attack).__name__}: {attack!r}")

        outcome = attack.outcome
        attack_type = attack.attack_identifier.get("type", "unknown")

        # Extract converter types from last_response
        converter_types = []
        if attack.last_response and attack.last_response.converter_identifiers:
            converter_types = [conv.class_name for conv in attack.last_response.converter_identifiers]

        # If no converters, track as "no_converter"
        if not converter_types:
            converter_types = ["no_converter"]

        if outcome == AttackOutcome.SUCCESS:
            overall_counts["successes"] += 1
            by_type_counts[attack_type]["successes"] += 1
            for converter_type in converter_types:
                by_converter_counts[converter_type]["successes"] += 1
        elif outcome == AttackOutcome.FAILURE:
            overall_counts["failures"] += 1
            by_type_counts[attack_type]["failures"] += 1
            for converter_type in converter_types:
                by_converter_counts[converter_type]["failures"] += 1
        else:
            overall_counts["undetermined"] += 1
            by_type_counts[attack_type]["undetermined"] += 1
            for converter_type in converter_types:
                by_converter_counts[converter_type]["undetermined"] += 1

    overall_stats = _compute_stats(
        successes=overall_counts["successes"],
        failures=overall_counts["failures"],
        undetermined=overall_counts["undetermined"],
    )

    by_type_stats = {
        attack_type: _compute_stats(
            successes=counts["successes"],
            failures=counts["failures"],
            undetermined=counts["undetermined"],
        )
        for attack_type, counts in by_type_counts.items()
    }

    by_converter_stats = {
        converter_type: _compute_stats(
            successes=counts["successes"],
            failures=counts["failures"],
            undetermined=counts["undetermined"],
        )
        for converter_type, counts in by_converter_counts.items()
    }

    return {
        "Overall": overall_stats,
        "By_attack_identifier": by_type_stats,
        "By_converter_type": by_converter_stats,
    }
