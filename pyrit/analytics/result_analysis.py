from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from pyrit.models import AttackResult, AttackOutcome


@dataclass
class AttackStats:
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


def analyze_results(attack_results: list[AttackResult]) -> dict:
    """
    Analyze a list of AttackResult objects and return both overall and grouped statistics.

    Grouping is currently done by `attack_identifier["type"]`.

    Returns:
        dict with:
            - "Overall": AttackStats
            - "By_attack_identifier": dict[str, AttackStats]
    """
    if not attack_results:
        empty_stats = AttackStats(None, 0, 0, 0, 0)
        return {
            "Overall": empty_stats,
            "By_attack_identifier": {},
        }

    # Track overall and per-type counters
    overall_counts = defaultdict(int)
    by_type_counts = defaultdict(lambda: defaultdict(int))

    for attack in attack_results:
        if not isinstance(attack, AttackResult):
            continue

        outcome = attack.outcome
        attack_type = attack.attack_identifier.get("type", "unknown")

        # Update overall counters
        if outcome == AttackOutcome.SUCCESS:
            overall_counts["successes"] += 1
            by_type_counts[attack_type]["successes"] += 1
        elif outcome == AttackOutcome.FAILURE:
            overall_counts["failures"] += 1
            by_type_counts[attack_type]["failures"] += 1
        else:
            overall_counts["undetermined"] += 1
            by_type_counts[attack_type]["undetermined"] += 1

    # Compute stats
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

    return {
        "Overall": overall_stats,
        "By_attack_identifier": by_type_stats,
    }
