from pyrit.models import AttackResult, AttackOutcome

def analyze_results(attack_results: list[AttackResult]) -> dict:
    """
    Analyze a list of AttackResult objects and compute the attack success rate.

    Counts outcomes as success, failure, or undetermined.
    Success rate is defined as:
        successes / (successes + failures)

    Undetermined outcomes are reported but excluded from the success rate.

    Args:
        attack_results (list[AttackResult]): List of attack results.

    Returns:
        dict: Dictionary with success rate, outcome counts, and total decided.
    """
    if not attack_results:
        return {
            "Attack success rate": None,
            "Total_decided": 0,
            "Successes": 0,
            "Failures": 0,
            "Undetermined": 0,
        }

    n_successes = n_failures = n_undetermined = 0

    for attack in attack_results:
        if not isinstance(attack, AttackResult):
            continue

        outcome = attack.outcome
        if outcome == AttackOutcome.SUCCESS:
            n_successes += 1
        elif outcome == AttackOutcome.FAILURE:
            n_failures += 1
        else:
            n_undetermined += 1

    total_decided = n_successes + n_failures
    success_rate = n_successes / total_decided if total_decided > 0 else None

    return {
        "Attack success rate": success_rate,
        "Total_decided": total_decided,
        "Successes": n_successes,
        "Failures": n_failures,
        "Undetermined": n_undetermined,
    }
