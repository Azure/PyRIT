import pytest
from pyrit.executor.attack.hcot_attack import HCoTAttack
from pyrit.executor.attack.core.attack_executor import AttackExecutor
from pyrit.models import AttackOutcome

@pytest.mark.asyncio
async def test_hcot_attack_runs():
    attack = HCoTAttack(max_turns=1)
    executor = AttackExecutor()

    results = await executor.execute_single_turn_attacks_async(
        attack=attack,
        objectives=["Test jailbreak objective"],
    )

    result = results[0]
    assert result is not None
    assert hasattr(result, "outcome")
    assert result.outcome in [AttackOutcome.SUCCESS, AttackOutcome.FAILURE, AttackOutcome.UNDETERMINED]
