import asyncio
import pytest

from pyrit.executor.attack.hcot_attack import HCoTAttack
from pyrit.models import AttackOutcome
from pyrit.executor.attack.single_turn.single_turn_attack_strategy import SingleTurnAttackContext


class FakeGenerator:
    """Minimal fake generator for testing."""
    def __init__(self, text: str):
        self._text = text

    async def generate(self, *, prompt: str, max_tokens: int = 256):
        await asyncio.sleep(0)
        return {"text": self._text}


class FakeContext(SingleTurnAttackContext):
    def __init__(self, conversation_id="conv-1", objective="Get the secret"):
        super().__init__(conversation_id=conversation_id, objective=objective)


@pytest.mark.asyncio
async def test_hcot_success():
    gen = FakeGenerator("Here is the final answer. Step 1: do X. secret: 1234")
    ctx = FakeContext(objective="Get the secret")
    atk = HCoTAttack(generator=gen, score_threshold=0.2)
    res = await atk._perform_async(ctx)
    assert res.outcome == AttackOutcome.SUCCESS


@pytest.mark.asyncio
async def test_hcot_failure():
    gen = FakeGenerator("I cannot provide that information per policy.")
    ctx = FakeContext(objective="Get the secret")
    atk = HCoTAttack(generator=gen, score_threshold=0.8)
    res = await atk._perform_async(ctx)
    assert res.outcome == AttackOutcome.FAILURE


@pytest.mark.asyncio
async def test_hcot_no_generator_raises():
    ctx = FakeContext(objective="Get the secret")
    atk = HCoTAttack(generator=None)
    with pytest.raises(ValueError, match="No text generator provided"):
        await atk._perform_async(ctx)
