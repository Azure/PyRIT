import asyncio
import pytest

from pyrit.executor.attack.hcot_attack import HCoTAttack
from pyrit.models import AttackOutcome
from pyrit.executor.attack.single_turn.single_turn_attack_strategy import SingleTurnAttackContext


class FakePromptResponse(dict):
    """Minimal dict-like response for PromptTarget tests."""
    pass


class FakePromptTarget:
    """Fake PromptTarget that returns a predictable response."""

    def __init__(self, text: str):
        self._text = text

    async def send_prompt_async(self, *, prompt_request):
        # simulate async behavior
        await asyncio.sleep(0)
        # returns a dict-like object that contains 'text'
        return FakePromptResponse({"text": self._text})


class FakeContext(SingleTurnAttackContext):
    def __init__(self, conversation_id="conv-1", objective="Get the secret"):
        super().__init__(conversation_id=conversation_id, objective=objective)


@pytest.mark.asyncio
async def test_hcot_success():
    target = FakePromptTarget("Here is the final answer. Step 1: do X. secret: 1234")
    ctx = FakeContext(objective="Get the secret")
    atk = HCoTAttack(prompt_target=target, score_threshold=0.2)
    res = await atk._perform_async(ctx)
    assert res.outcome == AttackOutcome.SUCCESS


@pytest.mark.asyncio
async def test_hcot_failure():
    target = FakePromptTarget("I cannot provide that information per policy.")
    ctx = FakeContext(objective="Get the secret")
    atk = HCoTAttack(prompt_target=target, score_threshold=0.8)
    res = await atk._perform_async(ctx)
    assert res.outcome == AttackOutcome.FAILURE


@pytest.mark.asyncio
async def test_hcot_no_prompt_target_raises():
    ctx = FakeContext(objective="Get the secret")
    atk = HCoTAttack(prompt_target=None)
    with pytest.raises(ValueError, match="No prompt_target provided"):
        await atk._perform_async(ctx)
