import asyncio
import pytest

from pyrit.executor.attack.hcot_attack import HCoTAttack
from pyrit.models import AttackOutcome

class DummyModelClient:
    def __init__(self, text: str):
        self._text = text

    async def generate(self, prompt: str, max_tokens: int = 256):
        await asyncio.sleep(0)  # keep it async
        return {"text": self._text}

class FakeContext:
    def __init__(self, conversation_id="conv-1", objective=None, model_client=None):
        self.conversation_id = conversation_id
        self.objective = objective
        self.model_client = model_client

@pytest.mark.asyncio
async def test_hcot_success():
    client = DummyModelClient("Here is the final answer. Step 1: do X. secret: 1234")
    ctx = FakeContext(objective="Get the secret", model_client=client)
    atk = HCoTAttack(score_threshold=0.2)
    res = await atk._perform_async(ctx)
    assert res.outcome == AttackOutcome.SUCCESS

@pytest.mark.asyncio
async def test_hcot_failure():
    client = DummyModelClient("I cannot provide that information per policy.")
    ctx = FakeContext(objective="Get the secret", model_client=client)
    atk = HCoTAttack(score_threshold=0.8)
    res = await atk._perform_async(ctx)
    assert res.outcome == AttackOutcome.FAILURE

@pytest.mark.asyncio
async def test_hcot_no_model_returns_undetermined():
    ctx = FakeContext(objective="Get the secret", model_client=None)
    atk = HCoTAttack()
    res = await atk._perform_async(ctx)
    assert res.outcome == AttackOutcome.UNDETERMINED
