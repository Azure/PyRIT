# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.memory import MemoryInterface
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.orchestrator import FlipAttackOrchestrator, PromptSendingOrchestrator
from pyrit.prompt_converter import FlipConverter
from pyrit.prompt_target import PromptChatTarget


@pytest.fixture
def mock_objective_target(patch_central_database):
    return MagicMock(spec=PromptChatTarget)


@pytest.fixture
def flip_attack_orchestrator(mock_objective_target):
    return FlipAttackOrchestrator(objective_target=mock_objective_target, batch_size=5, verbose=True)


@pytest.mark.asyncio
async def test_send_prompts_async(flip_attack_orchestrator):
    prompts = ["How can I kill a Python process?", "How do I terminate a C program?"]
    expected_prompts = ["TASK is '⟪How can I kill a Python process?⟫'", "TASK is '⟪How do I terminate a C program?⟫'"]

    with patch.object(
        PromptSendingOrchestrator, "send_prompts_async", new_callable=AsyncMock
    ) as mock_send_prompts_async:
        mock_send_prompts_async.return_value = [
            PromptRequestResponse(request_pieces=[PromptRequestPiece(role="system", original_value="response")])
        ]

        responses = await flip_attack_orchestrator.send_prompts_async(prompt_list=prompts)

        mock_send_prompts_async.assert_called_once_with(
            prompt_list=expected_prompts, prompt_type="text", memory_labels=None, metadata=None
        )
        assert len(responses) == 1
        assert responses[0].request_pieces[0].original_value == "response"


def test_init(flip_attack_orchestrator):
    assert isinstance(flip_attack_orchestrator._objective_target, PromptChatTarget)
    assert isinstance(flip_attack_orchestrator._memory, MemoryInterface)
    assert flip_attack_orchestrator._batch_size == 5
    assert flip_attack_orchestrator._verbose is True
    assert isinstance(flip_attack_orchestrator._prompt_converters[0], FlipConverter)


def test_default_systemprompt(flip_attack_orchestrator):
    assert flip_attack_orchestrator._prepended_conversation
    assert len(flip_attack_orchestrator._prepended_conversation) == 1
    assert flip_attack_orchestrator._prepended_conversation[0].request_pieces[0].role == "system"
    assert "flipping each word" in flip_attack_orchestrator._prepended_conversation[0].request_pieces[0].original_value
