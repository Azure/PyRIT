# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pyrit.orchestrator.flip_attack_orchestrator import FlipAttackOrchestrator
from pyrit.orchestrator.prompt_sending_orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import PromptTarget
from pyrit.memory import MemoryInterface
from pyrit.memory import DuckDBMemory, CentralMemory
from pyrit.models import PromptRequestResponse, PromptRequestPiece
from pyrit.prompt_converter.flip_converter import FlipConverter


@pytest.fixture
def mock_prompt_target():
    return MagicMock(spec=PromptTarget)


@pytest.fixture
def mock_central_memory_instance():
    """Fixture to mock CentralMemory.get_memory_instance"""
    duckdb_in_memory = DuckDBMemory(db_path=":memory:")
    with patch.object(CentralMemory, "get_memory_instance", return_value=duckdb_in_memory) as duck_db_memory:
        yield duck_db_memory


@pytest.fixture
def flip_attack_orchestrator(mock_prompt_target, mock_central_memory_instance):
    return FlipAttackOrchestrator(prompt_target=mock_prompt_target, batch_size=5, verbose=True)


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
    assert isinstance(flip_attack_orchestrator._prompt_target, PromptTarget)
    assert isinstance(flip_attack_orchestrator._memory, MemoryInterface)
    assert flip_attack_orchestrator._batch_size == 5
    assert flip_attack_orchestrator._verbose is True
    assert isinstance(flip_attack_orchestrator._prompt_converters[0], FlipConverter)


def test_default_systemprompt(flip_attack_orchestrator):
    assert flip_attack_orchestrator._prepended_conversation
    assert len(flip_attack_orchestrator._prepended_conversation) == 1
    assert flip_attack_orchestrator._prepended_conversation[0].request_pieces[0].role == "system"
    assert "flipping each word" in flip_attack_orchestrator._prepended_conversation[0].request_pieces[0].original_value
