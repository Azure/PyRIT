# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.memory import MemoryInterface
from pyrit.models import (
    SeedPromptGroup,
)
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
async def test_run_attack_async(flip_attack_orchestrator):
    """
    Tests that run_attack_async properly formats the prompt and calls the parent class method
    """
    objective = "How can I kill a Python process?"
    expected_prompt = "TASK is '⟪How can I kill a Python process?⟫'"

    with patch.object(PromptSendingOrchestrator, "run_attack_async", new_callable=AsyncMock) as mock_run_attack_async:
        mock_run_attack_async.return_value = MagicMock()

        await flip_attack_orchestrator.run_attack_async(objective=objective)

        # Verify the call to parent class method
        mock_run_attack_async.assert_called_once()
        call_kwargs = mock_run_attack_async.call_args.kwargs

        # Check the seed prompt group
        seed_prompt = call_kwargs["seed_prompt"]
        assert isinstance(seed_prompt, SeedPromptGroup)
        assert len(seed_prompt.prompts) == 1
        assert seed_prompt.prompts[0].value == expected_prompt
        assert seed_prompt.prompts[0].data_type == "text"

        # Check the prepended conversation
        prepended = call_kwargs["prepended_conversation"]
        assert len(prepended) == 1
        assert prepended[0].request_pieces[0].role == "system"
        assert "flipping each word" in prepended[0].request_pieces[0].original_value


def test_init(flip_attack_orchestrator):
    """
    Tests that the orchestrator is initialized with the correct configuration
    """
    assert isinstance(flip_attack_orchestrator._objective_target, PromptChatTarget)
    assert isinstance(flip_attack_orchestrator._memory, MemoryInterface)
    assert flip_attack_orchestrator._batch_size == 5
    assert flip_attack_orchestrator._verbose is True

    # Check that the flip converter is the first converter in the configuration
    assert len(flip_attack_orchestrator._request_converter_configurations) == 1
    assert isinstance(flip_attack_orchestrator._request_converter_configurations[0].converters[0], FlipConverter)


def test_system_prompt(flip_attack_orchestrator):
    """
    Tests that the system prompt is properly loaded and formatted
    """
    assert flip_attack_orchestrator._system_prompt
    assert len(flip_attack_orchestrator._system_prompt.request_pieces) == 1
    assert flip_attack_orchestrator._system_prompt.request_pieces[0].role == "system"
    assert "flipping each word" in flip_attack_orchestrator._system_prompt.request_pieces[0].original_value
