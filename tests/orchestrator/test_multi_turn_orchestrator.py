# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import uuid

from unittest.mock import AsyncMock, MagicMock, patch

from pyrit.models import PromptRequestResponse, PromptRequestPiece, Score
from pyrit.orchestrator.multi_turn.multi_turn_orchestrator import MultiTurnAttackResult
from pyrit.orchestrator.multi_turn.red_teaming_orchestrator import RedTeamingOrchestrator


@pytest.fixture
def orchestrator():
    objective_scorer = MagicMock()
    objective_scorer.scorer_type = "true_false"
    orchestrator = RedTeamingOrchestrator(
        objective_target=MagicMock(),
        adversarial_chat=MagicMock(),
        max_turns=2,
        objective_scorer=objective_scorer,
    )

    orchestrator._memory = MagicMock()
    return orchestrator


def test_prepare_conversation_no_prepended_conversation(orchestrator):
    orchestrator._prepended_conversation = None
    new_conversation_id = str(uuid.uuid4())
    num_turns = orchestrator._prepare_conversation(new_conversation_id=new_conversation_id)
    assert num_turns == 0
    orchestrator._memory.add_request_response_to_memory.assert_not_called()


def test_prepare_conversation_with_prepended_conversation(orchestrator):
    request_piece = MagicMock()
    request_piece.role = "assistant"
    id = uuid.uuid4()
    request_piece.id = id
    request = MagicMock()
    request.request_pieces = [request_piece]

    orchestrator._prepended_conversation = [request]
    new_conversation_id = str(uuid.uuid4())
    num_turns = orchestrator._prepare_conversation(new_conversation_id=new_conversation_id)

    assert num_turns == 1
    assert request_piece.conversation_id == new_conversation_id
    assert request_piece.id != id
    orchestrator._memory.add_request_response_to_memory.assert_called_once_with(request=request)


def test_prepare_conversation_exceeds_max_turns(orchestrator):
    request_piece = MagicMock()
    request_piece.role = "assistant"
    request = MagicMock()
    request.request_pieces = [request_piece]
    orchestrator._prepended_conversation = [request] * (orchestrator._max_turns)
    new_conversation_id = str(uuid.uuid4())

    with pytest.raises(ValueError):
        orchestrator._prepare_conversation(new_conversation_id=new_conversation_id)


@pytest.mark.asyncio
async def test_print_conversation_no_messages():
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.get_scores_by_prompt_ids.return_value = []
    with patch("pyrit.memory.CentralMemory.get_memory_instance", return_value=mock_memory):
        result = MultiTurnAttackResult("conversation_id_123", False, "Test Objective")

    await result.print_conversation_async()

    mock_memory.get_conversation.assert_called_once_with(conversation_id="conversation_id_123")
    mock_memory.get_scores_by_prompt_ids.assert_not_called()


@pytest.mark.asyncio
async def test_print_conversation_with_messages():

    id_1 = uuid.uuid4()

    message_1 = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                original_value="Hello I am prompt 1",
                original_value_data_type="text",
                sequence=0,
                role="user",
                conversation_id="conversation_id_123",
                id=id_1,
            )
        ]
    )

    message_2 = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                original_value="Hello I am prompt 2",
                original_value_data_type="text",
                sequence=1,
                role="assistant",
                conversation_id="conversation_id_123",
            )
        ]
    )

    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = [message_1, message_2]

    score = Score(
        score_value=0.9,
        score_type="float_scale",
        score_value_description="description",
        score_category="category",
        score_rationale="rationale",
        score_metadata="",
        prompt_request_response_id="123",
    )

    mock_memory.get_scores_by_prompt_ids.return_value = [score]

    with patch("pyrit.memory.CentralMemory.get_memory_instance", return_value=mock_memory):
        with patch(
            "pyrit.orchestrator.multi_turn.multi_turn_orchestrator.display_image_response", new_callable=AsyncMock
        ) as mock_display_image_response:
            result = MultiTurnAttackResult("conversation_id_123", True, "Test Objective")

            await result.print_conversation_async()

            mock_memory.get_conversation.assert_called_once_with(conversation_id="conversation_id_123")
            assert mock_display_image_response.call_count == 2
            assert mock_memory.get_scores_by_prompt_ids.call_count == 2
