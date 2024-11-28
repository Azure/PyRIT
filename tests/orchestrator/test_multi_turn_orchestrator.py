# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pyrit.models import PromptRequestResponse, PromptRequestPiece, Score
from pyrit.orchestrator.multi_turn.multi_turn_orchestrator import MultiTurnAttackResult


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
