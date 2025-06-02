# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.models import PromptRequestPiece, PromptRequestResponse, Score
from pyrit.orchestrator.multi_turn.multi_turn_orchestrator import OrchestratorResult
from pyrit.orchestrator.multi_turn.red_teaming_orchestrator import (
    RedTeamingOrchestrator,
)
from pyrit.prompt_target import PromptChatTarget


@pytest.fixture
def orchestrator(patch_central_database):
    objective_scorer = MagicMock()
    objective_scorer.scorer_type = "true_false"
    objective_target = MagicMock(PromptChatTarget)
    orchestrator = RedTeamingOrchestrator(
        objective_target=objective_target,
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
    assert num_turns == 1
    orchestrator._memory.add_request_response_to_memory.assert_not_called()


def test_prepare_conversation_with_prepended_conversation(orchestrator):
    system_piece_id = uuid.uuid4()
    orchestrator._prepended_conversation = [
        PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    id=system_piece_id,
                    role="system",
                    original_value="Hello I am a system prompt",
                )
            ]
        ),
        PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    id=uuid.uuid4(),
                    role="user",
                    original_value="Hello I am a user prompt",
                )
            ]
        ),
        PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    id=uuid.uuid4(),
                    role="assistant",
                    original_value="Hello I am an assistant prompt",
                )
            ]
        ),
    ]

    system_piece = orchestrator._prepended_conversation[0].request_pieces[0]
    user_piece = orchestrator._prepended_conversation[1].request_pieces[0]
    assistant_piece = orchestrator._prepended_conversation[2].request_pieces[0]

    mocked_score = MagicMock()
    orchestrator._memory.get_scores_by_prompt_ids.return_value = [mocked_score]

    new_conversation_id = str(uuid.uuid4())
    turn_num = orchestrator._prepare_conversation(new_conversation_id=new_conversation_id)

    # Check individual piece components are set correctly, using system piece as an example
    assert turn_num == 2
    assert system_piece.conversation_id == new_conversation_id
    assert system_piece.id != system_piece_id
    assert system_piece.original_prompt_id == system_piece_id
    assert system_piece.orchestrator_identifier == orchestrator.get_identifier()

    # Assert system prompt set
    orchestrator._objective_target.set_system_prompt.assert_called_once()

    # Assert calls to memory for user and assistant messages
    for request in orchestrator._prepended_conversation[-1:]:
        orchestrator._memory.add_request_response_to_memory.assert_any_call(request=request)

    # Check globals are set correctly
    assert orchestrator._last_prepended_user_message == user_piece.converted_value
    orchestrator._memory.get_scores_by_prompt_ids.assert_called_once_with(
        prompt_request_response_ids=[str(assistant_piece.original_prompt_id)]
    )
    assert orchestrator._last_prepended_assistant_message_scores == [mocked_score]


def test_prepare_conversation_with_prepended_conversation_no_assistant_scores(orchestrator):
    orchestrator._prepended_conversation = [
        PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    id=uuid.uuid4(),
                    role="user",
                    original_value="Hello I am a user prompt",
                )
            ]
        ),
        PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    id=uuid.uuid4(),
                    role="assistant",
                    original_value="Hello I am an assistant prompt",
                )
            ]
        ),
    ]

    assistant_piece = orchestrator._prepended_conversation[1].request_pieces[0]
    orchestrator._memory.get_scores_by_prompt_ids.return_value = []
    orchestrator._prepare_conversation(new_conversation_id=str(uuid.uuid4()))

    # Check globals are set correctly
    # Last prepended user message should be none as it is not needed when no scores are returned
    assert orchestrator._last_prepended_user_message == ""
    orchestrator._memory.get_scores_by_prompt_ids.assert_called_once_with(
        prompt_request_response_ids=[str(assistant_piece.original_prompt_id)]
    )
    assert orchestrator._last_prepended_assistant_message_scores == []


def test_prepare_conversation_with_invalid_prepended_conversation(orchestrator):
    # Invalid conversation as it is missing a user message before the assistant message
    orchestrator._prepended_conversation = [
        PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    id=uuid.uuid4(),
                    role="system",
                    original_value="Hello I am a system prompt",
                )
            ]
        ),
        PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    id=uuid.uuid4(),
                    role="assistant",
                    original_value="Hello I am an assistant prompt",
                )
            ]
        ),
    ]

    with pytest.raises(
        ValueError, match="There must be a user message preceding the assistant message in prepended conversations."
    ):
        orchestrator._prepare_conversation(new_conversation_id=str(uuid.uuid4()))


def test_prepare_conversation_with_custom_user_message(orchestrator):
    orchestrator._prepended_conversation = [
        PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    id=uuid.uuid4(),
                    role="user",
                    original_value="Hello I am a user prompt 1",
                )
            ]
        ),
        PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    id=uuid.uuid4(),
                    role="assistant",
                    original_value="Hello I am an assistant prompt",
                )
            ]
        ),
        PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    id=uuid.uuid4(),
                    role="user",
                    original_value="Hello I am a user prompt 2",
                )
            ]
        ),
    ]

    turn_num = orchestrator._prepare_conversation(new_conversation_id=str(uuid.uuid4()))

    # There is one complete turn in the prepended conversation, so turn_num should be 2
    # as this is mid-conversation
    assert turn_num == 2

    # Assert calls to memory
    for request in orchestrator._prepended_conversation[:-1]:
        orchestrator._memory.add_request_response_to_memory.assert_any_call(request=request)

    # Verify the last user message is not added to memory as part of preparing the conversation
    calls = orchestrator._memory.add_request_response_to_memory.mock_calls
    assert len(calls) == 2
    assert calls[-1].kwargs["request"] == orchestrator._prepended_conversation[1]

    # Check globals are set correctly
    user_piece = orchestrator._prepended_conversation[2].request_pieces[0]
    assert orchestrator._last_prepended_user_message == user_piece.converted_value
    assert orchestrator._last_prepended_assistant_message_scores == []


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
        result = OrchestratorResult("conversation_id_123", "failure", "Test Objective")

    await result.print_conversation_async()

    mock_memory.get_conversation.assert_called_once_with(conversation_id="conversation_id_123")
    mock_memory.get_scores_by_prompt_ids.assert_not_called()


@pytest.mark.parametrize("include_auxiliary_scores, get_scores_by_prompt_id_call_count", [(False, 0), (True, 2)])
@pytest.mark.asyncio
async def test_print_conversation_with_messages(include_auxiliary_scores, get_scores_by_prompt_id_call_count):

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
            "pyrit.orchestrator.models.orchestrator_result.display_image_response", new_callable=AsyncMock
        ) as mock_display_image_response:
            result = OrchestratorResult(
                conversation_id="conversation_id_123",
                objective="Test Objective",
                status="success",
                objective_score=score,
            )

            await result.print_conversation_async(include_auxiliary_scores=include_auxiliary_scores)

            mock_memory.get_conversation.assert_called_once_with(conversation_id="conversation_id_123")
            assert mock_display_image_response.call_count == 2
            assert mock_memory.get_scores_by_prompt_ids.call_count == get_scores_by_prompt_id_call_count
