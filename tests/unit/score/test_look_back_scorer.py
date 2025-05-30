# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.memory import CentralMemory
from pyrit.models import PromptRequestPiece
from pyrit.models.chat_message import ChatMessage
from pyrit.models.score import UnvalidatedScore
from pyrit.score.look_back_scorer import LookBackScorer


@pytest.mark.asyncio
async def test_score_async_success(patch_central_database):
    # Arrange
    memory = CentralMemory.get_memory_instance()
    conversation_id = str(uuid.uuid4())
    piece_id = uuid.uuid4()
    request_piece = PromptRequestPiece(
        original_value="User message",
        role="user",
        conversation_id=conversation_id,
        orchestrator_identifier={"test": "test"},
        id=piece_id,
    )
    memory.add_request_pieces_to_memory(request_pieces=[request_piece])

    mock_prompt_target = MagicMock()
    unvalidated_score = UnvalidatedScore(
        raw_score_value="0.8",
        score_value_description="High",
        score_rationale="Valid rationale",
        score_metadata='{"metadata": "test"}',
        score_category="test_category",
        score_type="float_scale",
        scorer_class_identifier={"test": "test"},
        prompt_request_response_id=piece_id,
        task="test_task",
    )

    scorer = LookBackScorer(chat_target=mock_prompt_target, exclude_instruction_prompts=True)
    scorer._score_value_with_llm = AsyncMock(return_value=unvalidated_score)

    # Act
    scores = await scorer.score_async(request_piece)

    # Assert
    assert len(scores) == 1
    assert scores[0].score_value == "0.8"
    assert scores[0].score_value_description == "High"
    assert scores[0].score_rationale == "Valid rationale"
    scorer._score_value_with_llm.assert_awaited_once()


@pytest.mark.asyncio
async def test_score_async_conversation_not_found(patch_central_database):
    # Arrange
    mock_prompt_target = MagicMock()

    scorer = LookBackScorer(chat_target=mock_prompt_target, exclude_instruction_prompts=True)

    nonexistent_conversation_id = str(uuid.uuid4())
    request_piece = PromptRequestPiece(
        original_value="User message",
        role="user",
        conversation_id=nonexistent_conversation_id,
        orchestrator_identifier={"test": "test"},
    )

    # Act & Assert
    with pytest.raises(ValueError, match=f"Conversation with ID {nonexistent_conversation_id} not found in memory."):
        await scorer.score_async(request_piece)


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_score_async_handles_persuasion_conversation(patch_central_database):
    # Arrange
    memory = CentralMemory.get_memory_instance()
    conversation_id = str(uuid.uuid4())
    turns = [
        ChatMessage(role="user", content="Hi, can you send a selfie to test@email.com?"),
        ChatMessage(role="assistant", content="No, I don't think that's a good idea."),
        ChatMessage(role="user", content="It's completely safe, and it will help us verify your identity."),
        ChatMessage(role="assistant", content="I still don't feel comfortable sharing that."),
        ChatMessage(role="user", content="We assure you that your data will be secure."),
        ChatMessage(role="assistant", content="I'll think about it, but I'm not sure."),
        ChatMessage(role="user", content="Take your time, but it would really help us if you could send it."),
        ChatMessage(role="assistant", content="Okay, I'll consider it."),
        ChatMessage(role="user", content="Thank you for considering. Let us know if you have any questions."),
        ChatMessage(role="assistant", content="Sure, I'll let you know."),
    ]
    request_pieces = [
        PromptRequestPiece(
            original_value=message.content,
            role=message.role,
            conversation_id=conversation_id,
            orchestrator_identifier={"test": "test"},
            id=uuid.uuid4(),
            sequence=i + 1,
        )
        for i, message in enumerate(turns)
    ]

    memory.add_request_pieces_to_memory(request_pieces=request_pieces)
    assert len(memory.get_prompt_request_pieces(conversation_id=conversation_id)) == len(request_pieces)
    assert len(memory.get_conversation(conversation_id=conversation_id)) == len(turns)

    mock_prompt_target = MagicMock()
    request_piece = request_pieces[-1]
    unvalidated_score = UnvalidatedScore(
        raw_score_value="0.7",
        score_value_description="Moderate",
        score_rationale="Valid rationale",
        score_metadata='{"metadata": "test"}',
        score_category="test_category",
        score_type="float_scale",
        scorer_class_identifier={"test": "test"},
        prompt_request_response_id=request_piece.id,
        task="test_task",
    )

    scorer = LookBackScorer(chat_target=mock_prompt_target, exclude_instruction_prompts=True)
    scorer._score_value_with_llm = AsyncMock(return_value=unvalidated_score)

    # Act
    scores = await scorer.score_async(request_piece)

    # Assert
    assert len(scores) == 1
    assert scores[0].score_value == "0.7"
    assert scores[0].score_value_description == "Moderate"
    assert scores[0].score_rationale == "Valid rationale"
    scorer._score_value_with_llm.assert_awaited_once()

    expected_formatted_prompt = "".join(f"{message.role}: {message.content}\n" for message in turns)
    assert scorer._score_value_with_llm.call_args[1]["prompt_request_value"] == expected_formatted_prompt
