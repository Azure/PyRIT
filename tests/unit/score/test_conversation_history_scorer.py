# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.memory import CentralMemory
from pyrit.models import MessagePiece, Score, UnvalidatedScore
from pyrit.score import SelfAskGeneralFloatScaleScorer, create_conversation_scorer
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator


@pytest.mark.asyncio
async def test_conversation_history_scorer_score_async_success(patch_central_database):
    memory = CentralMemory.get_memory_instance()
    conversation_id = str(uuid.uuid4())

    message_pieces = [
        MessagePiece(
            role="user",
            original_value="Hello, I need help",
            conversation_id=conversation_id,
            sequence=1,
        ),
        MessagePiece(
            role="assistant",
            original_value="How can I help you?",
            conversation_id=conversation_id,
            sequence=2,
        ),
        MessagePiece(
            role="user",
            original_value="I'm feeling overwhelmingly sad",
            conversation_id=conversation_id,
            sequence=3,
        ),
        MessagePiece(
            role="assistant",
            original_value="Here's a joke to make you laugh instead",
            conversation_id=conversation_id,
            sequence=4,
        ),
    ]

    memory.add_message_pieces_to_memory(message_pieces=message_pieces)

    message = MagicMock()
    message.message_pieces = [message_pieces[-1]]  # Score the last message

    # Mock the underlying scorer
    mock_scorer = MagicMock(spec=SelfAskGeneralFloatScaleScorer)
    mock_scorer._validator = ScorerPromptValidator(supported_data_types=["text"])
    score = Score(
        score_value="0.25",
        score_value_description="Moderate concern",
        score_rationale="Valid rationale",
        score_metadata={"test": "metadata"},
        score_category=["test_harm"],
        scorer_class_identifier={"test": "test"},
        message_piece_id=message_pieces[-1].id or uuid.uuid4(),
        objective="test_objective",
        score_type="float_scale",
    )
    mock_scorer._score_async = AsyncMock(return_value=[score])

    scorer = create_conversation_scorer(scorer=mock_scorer)
    scores = await scorer.score_async(message)

    assert len(scores) == 1
    result_score = scores[0]
    assert result_score.score_value == "0.25"
    assert result_score.score_value_description == "Moderate concern"
    assert result_score.score_rationale == "Valid rationale"

    # Verify the underlying scorer was called with conversation history
    mock_scorer._score_async.assert_awaited_once()
    call_args = mock_scorer._score_async.call_args
    called_message = call_args[1]["message"]
    called_piece = called_message.message_pieces[0]

    # Verify the conversation text was built correctly
    expected_conversation = (
        "User: Hello, I need help\n"
        "Assistant: How can I help you?\n"
        "User: I'm feeling overwhelmingly sad\n"
        "Assistant: Here's a joke to make you laugh instead\n"
    )
    assert called_piece.original_value == expected_conversation
    assert called_piece.converted_value == expected_conversation


@pytest.mark.asyncio
async def test_conversation_history_scorer_conversation_not_found(patch_central_database):
    mock_scorer = MagicMock(spec=SelfAskGeneralFloatScaleScorer)
    mock_scorer._validator = ScorerPromptValidator(supported_data_types=["text"])
    scorer = create_conversation_scorer(scorer=mock_scorer)

    nonexistent_conversation_id = str(uuid.uuid4())
    message_piece = MessagePiece(
        role="assistant",
        original_value="Test response",
        conversation_id=nonexistent_conversation_id,
    )
    message = MagicMock()
    message.message_pieces = [message_piece]

    with pytest.raises(ValueError, match=f"Conversation with ID {nonexistent_conversation_id} not found in memory"):
        await scorer.score_async(message)


@pytest.mark.asyncio
async def test_conversation_history_scorer_chronological_ordering(patch_central_database):
    memory = CentralMemory.get_memory_instance()
    conversation_id = str(uuid.uuid4())

    # Create messages with explicit ordering
    message_pieces = [
        MessagePiece(
            role="user",
            original_value="First message",
            conversation_id=conversation_id,
            sequence=1,
        ),
        MessagePiece(
            role="user",
            original_value="Second message",
            conversation_id=conversation_id,
            sequence=2,
        ),
        MessagePiece(
            role="assistant",
            original_value="Third message",
            conversation_id=conversation_id,
            sequence=3,
        ),
    ]
    memory.add_message_pieces_to_memory(message_pieces=message_pieces)

    message = MagicMock()
    message.message_pieces = [message_pieces[0]]

    mock_scorer = MagicMock(spec=SelfAskGeneralFloatScaleScorer)
    mock_scorer._validator = ScorerPromptValidator(supported_data_types=["text"])
    score = Score(
        score_value="0.2",
        score_value_description="Low concern",
        score_rationale="Test rationale",
        score_metadata={},
        score_category=["test"],
        scorer_class_identifier={"test": "test"},
        message_piece_id=message_pieces[0].id or str(uuid.uuid4()),
        objective="test",
        score_type="float_scale",
    )
    mock_scorer._score_async = AsyncMock(return_value=[score])

    scorer = create_conversation_scorer(scorer=mock_scorer)

    await scorer.score_async(message)

    call_args = mock_scorer._score_async.call_args
    called_message = call_args[1]["message"]
    called_piece = called_message.message_pieces[0]

    expected_conversation = "User: First message\n" "User: Second message\n" "Assistant: Third message\n"
    assert called_piece.original_value == expected_conversation


@pytest.mark.asyncio
async def test_conversation_history_scorer_filters_roles_correctly(patch_central_database):
    memory = CentralMemory.get_memory_instance()
    conversation_id = str(uuid.uuid4())

    message_pieces = [
        MessagePiece(
            role="user",
            original_value="User message",
            conversation_id=conversation_id,
            sequence=1,
        ),
        MessagePiece(
            role="system",
            original_value="System message",
            conversation_id=conversation_id,
            sequence=2,
        ),
        MessagePiece(
            role="assistant",
            original_value="Assistant message",
            conversation_id=conversation_id,
            sequence=3,
        ),
    ]

    memory.add_message_pieces_to_memory(message_pieces=message_pieces)

    message = MagicMock()
    message.message_pieces = [message_pieces[0]]

    mock_scorer = MagicMock(spec=SelfAskGeneralFloatScaleScorer)
    mock_scorer._validator = ScorerPromptValidator(supported_data_types=["text"])
    score = Score(
        score_value="0.4",
        score_value_description="Test",
        score_rationale="Test rationale",
        score_metadata={},
        score_category=["test"],
        scorer_class_identifier={"test": "test"},
        message_piece_id=message_pieces[0].id or str(uuid.uuid4()),
        objective="test",
        score_type="float_scale",
    )
    mock_scorer._score_async = AsyncMock(return_value=[score])

    scorer = create_conversation_scorer(scorer=mock_scorer)
    await scorer.score_async(message)

    call_args = mock_scorer._score_async.call_args
    called_message = call_args[1]["message"]
    called_piece = called_message.message_pieces[0]

    expected_conversation = "User: User message\n" "Assistant: Assistant message\n"
    assert called_piece.original_value == expected_conversation
    assert "System message" not in called_piece.original_value


@pytest.mark.asyncio
async def test_conversation_history_scorer_score_text_async_delegates(patch_central_database):
    mock_scorer = MagicMock(spec=SelfAskGeneralFloatScaleScorer)
    mock_scorer._validator = ScorerPromptValidator(supported_data_types=["text"])
    unvalidated_score = UnvalidatedScore(
        raw_score_value="0.8",
        score_value_description="Test score",
        score_rationale="Test rationale",
        score_metadata={},
        score_category=["test"],
        scorer_class_identifier={"test": "test"},
        message_piece_id=uuid.uuid4(),
        objective="test_objective",
    )
    mock_scorer.score_text_async = AsyncMock(return_value=[unvalidated_score])

    scorer = create_conversation_scorer(scorer=mock_scorer)
    test_text = "Test text to score"
    test_objective = "Test objective"

    # ConversationScorer requires a conversation in memory, so calling score_text_async
    # without a pre-existing conversation should raise a ValueError
    with pytest.raises(ValueError, match="Conversation with ID .* not found in memory"):
        await scorer.score_text_async(text=test_text, objective=test_objective)


@pytest.mark.asyncio
async def test_conversation_history_scorer_preserves_metadata(patch_central_database):
    memory = CentralMemory.get_memory_instance()
    conversation_id = str(uuid.uuid4())

    message_piece = MessagePiece(
        role="assistant",
        original_value="Response",
        conversation_id=conversation_id,
        labels={"test": "label"},
        prompt_target_identifier={"target": "test"},
        attack_identifier={"attack": "test"},
        sequence=1,
    )

    memory.add_message_pieces_to_memory(message_pieces=[message_piece])

    message = MagicMock()
    message.message_pieces = [message_piece]

    mock_scorer = MagicMock(spec=SelfAskGeneralFloatScaleScorer)
    mock_scorer._validator = ScorerPromptValidator(supported_data_types=["text"])
    score = Score(
        score_value="0.2",
        score_value_description="Test",
        score_rationale="Test rationale",
        score_metadata={},
        score_category=["test"],
        scorer_class_identifier={"test": "test"},
        message_piece_id=message_piece.id or str(uuid.uuid4()),
        objective="test",
        score_type="float_scale",
    )
    mock_scorer._score_async = AsyncMock(return_value=[score])

    scorer = create_conversation_scorer(scorer=mock_scorer)

    await scorer.score_async(message)

    call_args = mock_scorer._score_async.call_args
    called_message = call_args[1]["message"]
    called_piece = called_message.message_pieces[0]

    assert called_piece.id == message_piece.id
    assert called_piece.conversation_id == message_piece.conversation_id
    assert called_piece.labels == message_piece.labels
    assert called_piece.prompt_target_identifier == message_piece.prompt_target_identifier
    assert called_piece.attack_identifier == message_piece.attack_identifier
