# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import uuid
from unittest.mock import MagicMock, patch

import pytest
from unit.mocks import get_image_message_piece

from pyrit.memory.central_memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import MessagePiece, Message
from pyrit.score import QuestionAnswerScorer


@pytest.fixture
def image_message_piece() -> MessagePiece:
    return get_image_message_piece()


@pytest.fixture
def text_message_piece(patch_central_database) -> MessagePiece:
    piece = MessagePiece(
        id=uuid.uuid4(),
        role="user",
        original_value="test content",
        original_value_data_type="text",
        prompt_metadata={"correct_answer_index": "0", "correct_answer": "Paris"},
    )
    piece.id = None
    return piece


@pytest.mark.asyncio
async def test_question_answer_scorer_validate_image(image_message_piece: MessagePiece):

    scorer = QuestionAnswerScorer(category=["new_category"])
    request_response = Message(message_pieces=[image_message_piece])
    with pytest.raises(ValueError, match="There are no valid pieces to score."):
        await scorer.score_async(request_response)

    os.remove(image_message_piece.converted_value)


@pytest.mark.asyncio
async def test_question_answer_scorer_validate_missing_metadata():
    request = MessagePiece(
        id="test_id",
        role="user",
        original_value="test content",
        converted_value="test response",
        converted_value_data_type="text",
        prompt_metadata={},
    ).to_message()
    scorer = QuestionAnswerScorer(category=["new_category"])
    with pytest.raises(ValueError, match="There are no valid pieces to score."):
        await scorer.score_async(request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response,expected_score",
    [
        ("0: Paris", True),
        ("Paris", True),
        ("1: London", False),
        ("London", False),
        ("The answer is 0: Paris", True),
        ("The answer is Paris", True),
    ],
)
async def test_question_answer_scorer_score(
    response: str, expected_score: bool, text_message_piece: MessagePiece
):
    text_message_piece.converted_value = response
    scorer = QuestionAnswerScorer(category=["new_category"])
    request_response = Message(message_pieces=[text_message_piece])

    scores = await scorer.score_async(request_response)

    assert len(scores) == 1
    result_score = scores[0]
    assert result_score.get_value() == expected_score
    assert result_score.score_type == "true_false"
    assert result_score.score_category == ["new_category"]


@pytest.mark.asyncio
async def test_question_answer_scorer_adds_to_memory():
    memory = MagicMock(MemoryInterface)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = QuestionAnswerScorer(category=["new_category"])
        request_response = MessagePiece(
            id="test_id",
            role="user",
            original_value="test content",
            converted_value="0: Paris",
            converted_value_data_type="text",
            prompt_metadata={"correct_answer_index": "0", "correct_answer": "Paris"},
        ).to_message()

        await scorer.score_async(request_response)

        memory.add_scores_to_memory.assert_called_once()


@pytest.mark.asyncio
async def test_question_answer_scorer_no_category():
    memory = MagicMock(MemoryInterface)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = QuestionAnswerScorer()
        request_response = MessagePiece(
            id="test_id",
            role="user",
            original_value="test content",
            converted_value="0: Paris",
            converted_value_data_type="text",
            prompt_metadata={"correct_answer_index": "0", "correct_answer": "Paris"},
        ).to_message()
        await scorer.score_async(request_response)

        memory.add_scores_to_memory.assert_called_once()
