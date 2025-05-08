# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import uuid
from unittest.mock import MagicMock, patch

import pytest
from unit.mocks import get_image_request_piece

from pyrit.memory.central_memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import PromptRequestPiece
from pyrit.score.question_answer_scorer import QuestionAnswerScorer


@pytest.fixture
def image_request_piece() -> PromptRequestPiece:
    return get_image_request_piece()


@pytest.fixture
def text_request_piece(patch_central_database) -> PromptRequestPiece:
    piece = PromptRequestPiece(
        id=uuid.uuid4(),
        role="user",
        original_value="test content",
        original_value_data_type="text",
        prompt_metadata={"correct_answer_index": "0", "correct_answer": "Paris"},
    )
    piece.id = None
    return piece


@pytest.mark.asyncio
async def test_question_answer_scorer_validate_image(image_request_piece: PromptRequestPiece):
    scorer = QuestionAnswerScorer(category="new_category")
    with pytest.raises(ValueError, match="Question Answer Scorer only supports text data type"):
        await scorer.score_async(image_request_piece)

    os.remove(image_request_piece.converted_value)


@pytest.mark.asyncio
async def test_question_answer_scorer_validate_missing_metadata():
    request_piece = PromptRequestPiece(
        id="test_id",
        role="user",
        original_value="test content",
        converted_value="test response",
        converted_value_data_type="text",
        prompt_metadata={},
    )
    scorer = QuestionAnswerScorer(category="new_category")
    with pytest.raises(
        ValueError, match="Question Answer Scorer requires metadata with either correct_answer_index or correct_answer"
    ):
        await scorer.score_async(request_piece)


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
    response: str, expected_score: bool, text_request_piece: PromptRequestPiece
):
    text_request_piece.converted_value = response
    scorer = QuestionAnswerScorer(category="new_category")

    score = await scorer.score_async(text_request_piece)

    assert len(score) == 1
    assert score[0].score_value == str(expected_score)
    assert score[0].score_type == "true_false"
    assert score[0].score_category == "new_category"


@pytest.mark.asyncio
async def test_question_answer_scorer_adds_to_memory():
    memory = MagicMock(MemoryInterface)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = QuestionAnswerScorer(category="new_category")
        request_piece = PromptRequestPiece(
            id="test_id",
            role="user",
            original_value="test content",
            converted_value="0: Paris",
            converted_value_data_type="text",
            prompt_metadata={"correct_answer_index": "0", "correct_answer": "Paris"},
        )
        await scorer.score_async(request_piece)

        memory.add_scores_to_memory.assert_called_once()


@pytest.mark.asyncio
async def test_question_answer_scorer_no_category():
    memory = MagicMock(MemoryInterface)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = QuestionAnswerScorer()
        request_piece = PromptRequestPiece(
            id="test_id",
            role="user",
            original_value="test content",
            converted_value="0: Paris",
            converted_value_data_type="text",
            prompt_metadata={"correct_answer_index": "0", "correct_answer": "Paris"},
        )
        await scorer.score_async(request_piece)

        memory.add_scores_to_memory.assert_called_once()
