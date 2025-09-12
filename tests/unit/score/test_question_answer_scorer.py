# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import uuid
from unittest.mock import MagicMock, patch

import pytest
from unit.mocks import get_image_request_piece

from pyrit.memory.central_memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.score import QuestionAnswerScorer


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

    scorer = QuestionAnswerScorer(category=["new_category"])
    request_response = PromptRequestResponse(request_pieces=[image_request_piece])
    with pytest.raises(ValueError, match="There are no valid pieces to score."):
        await scorer.score_async(request_response)

    os.remove(image_request_piece.converted_value)


@pytest.mark.asyncio
async def test_question_answer_scorer_validate_missing_metadata():
    request = PromptRequestPiece(
        id="test_id",
        role="user",
        original_value="test content",
        converted_value="test response",
        converted_value_data_type="text",
        prompt_metadata={},
    ).to_prompt_request_response()
    scorer = QuestionAnswerScorer(category=["new_category"])
    with pytest.raises(
        ValueError, match="There are no valid pieces to score."
    ):
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
    response: str, expected_score: bool, text_request_piece: PromptRequestPiece
):
    text_request_piece.converted_value = response
    scorer = QuestionAnswerScorer(category=["new_category"])
    request_response = PromptRequestResponse(request_pieces=[text_request_piece])

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
        request_response = PromptRequestPiece(
            id="test_id",
            role="user",
            original_value="test content",
            converted_value="0: Paris",
            converted_value_data_type="text",
            prompt_metadata={"correct_answer_index": "0", "correct_answer": "Paris"},
        ).to_prompt_request_response()

        await scorer.score_async(request_response)

        memory.add_scores_to_memory.assert_called_once()


@pytest.mark.asyncio
async def test_question_answer_scorer_no_category():
    memory = MagicMock(MemoryInterface)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = QuestionAnswerScorer()
        request_response = PromptRequestPiece(
            id="test_id",
            role="user",
            original_value="test content",
            converted_value="0: Paris",
            converted_value_data_type="text",
            prompt_metadata={"correct_answer_index": "0", "correct_answer": "Paris"},
        ).to_prompt_request_response()
        await scorer.score_async(request_response)

        memory.add_scores_to_memory.assert_called_once()
