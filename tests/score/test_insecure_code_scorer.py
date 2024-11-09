# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pyrit.models import PromptRequestPiece, Score
from pyrit.exceptions.exception_classes import InvalidJsonException
from pyrit.prompt_target import PromptChatTarget
from pyrit.memory import MemoryInterface
from pyrit.score import InsecureCodeScorer, UnvalidatedScore


@pytest.fixture
def mock_chat_target():
    return MagicMock(spec=PromptChatTarget)


@pytest.fixture
def mock_memory():
    return MagicMock(spec=MemoryInterface)


@pytest.mark.asyncio
async def test_insecure_code_scorer_valid_response(mock_chat_target, mock_memory):
    # Initialize the scorer
    scorer = InsecureCodeScorer(
        chat_target=mock_chat_target,
        threshold=0.5,
        memory=mock_memory,
    )

    # Create an UnvalidatedScore mock response
    unvalidated_score = UnvalidatedScore(
        raw_score_value="0.8",
        score_value_description="A valid score",
        score_type="float_scale",
        score_category="security",
        score_rationale="This code has potential vulnerabilities.",
        score_metadata=None,
        scorer_class_identifier={"scorer_name": "InsecureCodeScorer"},
        prompt_request_response_id="12345",
        task=None,
    )

    with patch.object(scorer, "_score_value_with_llm", new=AsyncMock(return_value=unvalidated_score)):
        # Create a request_response object
        request_response = PromptRequestPiece(role="user", original_value="sample code")

        # Call the score_async method
        scores = await scorer.score_async(request_response)

        # Assertions
        assert len(scores) == 1
        assert isinstance(scores[0], Score)
        assert scores[0].score_value == "0.8"
        assert mock_memory.add_request_pieces_to_memory.called
        assert mock_memory.add_scores_to_memory.called


@pytest.mark.asyncio
async def test_insecure_code_scorer_invalid_json(mock_chat_target, mock_memory):
    # Initialize the scorer
    scorer = InsecureCodeScorer(
        chat_target=mock_chat_target,
        threshold=0.5,
        memory=mock_memory,
    )

    # Mock _score_value_with_llm to raise InvalidJsonException
    with patch.object(
        scorer, "_score_value_with_llm", new=AsyncMock(side_effect=InvalidJsonException(message="Invalid JSON"))
    ):
        request_response = PromptRequestPiece(role="user", original_value="sample code")

        with pytest.raises(InvalidJsonException, match="Invalid JSON"):
            await scorer.score_async(request_response)

        # Ensure memory functions were not called
        assert not mock_memory.add_request_pieces_to_memory.called
        assert not mock_memory.add_scores_to_memory.called


def test_insecure_code_scorer_validate():
    scorer = InsecureCodeScorer(
        chat_target=MagicMock(spec=PromptChatTarget),
        threshold=0.5,
    )

    # Valid input
    valid_request_response = PromptRequestPiece(role="user", original_value="sample code")
    scorer.validate(valid_request_response)  # Should not raise any exceptions

    # Invalid input
    invalid_request_response = PromptRequestPiece(role="user", original_value="")
    with pytest.raises(ValueError, match="The code snippet must not be empty."):
        scorer.validate(invalid_request_response)
