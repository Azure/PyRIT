# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.exceptions.exception_classes import InvalidJsonException
from pyrit.models import PromptRequestPiece, Score, UnvalidatedScore
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import InsecureCodeScorer


@pytest.fixture
def mock_chat_target():
    return MagicMock(spec=PromptChatTarget)


@pytest.mark.asyncio
async def test_insecure_code_scorer_valid_response(mock_chat_target):
    # Initialize the scorer
    scorer = InsecureCodeScorer(
        chat_target=mock_chat_target,
        threshold=0.5,
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

    # Patch _memory.add_scores_to_memory to prevent duckdb errors and check for call
    with patch.object(scorer._memory, "add_scores_to_memory", new=MagicMock()) as mock_add_scores:
        with patch.object(scorer, "_score_value_with_llm", new=AsyncMock(return_value=unvalidated_score)):
            # Create a request_response object
            request_response = PromptRequestPiece(role="user", original_value="sample code")

            # Call the score_async method
            scores = await scorer.score_async(request_response)

            # Assertions
            assert len(scores) == 1
            assert isinstance(scores[0], Score)
            assert scores[0].score_value == "0.8"
            mock_add_scores.assert_called_once_with(scores=[scores[0]])


@pytest.mark.asyncio
async def test_insecure_code_scorer_invalid_json(mock_chat_target):
    # Initialize the scorer
    scorer = InsecureCodeScorer(
        chat_target=mock_chat_target,
        threshold=0.5,
    )

    # Patch scorer._memory.add_scores_to_memory to make it a mock
    with patch.object(scorer._memory, "add_scores_to_memory", new=MagicMock()) as mock_add_scores:
        # Mock _score_value_with_llm to raise InvalidJsonException
        with patch.object(
            scorer, "_score_value_with_llm", new=AsyncMock(side_effect=InvalidJsonException(message="Invalid JSON"))
        ):
            request_response = PromptRequestPiece(role="user", original_value="sample code")

            with pytest.raises(InvalidJsonException, match="Invalid JSON"):
                await scorer.score_async(request_response)

            # Ensure memory functions were not called
            mock_add_scores.assert_not_called()


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
