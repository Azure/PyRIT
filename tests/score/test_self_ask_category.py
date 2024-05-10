# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from textwrap import dedent
from typing import Generator
from unittest.mock import AsyncMock, MagicMock

from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.score.self_ask_category_scorer import ContentClassifierPaths, SelfAskCategoryScorer

from tests.mocks import get_memory_interface


@pytest.fixture
def scorer_category_response_bullying() -> PromptRequestResponse:

    json_response = (
        dedent(
            """
        {"category_name": "bullying", "category_description": "This is bullying.", "rationale": "The message
        seems like it contains bullying."}
        """
        )
        .strip()
        .replace("\n", " ")
    )

    return PromptRequestResponse(request_pieces=[PromptRequestPiece(role="assistant", original_value=json_response)])


@pytest.fixture
def scorer_category_response_false() -> PromptRequestResponse:

    json_response = (
        dedent(
            """
        {"category_name": "no_harm", "category_description": "No harms", "rationale": "The message
        contains no harms."}
        """
        )
        .strip()
        .replace("\n", " ")
    )

    return PromptRequestResponse(request_pieces=[PromptRequestPiece(role="assistant", original_value=json_response)])


@pytest.fixture
def memory() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


def test_category_scorer_set_system_prompt():
    chat_target = MagicMock()

    scorer = SelfAskCategoryScorer(
        chat_target=chat_target,
        content_classifier=ContentClassifierPaths.HARMFUL_CONTENT_CLASSIFIER.value,
    )

    chat_target.set_system_prompt.assert_called_once()

    # assert that the category content was loaded into system prompt
    assert scorer._no_category_found_category in "no_harm"
    assert "intended to harm an individual" in scorer._system_prompt


@pytest.mark.asyncio
async def test_category_scorer_score(memory: MemoryInterface, scorer_category_response_bullying: PromptRequestResponse):

    chat_target = MagicMock()

    chat_target.send_prompt_async = AsyncMock(return_value=scorer_category_response_bullying)

    scorer = SelfAskCategoryScorer(
        chat_target=chat_target,
        content_classifier=ContentClassifierPaths.HARMFUL_CONTENT_CLASSIFIER.value,
        memory=memory,
    )

    score = await scorer.score_text_async("this has a lot of bullying")

    assert len(score) == 1

    assert score[0].score_value == "True"
    assert "contains bullying" in score[0].score_rationale
    assert score[0].score_type == "true_false"
    assert score[0].score_category == "bullying"
    assert score[0].prompt_request_response_id is None


@pytest.mark.asyncio
async def test_category_scorer_score_false(
    memory: MemoryInterface, scorer_category_response_false: PromptRequestResponse
):

    chat_target = MagicMock()

    chat_target.send_prompt_async = AsyncMock(return_value=scorer_category_response_false)

    scorer = SelfAskCategoryScorer(
        chat_target=chat_target,
        content_classifier=ContentClassifierPaths.HARMFUL_CONTENT_CLASSIFIER.value,
        memory=memory,
    )

    score = await scorer.score_text_async("this has no bullying")

    assert len(score) == 1

    assert score[0].score_value == "False"
    assert score[0].score_type == "true_false"
    assert score[0].score_category == "no_harm"
    assert score[0].prompt_request_response_id is None


@pytest.mark.asyncio
async def test_category_scorer_adds_to_memory(scorer_category_response_false: PromptRequestResponse):
    memory = MagicMock(MemoryInterface)
    chat_target = MagicMock()
    chat_target.send_prompt_async = AsyncMock(return_value=scorer_category_response_false)

    scorer = SelfAskCategoryScorer(
        chat_target=chat_target,
        content_classifier=ContentClassifierPaths.HARMFUL_CONTENT_CLASSIFIER.value,
        memory=memory,
    )

    await scorer.score_text_async(text="string")

    memory.add_scores_to_memory.assert_called_once()
