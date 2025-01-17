# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.exceptions.exception_classes import InvalidJsonException
from pyrit.memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.score.self_ask_category_scorer import (
    ContentClassifierPaths,
    SelfAskCategoryScorer,
)


@pytest.fixture
def scorer_category_response_bullying() -> PromptRequestResponse:

    json_response = (
        dedent(
            """
        {"category": "bullying", "category_description": "This is bullying.", "rationale": "The message
        seems like it contains bullying.", "score_value": "True"}
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
        {"category": "no_harm", "category_description": "No harms", "rationale": "The message
        contains no harms.", "score_value": "False"}
        """
        )
        .strip()
        .replace("\n", " ")
    )

    return PromptRequestResponse(request_pieces=[PromptRequestPiece(role="assistant", original_value=json_response)])


def test_category_scorer_set_no_category_found():
    chat_target = MagicMock()
    scorer = SelfAskCategoryScorer(
        chat_target=chat_target,
        content_classifier=ContentClassifierPaths.HARMFUL_CONTENT_CLASSIFIER.value,
    )

    # assert that the category content was loaded into system prompt
    assert scorer._no_category_found_category in "no_harm"
    assert "intended to harm an individual" in scorer._system_prompt


@pytest.mark.asyncio
async def test_category_scorer_set_system_prompt(
    scorer_category_response_bullying: PromptRequestResponse, patch_central_database
):
    chat_target = MagicMock()

    chat_target.send_prompt_async = AsyncMock(return_value=scorer_category_response_bullying)
    scorer = SelfAskCategoryScorer(
        chat_target=chat_target,
        content_classifier=ContentClassifierPaths.HARMFUL_CONTENT_CLASSIFIER.value,
    )

    await scorer.score_text_async("this has a lot of bullying")

    chat_target.set_system_prompt.assert_called_once()


@pytest.mark.asyncio
async def test_category_scorer_score(scorer_category_response_bullying: PromptRequestResponse, patch_central_database):

    chat_target = MagicMock()

    chat_target.send_prompt_async = AsyncMock(return_value=scorer_category_response_bullying)

    scorer = SelfAskCategoryScorer(
        chat_target=chat_target,
        content_classifier=ContentClassifierPaths.HARMFUL_CONTENT_CLASSIFIER.value,
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
    scorer_category_response_false: PromptRequestResponse, patch_central_database
):

    chat_target = MagicMock()

    chat_target.send_prompt_async = AsyncMock(return_value=scorer_category_response_false)

    scorer = SelfAskCategoryScorer(
        chat_target=chat_target,
        content_classifier=ContentClassifierPaths.HARMFUL_CONTENT_CLASSIFIER.value,
    )

    score = await scorer.score_text_async("this has no bullying")

    assert len(score) == 1

    assert score[0].score_value == "False"
    assert score[0].score_type == "true_false"
    assert score[0].score_category == "no_harm"
    assert score[0].prompt_request_response_id is None


@pytest.mark.asyncio
async def test_category_scorer_adds_to_memory(
    scorer_category_response_false: PromptRequestResponse, patch_central_database
):
    memory = MagicMock(MemoryInterface)
    chat_target = MagicMock()
    chat_target.send_prompt_async = AsyncMock(return_value=scorer_category_response_false)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):

        scorer = SelfAskCategoryScorer(
            chat_target=chat_target,
            content_classifier=ContentClassifierPaths.HARMFUL_CONTENT_CLASSIFIER.value,
        )

        await scorer.score_text_async(text="string")

        memory.add_scores_to_memory.assert_called_once()


@pytest.mark.asyncio
async def test_self_ask_objective_scorer_bad_json_exception_retries(patch_central_database):

    chat_target = MagicMock()

    bad_json_resp = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value="this is not a json")]
    )
    chat_target.send_prompt_async = AsyncMock(return_value=bad_json_resp)
    with patch.object(CentralMemory, "get_memory_instance", return_value=MagicMock()):

        scorer = SelfAskCategoryScorer(
            chat_target=chat_target,
            content_classifier=ContentClassifierPaths.HARMFUL_CONTENT_CLASSIFIER.value,
        )

        with pytest.raises(InvalidJsonException):
            await scorer.score_text_async("this has no bullying")
        assert chat_target.send_prompt_async.call_count == int(os.getenv("RETRY_MAX_NUM_ATTEMPTS"))


@pytest.mark.asyncio
async def test_self_ask_objective_scorer_json_missing_key_exception_retries(patch_central_database):

    chat_target = MagicMock()

    json_response = (
        dedent(
            """
            {"wrongly_named_category_name": "bullying",
            "category_description": "This is bullying.",
            "rationale": "The message seems like it contains bullying."}
            """
        )
        .strip()
        .replace("\n", " ")
    )

    bad_json_resp = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=json_response)]
    )
    chat_target.send_prompt_async = AsyncMock(return_value=bad_json_resp)
    with patch.object(CentralMemory, "get_memory_instance", return_value=MagicMock()):

        scorer = SelfAskCategoryScorer(
            chat_target=chat_target,
            content_classifier=ContentClassifierPaths.HARMFUL_CONTENT_CLASSIFIER.value,
        )

        with pytest.raises(InvalidJsonException):
            await scorer.score_text_async("this has no bullying")
        assert chat_target.send_prompt_async.call_count == int(os.getenv("RETRY_MAX_NUM_ATTEMPTS"))


@pytest.mark.asyncio
@pytest.mark.parametrize("max_requests_per_minute", [None, 10])
@pytest.mark.parametrize("batch_size", [1, 10])
async def test_score_prompts_batch_async(
    max_requests_per_minute: int,
    batch_size: int,
    scorer_category_response_false: PromptRequestResponse,
    patch_central_database,
):
    chat_target = AsyncMock()
    chat_target._max_requests_per_minute = max_requests_per_minute
    with patch.object(CentralMemory, "get_memory_instance", return_value=MagicMock()):
        scorer = SelfAskCategoryScorer(
            chat_target=chat_target,
            content_classifier=ContentClassifierPaths.HARMFUL_CONTENT_CLASSIFIER.value,
        )

        prompt = PromptRequestPiece(role="assistant", original_value="test")
        prompt2 = PromptRequestPiece(role="assistant", original_value="test 2")

        with patch.object(chat_target, "send_prompt_async", return_value=scorer_category_response_false):
            if batch_size != 1 and max_requests_per_minute:
                with pytest.raises(ValueError):
                    await scorer.score_prompts_with_tasks_batch_async(
                        request_responses=[prompt], batch_size=batch_size, tasks=[""]
                    )
            else:
                results = await scorer.score_prompts_with_tasks_batch_async(
                    request_responses=[prompt, prompt2], batch_size=batch_size, tasks=["", ""]
                )
                assert len(results) == 2
