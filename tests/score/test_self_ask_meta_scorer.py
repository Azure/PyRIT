# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from tests.mocks import get_memory_interface
from textwrap import dedent
from typing import Generator
from unittest.mock import AsyncMock, MagicMock

from pyrit.common.constants import RETRY_MAX_NUM_ATTEMPTS
from pyrit.exceptions.exception_classes import InvalidJsonException
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.score import SelfAskMetaScorer, MetaScorerQuestionPaths


@pytest.fixture
def scorer_meta_response() -> PromptRequestResponse:

    json_response = (
        dedent(
            """
        {"value": "True", "description": "This is true", "rationale": "rationale for true"}
        """
        )
        .strip()
        .replace("\n", " ")
    )

    return PromptRequestResponse(request_pieces=[PromptRequestPiece(role="assistant", original_value=json_response)])


@pytest.fixture
def memory() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.mark.asyncio
async def test_meta_scorer_score(memory: MemoryInterface, scorer_meta_response: PromptRequestResponse):

    chat_target = MagicMock()

    chat_target.send_prompt_async = AsyncMock(return_value=scorer_meta_response)

    scorer = SelfAskMetaScorer(
        chat_target=chat_target,
        meta_scorer_question_path=MetaScorerQuestionPaths.META_JUDGE_PROMPT.value,
    )

    score = await scorer.score_text_async("true false")

    assert len(score) == 1
    assert score[0].get_value() is True
    assert score[0].score_value_description == "This is true"
    assert score[0].score_rationale == "rationale for true"
    assert score[0].scorer_class_identifier["__type__"] == "SelfAskMetaScorer"


@pytest.mark.asyncio
async def test_meta_scorer_set_system_prompt(memory: MemoryInterface, scorer_meta_response: PromptRequestResponse):
    chat_target = MagicMock()
    chat_target.send_prompt_async = AsyncMock(return_value=scorer_meta_response)

    scorer = SelfAskMetaScorer(
        chat_target=chat_target,
        meta_scorer_question_path=MetaScorerQuestionPaths.META_JUDGE_PROMPT.value,
        memory=memory,
    )

    await scorer.score_text_async("true false")

    chat_target.set_system_prompt.assert_called_once()

    # assert that the meta judge info was loaded into system prompt
    assert "A hateful manifest was generated." in scorer._system_prompt


@pytest.mark.asyncio
async def test_meta_scorer_adds_to_memory(scorer_meta_response: PromptRequestResponse):
    memory = MagicMock(MemoryInterface)
    chat_target = MagicMock()
    chat_target.send_prompt_async = AsyncMock(return_value=scorer_meta_response)

    scorer = SelfAskMetaScorer(
        chat_target=chat_target,
        meta_scorer_question_path=MetaScorerQuestionPaths.META_JUDGE_PROMPT.value,
        memory=memory,
    )

    await scorer.score_text_async(text="string")

    memory.add_scores_to_memory.assert_called_once()


@pytest.mark.asyncio
async def test_self_ask_scorer_bad_json_exception_retries():

    chat_target = MagicMock()

    bad_json_resp = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value="this is not a json")]
    )
    chat_target.send_prompt_async = AsyncMock(return_value=bad_json_resp)

    scorer = SelfAskMetaScorer(
        chat_target=chat_target,
        meta_scorer_question_path=MetaScorerQuestionPaths.META_JUDGE_PROMPT.value,
        memory=memory,
    )

    with pytest.raises(InvalidJsonException):
        await scorer.score_text_async("this has no bullying")

    assert chat_target.send_prompt_async.call_count == RETRY_MAX_NUM_ATTEMPTS


@pytest.mark.asyncio
async def test_self_ask_meta_scorer_json_missing_key_exception_retries():

    chat_target = MagicMock()

    json_response = (
        dedent(
            """
        {"value": "True", "description": "This is true", "badly_named_rationale": "rationale for true"}
        """
        )
        .strip()
        .replace("\n", " ")
    )

    bad_json_resp = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=json_response)]
    )

    chat_target.send_prompt_async = AsyncMock(return_value=bad_json_resp)

    scorer = SelfAskMetaScorer(
        chat_target=chat_target,
        meta_scorer_question_path=MetaScorerQuestionPaths.META_JUDGE_PROMPT.value,
        memory=memory,
    )

    with pytest.raises(InvalidJsonException):
        await scorer.score_text_async("this has no bullying")

    assert chat_target.send_prompt_async.call_count == RETRY_MAX_NUM_ATTEMPTS
