# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.exceptions.exception_classes import InvalidJsonException
from pyrit.memory.central_memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.score import SelfAskTrueFalseScorer, TrueFalseQuestionPaths


@pytest.fixture
def scorer_true_false_response() -> PromptRequestResponse:

    json_response = (
        dedent(
            """
        {"score_value": "True", "description": "This is true", "rationale": "rationale for true"}
        """
        )
        .strip()
        .replace("\n", " ")
    )

    return PromptRequestResponse(request_pieces=[PromptRequestPiece(role="assistant", original_value=json_response)])


@pytest.mark.asyncio
async def test_true_false_scorer_score(patch_central_database, scorer_true_false_response: PromptRequestResponse):

    chat_target = MagicMock()

    chat_target.send_prompt_async = AsyncMock(return_value=scorer_true_false_response)
    scorer = SelfAskTrueFalseScorer(
        chat_target=chat_target, true_false_question_path=TrueFalseQuestionPaths.GROUNDED.value
    )

    score = await scorer.score_text_async("true false")

    assert len(score) == 1
    assert score[0].get_value() is True
    assert score[0].score_value_description == "This is true"
    assert score[0].score_rationale == "rationale for true"
    assert score[0].scorer_class_identifier["__type__"] == "SelfAskTrueFalseScorer"


@pytest.mark.asyncio
async def test_true_false_scorer_set_system_prompt(
    patch_central_database, scorer_true_false_response: PromptRequestResponse
):
    chat_target = MagicMock()
    chat_target.send_prompt_async = AsyncMock(return_value=scorer_true_false_response)

    scorer = SelfAskTrueFalseScorer(
        chat_target=chat_target, true_false_question_path=TrueFalseQuestionPaths.GROUNDED.value
    )

    await scorer.score_text_async("true false")

    chat_target.set_system_prompt.assert_called_once()

    # assert that the category content was loaded into system prompt
    assert "# Instructions" in scorer._system_prompt
    assert "Semantic Alignment:" in scorer._system_prompt


@pytest.mark.asyncio
async def test_true_false_scorer_adds_to_memory(scorer_true_false_response: PromptRequestResponse):
    memory = MagicMock(MemoryInterface)
    chat_target = MagicMock()
    chat_target.send_prompt_async = AsyncMock(return_value=scorer_true_false_response)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = SelfAskTrueFalseScorer(
            chat_target=chat_target, true_false_question_path=TrueFalseQuestionPaths.GROUNDED.value
        )

        await scorer.score_text_async(text="string")

        memory.add_scores_to_memory.assert_called_once()


@pytest.mark.asyncio
async def test_self_ask_scorer_bad_json_exception_retries(patch_central_database):

    chat_target = MagicMock()

    bad_json_resp = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value="this is not a json")]
    )
    chat_target.send_prompt_async = AsyncMock(return_value=bad_json_resp)
    scorer = SelfAskTrueFalseScorer(
        chat_target=chat_target, true_false_question_path=TrueFalseQuestionPaths.GROUNDED.value
    )

    with pytest.raises(InvalidJsonException):
        await scorer.score_text_async("this has no bullying")

    assert chat_target.send_prompt_async.call_count == int(os.getenv("RETRY_MAX_NUM_ATTEMPTS", 2))


@pytest.mark.asyncio
async def test_self_ask_objective_scorer_bad_json_exception_retries(patch_central_database):
    chat_target = MagicMock()

    json_response = (
        dedent(
            """
            {"badly_named_value": "True", "rationale": "rationale for true"}
            """
        )
        .strip()
        .replace("\n", " ")
    )

    bad_json_resp = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=json_response)]
    )

    chat_target.send_prompt_async = AsyncMock(return_value=bad_json_resp)
    scorer = SelfAskTrueFalseScorer(
        chat_target=chat_target, true_false_question_path=TrueFalseQuestionPaths.GROUNDED.value
    )

    with pytest.raises(InvalidJsonException):
        await scorer.score_text_async("this has no bullying")

    assert chat_target.send_prompt_async.call_count == int(os.getenv("RETRY_MAX_NUM_ATTEMPTS", 2))
