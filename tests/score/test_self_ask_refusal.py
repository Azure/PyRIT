# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile
from textwrap import dedent
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from pyrit.exceptions.exception_classes import InvalidJsonException
from pyrit.memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import PromptRequestPiece
from pyrit.models import PromptRequestResponse

from pyrit.score.self_ask_refusal_scorer import SelfAskRefusalScorer
from tests.mocks import get_memory_interface


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


@pytest.fixture
def memory() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.mark.asyncio
async def test_refusal_scorer_score(memory: MemoryInterface, scorer_true_false_response: PromptRequestResponse):

    chat_target = MagicMock()

    chat_target.send_prompt_async = AsyncMock(return_value=scorer_true_false_response)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = SelfAskRefusalScorer(chat_target=chat_target)

        score = await scorer.score_text_async("true false")

        assert len(score) == 1
        assert score[0].get_value() is True
        assert score[0].score_value_description == "This is true"
        assert score[0].score_rationale == "rationale for true"
        assert score[0].scorer_class_identifier["__type__"] == "SelfAskRefusalScorer"


@pytest.mark.asyncio
async def test_refusal_scorer_set_system_prompt(
    memory: MemoryInterface, scorer_true_false_response: PromptRequestResponse
):
    chat_target = MagicMock()
    chat_target.send_prompt_async = AsyncMock(return_value=scorer_true_false_response)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = SelfAskRefusalScorer(chat_target=chat_target)

        await scorer.score_text_async("true false")

        chat_target.set_system_prompt.assert_called_once()


@pytest.mark.asyncio
async def test_refusal_scorer_no_task(memory: MemoryInterface, scorer_true_false_response: PromptRequestResponse):
    chat_target = MagicMock()
    chat_target.send_prompt_async = AsyncMock(return_value=scorer_true_false_response)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = SelfAskRefusalScorer(chat_target=chat_target)

        await scorer.score_text_async("true false")

        _, kwargs = chat_target.send_prompt_async.call_args
        assert kwargs["prompt_request"].request_pieces[0].original_value == "response_to_evaluate_input: true false"


@pytest.mark.asyncio
async def test_refusal_scorer_with_task(memory: MemoryInterface, scorer_true_false_response: PromptRequestResponse):
    chat_target = MagicMock()
    chat_target.send_prompt_async = AsyncMock(return_value=scorer_true_false_response)
    with patch.object(CentralMemory, "get_memory_instance"):
        scorer = SelfAskRefusalScorer(chat_target=chat_target)

        await scorer.score_text_async("true false", task="task")

        _, kwargs = chat_target.send_prompt_async.call_args

        expected = "conversation_objective: task\nresponse_to_evaluate_input: true false"
        assert kwargs["prompt_request"].request_pieces[0].original_value == expected


@pytest.mark.asyncio
async def test_refusal_scorer_image_non_block(
    memory: MemoryInterface, scorer_true_false_response: PromptRequestResponse
):
    chat_target = MagicMock()
    chat_target.send_prompt_async = AsyncMock(return_value=scorer_true_false_response)

    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = SelfAskRefusalScorer(chat_target=chat_target)

        with tempfile.NamedTemporaryFile(delete=True) as temp_image:
            result = await scorer.score_image_async(temp_image.name)
            assert result[0].get_value() is False

        # Any image doesn't need LLM evaluation, it just checks whether it was blocked
        chat_target.send_prompt_async.assert_not_called()


@pytest.mark.asyncio
async def test_refusal_scorer_adds_to_memory(scorer_true_false_response: PromptRequestResponse):
    memory = MagicMock(MemoryInterface)
    chat_target = MagicMock()
    chat_target.send_prompt_async = AsyncMock(return_value=scorer_true_false_response)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = SelfAskRefusalScorer(chat_target=chat_target)
        await scorer.score_text_async(text="string")

        memory.add_scores_to_memory.assert_called_once()


@pytest.mark.asyncio
async def test_refusal_scorer_bad_json_exception_retries(memory: MemoryInterface):

    chat_target = MagicMock()

    bad_json_resp = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value="this is not a json")]
    )
    chat_target.send_prompt_async = AsyncMock(return_value=bad_json_resp)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = SelfAskRefusalScorer(chat_target=chat_target)

        with pytest.raises(InvalidJsonException):
            await scorer.score_text_async("this has no bullying")

        assert chat_target.send_prompt_async.call_count == int(os.getenv("RETRY_MAX_NUM_ATTEMPTS", 2))


@pytest.mark.asyncio
async def test_self_ask_objective_scorer_bad_json_exception_retries(memory: MemoryInterface):
    chat_target = MagicMock()

    json_response = (
        dedent(
            """
            {"bad_value_key": "True", "rationale": "rationale for true"}
            """
        )
        .strip()
        .replace("\n", " ")
    )

    bad_json_resp = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=json_response)]
    )

    chat_target.send_prompt_async = AsyncMock(return_value=bad_json_resp)

    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = SelfAskRefusalScorer(chat_target=chat_target)

        with pytest.raises(InvalidJsonException):
            await scorer.score_text_async("this has no bullying")

        assert chat_target.send_prompt_async.call_count == int(os.getenv("RETRY_MAX_NUM_ATTEMPTS", 2))


@pytest.mark.asyncio
async def test_score_async_filtered_response(memory: MemoryInterface):
    chat_target = MagicMock()
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = SelfAskRefusalScorer(chat_target=chat_target)

        request_piece = PromptRequestPiece(
            role="assistant", original_value="blocked response", response_error="blocked"
        )
        scores = await scorer.score_async(request_piece)

        assert len(scores) == 1
        assert scores[0].score_value == "True"
        assert scores[0].score_value_description == "Refusal detected"
        assert scores[0].score_rationale == "Content was filtered, constituting a refusal."
        assert scores[0].scorer_class_identifier["__type__"] == "SelfAskRefusalScorer"
