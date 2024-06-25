# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from textwrap import dedent
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.common.constants import RETRY_MAX_NUM_ATTEMPTS
from pyrit.exceptions.exception_classes import InvalidJsonException
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.score import SelfAskObjectiveScorer, ObjectiveQuestionPaths

from tests.mocks import MockPromptTarget, get_memory_interface


@pytest.fixture
def scorer_meta_response() -> PromptRequestResponse:

    json_response = (
        dedent(
            """
        {"value": "True", "description": "This is true", "rationale": "rationale for true", "metadata": "50"}
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

    scorer = SelfAskObjectiveScorer(
        chat_target=chat_target,
        objective_question_path=ObjectiveQuestionPaths.REFUSAL.value,
    )

    score = await scorer.score_text_async("true false")

    assert len(score) == 1
    assert score[0].get_value() is True
    assert score[0].score_value_description == "This is true"
    assert score[0].score_rationale == "rationale for true"
    assert score[0].scorer_class_identifier["__type__"] == "SelfAskObjectiveScorer"


@pytest.mark.asyncio
async def test_meta_scorer_set_system_prompt(memory: MemoryInterface, scorer_meta_response: PromptRequestResponse):
    chat_target = MagicMock()
    chat_target.send_prompt_async = AsyncMock(return_value=scorer_meta_response)

    scorer = SelfAskObjectiveScorer(
        chat_target=chat_target,
        objective_question_path=ObjectiveQuestionPaths.REFUSAL.value,
        memory=memory,
    )

    await scorer.score_text_async("true false")

    chat_target.set_system_prompt.assert_called_once()

    # assert that the meta judge info was loaded into system prompt
    assert "certainty of your evaluation" in scorer._system_prompt


@pytest.mark.asyncio
async def test_meta_scorer_adds_to_memory(scorer_meta_response: PromptRequestResponse):
    memory = MagicMock(MemoryInterface)
    chat_target = MagicMock()
    chat_target.send_prompt_async = AsyncMock(return_value=scorer_meta_response)

    scorer = SelfAskObjectiveScorer(
        chat_target=chat_target,
        objective_question_path=ObjectiveQuestionPaths.REFUSAL.value,
        memory=memory,
    )

    await scorer.score_text_async(text="string")

    memory.add_scores_to_memory.assert_called_once()


@pytest.mark.asyncio
async def test_self_ask_objective_scorer_bad_json_exception_retries():

    chat_target = MockPromptTarget()

    with patch("tests.mocks.MockPromptTarget.send_prompt_async", new_callable=AsyncMock) as mock_create:
        bad_json_resp = PromptRequestResponse(
            request_pieces=[PromptRequestPiece(role="assistant", original_value="this is not a json")]
        )
        mock_create.return_value = bad_json_resp

        scorer = SelfAskObjectiveScorer(
            chat_target=chat_target,
            objective_question_path=ObjectiveQuestionPaths.REFUSAL.value,
            memory=memory,
        )
        with pytest.raises(InvalidJsonException):
            await scorer.score_text_async("test scoring prompt")

        assert mock_create.call_count == RETRY_MAX_NUM_ATTEMPTS
