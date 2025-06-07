# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.score.general_scorer import SelfAskGeneralScorer


@pytest.fixture
def general_scorer_response() -> PromptRequestResponse:
    json_response = (
        dedent(
            """
        {"score_value": "1",
         "rationale": "This is the rationale.",
         "description": "This is the description."}
        """
        )
        .strip()
        .replace("\n", " ")
    )

    return PromptRequestResponse(request_pieces=[PromptRequestPiece(role="assistant", original_value=json_response)])


@pytest.mark.asyncio
async def test_general_scorer_score_async(patch_central_database, general_scorer_response: PromptRequestResponse):
    chat_target = MagicMock()
    chat_target.send_prompt_async = AsyncMock(return_value=general_scorer_response)

    scorer = SelfAskGeneralScorer(
        chat_target=chat_target,
        system_prompt_format_string="This is a system prompt.",
        scorer_type="float_scale",
        category=["test_category"],
        min_value=1,
        max_value=5,
    )
    assert scorer

    score = await scorer.score_text_async(
        text="test prompt",
    )

    assert len(score) == 1
    assert score[0].score_value == "0.0"
    assert "This is the rationale." in score[0].score_rationale
    assert "This is the description." in score[0].score_value_description


@pytest.mark.asyncio
async def test_general_scorer_invalid_scorer_type():
    chat_target = MagicMock()

    with pytest.raises(ValueError, match="Scorer type invalid_type is not a valid scorer type."):
        SelfAskGeneralScorer(
            chat_target=chat_target,
            system_prompt_format_string="This is a system prompt.",
            scorer_type="invalid_type",
        )


@pytest.mark.asyncio
async def test_general_scorer_score_async_with_prompt_f_string(
    general_scorer_response: PromptRequestResponse, patch_central_database
):
    chat_target = MagicMock()
    chat_target.send_prompt_async = AsyncMock(return_value=general_scorer_response)

    scorer = SelfAskGeneralScorer(
        chat_target=chat_target,
        system_prompt_format_string="This is a system prompt.",
        prompt_format_string="Is this cyberbullying? {prompt}",
        scorer_type="float_scale",
        category=["test_category"],
        min_value=1,
        max_value=5,
    )

    score = await scorer.score_text_async(
        text="this is a test prompt",
    )

    assert len(score) == 1
    assert score[0].score_value == "0.0"
    assert "This is the rationale." in score[0].score_rationale
    assert "This is the description." in score[0].score_value_description
    args = chat_target.send_prompt_async.call_args
    prompt = args[1]["prompt_request"].request_pieces[0].converted_value
    assert prompt == "Is this cyberbullying? this is a test prompt"


@pytest.mark.asyncio
async def test_general_scorer_score_async_handles_custom_keys(patch_central_database):
    chat_target = MagicMock()
    assert chat_target

    json_response = (
        dedent(
            """
        {"score_value": "1",
         "rationale_diff_key": "This is the rationale.",
         "description": "This is the description."}
        """
        )
        .strip()
        .replace("\n", " ")
    )
    # Simulate a response missing some keys
    response = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=json_response)]
    )
    chat_target.send_prompt_async = AsyncMock(return_value=response)

    assert chat_target

    scorer = SelfAskGeneralScorer(
        chat_target=chat_target,
        system_prompt_format_string="This is a system prompt.",
        prompt_format_string="This is a prompt format string.",
        scorer_type="float_scale",
        category=["test_category"],
        rationale_output_key="rationale_diff_key",
    )
    await scorer.score_text_async(text="this is a test prompt")
