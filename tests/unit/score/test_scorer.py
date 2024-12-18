# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from textwrap import dedent
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.exceptions import InvalidJsonException, remove_markdown_json
from pyrit.models import PromptRequestPiece
from pyrit.models import PromptRequestResponse, Score
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import Scorer


class MockScorer(Scorer):
    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        return []

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        pass


@pytest.fixture
def good_json():
    return (
        dedent(
            """
            {"score_value": "1",
            "rationale": "The message does not contain any code or instructions that could be used to cause harm"}
            """
        )
        .strip()
        .replace("\n", " ")
    )


BAD_JSON = "this is not json"

KEY_ERROR_JSON = (
    dedent(
        """
        {"poor_value": "1",
        "rationale": "The message does not contain any code or instructions that could be used to cause harm"}
        """
    )
    .strip()
    .replace("\n", " ")
)

KEY_ERROR2_JSON = (
    dedent(
        """
        {"score_value": "1",
        "bad_rationale": "The message does not contain any code or instructions that could be used to cause harm"}
        """
    )
    .strip()
    .replace("\n", " ")
)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_json", [BAD_JSON, KEY_ERROR_JSON, KEY_ERROR2_JSON])
async def test_scorer_send_chat_target_async_bad_json_exception_retries(bad_json: str):

    chat_target = MagicMock(PromptChatTarget)

    bad_json_resp = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=bad_json)]
    )
    chat_target.send_prompt_async = AsyncMock(return_value=bad_json_resp)

    scorer = MockScorer()
    scorer.scorer_type = "true_false"

    with pytest.raises(InvalidJsonException):
        await scorer._score_value_with_llm(
            prompt_target=chat_target,
            system_prompt="system_prompt",
            prompt_request_value="prompt_request_value",
            prompt_request_data_type="text",
            scored_prompt_id="123",
            category="category",
            task="task",
        )

        assert chat_target.send_prompt_async.call_count == int(os.getenv("RETRY_MAX_NUM_ATTEMPTS"))


@pytest.mark.asyncio
async def test_scorer_send_chat_target_async_good_response(good_json):

    chat_target = MagicMock(PromptChatTarget)

    good_json_resp = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=good_json)]
    )
    chat_target.send_prompt_async = AsyncMock(return_value=good_json_resp)

    scorer = MockScorer()
    scorer.scorer_type = "true_false"

    await scorer._score_value_with_llm(
        prompt_target=chat_target,
        system_prompt="system_prompt",
        prompt_request_value="prompt_request_value",
        prompt_request_data_type="text",
        scored_prompt_id="123",
        category="category",
        task="task",
    )

    assert chat_target.send_prompt_async.call_count == int(1)


@pytest.mark.asyncio
async def test_scorer_remove_markdown_json_called(good_json):

    chat_target = MagicMock(PromptChatTarget)
    good_json_resp = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=good_json)]
    )
    chat_target.send_prompt_async = AsyncMock(return_value=good_json_resp)

    scorer = MockScorer()
    scorer.scorer_type = "true_false"

    with patch("pyrit.score.scorer.remove_markdown_json", wraps=remove_markdown_json) as mock_remove_markdown_json:
        await scorer._score_value_with_llm(
            prompt_target=chat_target,
            system_prompt="system_prompt",
            prompt_request_value="prompt_request_value",
            prompt_request_data_type="text",
            scored_prompt_id="123",
            category="category",
            task="task",
        )

        mock_remove_markdown_json.assert_called_once()
