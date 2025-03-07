# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
from typing import MutableSequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unit.mocks import get_sample_conversations

from pyrit.memory.central_memory import CentralMemory
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target import OpenAICompletionTarget


@pytest.fixture
def completions_response_json() -> dict:
    return {
        "id": "12345678-1a2b-3c4e5f-a123-12345678abcd",
        "object": "text_completion",
        "choices": [
            {
                "index": 0,
                "text": "hi",
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "model": "gpt-35-turbo",
    }


@pytest.fixture
def azure_completion_target(patch_central_database) -> OpenAICompletionTarget:
    return OpenAICompletionTarget(
        model_name="gpt-35-turbo",
        endpoint="https://mock.azure.com/",
        api_key="mock-api-key",
        api_version="some_version",
    )


@pytest.fixture
def sample_conversations() -> MutableSequence[PromptRequestPiece]:
    return get_sample_conversations()


@pytest.mark.asyncio
async def test_azure_completion_validate_request_length(
    azure_completion_target: OpenAICompletionTarget, sample_conversations: MutableSequence[PromptRequestPiece]
):
    request = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await azure_completion_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_azure_completion_validate_prompt_type(
    azure_completion_target: OpenAICompletionTarget, sample_conversations: MutableSequence[PromptRequestPiece]
):
    request_piece = sample_conversations[0]
    request_piece.converted_value_data_type = "image_path"
    request = PromptRequestResponse(request_pieces=[request_piece])
    with pytest.raises(ValueError, match="This target only supports text prompt input."):
        await azure_completion_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_azure_completion_validate_prev_convs(
    azure_completion_target: OpenAICompletionTarget, sample_conversations: MutableSequence[PromptRequestPiece]
):
    request_piece = sample_conversations[0]
    azure_completion_target._memory.add_request_response_to_memory(
        request=PromptRequestResponse(request_pieces=[request_piece])
    )
    request = PromptRequestResponse(request_pieces=[request_piece])

    with pytest.raises(ValueError, match="This target only supports a single turn conversation."):
        await azure_completion_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_azure_complete_async_return(
    completions_response_json: dict,
    azure_completion_target: OpenAICompletionTarget,
    sample_conversations: MutableSequence[PromptRequestPiece],
):
    request_piece = sample_conversations[0]
    request = PromptRequestResponse(request_pieces=[request_piece])

    openai_mock_return = MagicMock()
    openai_mock_return.text = json.dumps(completions_response_json)

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = openai_mock_return
        response: PromptRequestResponse = await azure_completion_target.send_prompt_async(prompt_request=request)
        assert len(response.request_pieces) == 1
        assert response.request_pieces[0].converted_value == "hi"


def test_azure_initialization_with_no_deployment_raises():
    with patch.object(CentralMemory, "get_memory_instance", return_value=MagicMock()):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                OpenAICompletionTarget()


def test_azure_invalid_endpoint_raises():
    with patch.object(CentralMemory, "get_memory_instance", return_value=MagicMock()):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                OpenAICompletionTarget(
                    model_name="gpt-4",
                    endpoint="",
                    api_key="xxxxx",
                    api_version="some_version",
                )
