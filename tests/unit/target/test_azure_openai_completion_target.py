# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
from typing import MutableSequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unit.mocks import get_sample_conversations

from pyrit.memory.central_memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
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
        assert response.get_value() == "hi"


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


@pytest.mark.asyncio
async def test_openai_completion_target_no_api_version(sample_conversations: MutableSequence[PromptRequestPiece]):
    target = OpenAICompletionTarget(
        api_key="test_key", endpoint="https://mock.azure.com", model_name="gpt-35-turbo", api_version=None
    )
    request_piece = sample_conversations[0]
    request = PromptRequestResponse(request_pieces=[request_piece])

    with patch("httpx.AsyncClient.request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = MagicMock()
        mock_request.return_value.status_code = 200
        mock_request.return_value.text = '{"choices": [{"text": "hi"}]}'

        await target.send_prompt_async(prompt_request=request)

        called_params = mock_request.call_args[1]["params"]
        assert "api-version" not in called_params


@pytest.mark.asyncio
async def test_openai_completion_target_default_api_version(sample_conversations: MutableSequence[PromptRequestPiece]):
    target = OpenAICompletionTarget(api_key="test_key", endpoint="https://mock.azure.com", model_name="gpt-35-turbo")
    request_piece = sample_conversations[0]
    request = PromptRequestResponse(request_pieces=[request_piece])

    with patch("httpx.AsyncClient.request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = MagicMock()
        mock_request.return_value.status_code = 200
        mock_request.return_value.text = '{"choices": [{"text": "hi"}]}'

        await target.send_prompt_async(prompt_request=request)

        called_params = mock_request.call_args[1]["params"]
        assert "api-version" in called_params
        assert called_params["api-version"] == "2024-06-01"


@pytest.mark.asyncio
async def test_send_prompt_async_calls_refresh_auth_headers(azure_completion_target: OpenAICompletionTarget):
    mock_memory = MagicMock(spec=MemoryInterface)
    mock_memory.get_conversation.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()

    azure_completion_target._memory = mock_memory

    with (
        patch.object(azure_completion_target, "refresh_auth_headers") as mock_refresh,
        patch.object(azure_completion_target, "_validate_request"),
        patch.object(azure_completion_target, "_construct_request_body", new_callable=AsyncMock) as mock_construct,
    ):

        mock_construct.return_value = {}

        with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async") as mock_make_request:
            mock_make_request.return_value = MagicMock(text='{"choices": [{"text": "test response"}]}')

            prompt_request = PromptRequestResponse(
                request_pieces=[
                    PromptRequestPiece(
                        role="user",
                        original_value="test prompt",
                        converted_value="test prompt",
                        converted_value_data_type="text",
                    )
                ]
            )
            await azure_completion_target.send_prompt_async(prompt_request=prompt_request)

            mock_refresh.assert_called_once()
