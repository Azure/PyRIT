# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import uuid
from typing import MutableSequence
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from unit.mocks import get_sample_conversations

from pyrit.exceptions.exception_classes import (
    EmptyResponseException,
    RateLimitException,
)
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target import OpenAIDALLETarget


@pytest.fixture
def dalle_target(patch_central_database) -> OpenAIDALLETarget:
    return OpenAIDALLETarget(
        model_name="test",
        endpoint="test",
        api_key="test",
    )


@pytest.fixture
def dalle_response_json() -> dict:
    return {
        "data": [
            {
                "b64_json": "aGVsbG8=",
            }
        ],
        "model": "dall-e-3",
    }


@pytest.fixture
def sample_conversations() -> MutableSequence[PromptRequestPiece]:
    return get_sample_conversations()


def test_initialization_with_required_parameters(dalle_target: OpenAIDALLETarget):
    assert dalle_target
    assert dalle_target._model_name == "test"


def test_initialization_invalid_num_images():
    with pytest.raises(ValueError):
        OpenAIDALLETarget(
            model_name="test",
            endpoint="test",
            api_key="test",
            dalle_version="dall-e-3",
            num_images=3,
        )


@pytest.mark.asyncio
async def test_send_prompt_async(
    dalle_target: OpenAIDALLETarget,
    sample_conversations: MutableSequence[PromptRequestPiece],
    dalle_response_json: dict,
):
    request = sample_conversations[0]

    openai_mock_return = MagicMock()
    openai_mock_return.text = json.dumps(dalle_response_json)

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:

        mock_request.return_value = openai_mock_return

        resp = await dalle_target.send_prompt_async(prompt_request=PromptRequestResponse([request]))
        assert resp
        path = resp.request_pieces[0].original_value
        assert os.path.isfile(path)

        with open(path, "r") as file:
            data = file.read()
            assert data == "hello"

        os.remove(path)


@pytest.mark.asyncio
async def test_send_prompt_async_empty_response(
    dalle_target: OpenAIDALLETarget,
    sample_conversations: MutableSequence[PromptRequestPiece],
    dalle_response_json: dict,
):
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    dalle_response_json["data"][0]["b64_json"] = ""
    openai_mock_return = MagicMock()
    openai_mock_return.text = json.dumps(dalle_response_json)

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:

        mock_request.return_value = openai_mock_return

        with pytest.raises(EmptyResponseException) as e:
            await dalle_target.send_prompt_async(prompt_request=PromptRequestResponse([request]))
            assert str(e.value) == "Status Code: 204, Message: The chat returned an empty response."
            assert mock_request.call_count == os.getenv("RETRY_MAX_NUM_ATTEMPTS")


@pytest.mark.asyncio
async def test_send_prompt_async_rate_limit_exception(
    dalle_target: OpenAIDALLETarget, sample_conversations: MutableSequence[PromptRequestPiece]
):
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    response = MagicMock()
    response.status_code = 429

    side_effect = httpx.HTTPStatusError("Rate Limit Reached", response=response, request=MagicMock())

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", side_effect=side_effect
    ) as mock_request:

        with pytest.raises(RateLimitException) as rle:
            await dalle_target.send_prompt_async(prompt_request=PromptRequestResponse([request]))
            assert mock_request.call_count == os.getenv("RETRY_MAX_NUM_ATTEMPTS")
            assert str(rle.value) == "Rate Limit Reached"


@pytest.mark.asyncio
async def test_send_prompt_async_bad_request_error(
    dalle_target: OpenAIDALLETarget, sample_conversations: MutableSequence[PromptRequestPiece]
):
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    response = MagicMock()
    response.status_code = 400

    side_effect = httpx.HTTPStatusError("Bad Request Error", response=response, request=MagicMock())

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", side_effect=side_effect
    ) as mock_request:
        with pytest.raises(httpx.HTTPStatusError) as rle:
            await dalle_target.send_prompt_async(prompt_request=PromptRequestResponse([request]))
            assert mock_request.call_count == 1
            assert str(rle.value) == "Bad Request Error"


@pytest.mark.asyncio
async def test_dalle_validate_request_length(
    dalle_target: OpenAIDALLETarget, sample_conversations: MutableSequence[PromptRequestPiece]
):
    request = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await dalle_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_dalle_validate_prompt_type(
    dalle_target: OpenAIDALLETarget, sample_conversations: MutableSequence[PromptRequestPiece]
):
    request_piece = sample_conversations[0]
    request_piece.converted_value_data_type = "image_path"
    request = PromptRequestResponse(request_pieces=[request_piece])
    with pytest.raises(ValueError, match="This target only supports text prompt input."):
        await dalle_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_send_prompt_async_empty_response_adds_memory(
    dalle_target: OpenAIDALLETarget,
    sample_conversations: MutableSequence[PromptRequestPiece],
    dalle_response_json: dict,
) -> None:

    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()

    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    dalle_response_json["data"][0]["b64_json"] = ""
    openai_mock_return = MagicMock()
    openai_mock_return.text = json.dumps(dalle_response_json)

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:

        mock_request.return_value = openai_mock_return
        dalle_target._memory = mock_memory

        with pytest.raises(EmptyResponseException):
            await dalle_target.send_prompt_async(prompt_request=PromptRequestResponse([request]))
            assert mock_memory.add_request_response_to_memory.call_count == os.getenv("RETRY_MAX_NUM_ATTEMPTS")


@pytest.mark.asyncio
async def test_send_prompt_async_rate_limit_adds_memory(
    dalle_target: OpenAIDALLETarget,
    sample_conversations: MutableSequence[PromptRequestPiece],
) -> None:
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()

    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    response = MagicMock()
    response.status_code = 429

    side_effect = httpx.HTTPStatusError("Rate Limit Reached", response=response, request=MagicMock())

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async", side_effect=side_effect):

        dalle_target._memory = mock_memory

        with pytest.raises(RateLimitException):
            await dalle_target.send_prompt_async(prompt_request=PromptRequestResponse([request]))
            assert mock_memory.add_request_response_to_memory.call_count == os.getenv("RETRY_MAX_NUM_ATTEMPTS")


@pytest.mark.asyncio
async def test_send_prompt_async_bad_request_adds_memory(
    dalle_target: OpenAIDALLETarget,
    sample_conversations: MutableSequence[PromptRequestPiece],
) -> None:

    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()

    dalle_target._memory = mock_memory

    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    response = MagicMock()
    response.status_code = 400

    side_effect = httpx.HTTPStatusError("Bad Request Error", response=response, request=MagicMock())

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async", side_effect=side_effect):
        with pytest.raises(httpx.HTTPStatusError) as rle:
            await dalle_target.send_prompt_async(prompt_request=PromptRequestResponse([request]))
            assert dalle_target._memory.add_request_response_to_memory.assert_called_once()
            assert str(rle.value) == "Bad Request Error"


def test_is_json_response_supported(patch_central_database):
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()

    mock_dalle_target = OpenAIDALLETarget(model_name="test", endpoint="test", api_key="test")
    assert mock_dalle_target.is_json_response_supported() is False


@pytest.mark.asyncio
async def test_dalle_target_no_api_version(
    dalle_target: OpenAIDALLETarget,
    sample_conversations: MutableSequence[PromptRequestPiece],
    dalle_response_json: dict,
):
    target = OpenAIDALLETarget(
        api_key="test_key", endpoint="https://mock.azure.com", model_name="dalle-3", api_version=None
    )
    request = PromptRequestResponse([sample_conversations[0]])

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = MagicMock()
        mock_request.return_value.status_code = 200
        mock_request.return_value.text = json.dumps(dalle_response_json)

        await target.send_prompt_async(prompt_request=request)

        called_params = mock_request.call_args[1]["params"]
        assert "api-version" not in called_params


@pytest.mark.asyncio
async def test_dalle_target_default_api_version(
    dalle_target: OpenAIDALLETarget,
    sample_conversations: MutableSequence[PromptRequestPiece],
    dalle_response_json: dict,
):
    target = OpenAIDALLETarget(api_key="test_key", endpoint="https://mock.azure.com", model_name="dalle-3")
    request = PromptRequestResponse([sample_conversations[0]])

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = MagicMock()
        mock_request.return_value.status_code = 200
        mock_request.return_value.text = json.dumps(dalle_response_json)

        await target.send_prompt_async(prompt_request=request)

        called_params = mock_request.call_args[1]["params"]
        assert "api-version" in called_params
        assert called_params["api-version"] == "2024-06-01"


@pytest.mark.asyncio
async def test_send_prompt_async_calls_refresh_auth_headers(dalle_target):
    mock_memory = MagicMock(spec=MemoryInterface)
    mock_memory.get_conversation.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()

    dalle_target._memory = mock_memory

    dalle_target.refresh_auth_headers = MagicMock()
    dalle_target._validate_request = MagicMock()
    dalle_target._construct_request_body = AsyncMock(return_value={})

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async") as mock_make_request:
        mock_response = MagicMock()
        mock_response.text = json.dumps({"data": [{"b64_json": "aGVsbG8="}]})  # Base64 encoded "hello"
        mock_make_request.return_value = mock_response

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
        await dalle_target.send_prompt_async(prompt_request=prompt_request)

        dalle_target.refresh_auth_headers.assert_called_once()
