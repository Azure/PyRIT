# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import BadRequestError, RateLimitError
from unit.mocks import get_sample_conversations

from pyrit.exceptions.exception_classes import EmptyResponseException
from pyrit.memory.central_memory import CentralMemory
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target import OpenAIDALLETarget


@pytest.fixture
def dalle_target(patch_central_database) -> OpenAIDALLETarget:
    return OpenAIDALLETarget(
        model_name="test",
        target_uri="test",
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
def sample_conversations() -> list[PromptRequestPiece]:
    return get_sample_conversations()


def test_initialization_with_required_parameters(dalle_target: OpenAIDALLETarget):
    assert dalle_target
    assert dalle_target._model_name == "test"


def test_initialization_invalid_num_images():
    with pytest.raises(ValueError):
        OpenAIDALLETarget(
            model_name="test",
            target_uri="test",
            api_key="test",
            dalle_version="dall-e-3",
            num_images=3,
        )


@pytest.mark.asyncio
async def test_send_prompt_async(
    dalle_target: OpenAIDALLETarget,
    sample_conversations: list[PromptRequestPiece],
    dalle_response_json: dict
):
    request = sample_conversations[0]

    openai_mock_return = MagicMock()
    openai_mock_return.text = json.dumps(dalle_response_json)

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock) as mock_request:

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
    sample_conversations: list[PromptRequestPiece],
    dalle_response_json: dict
):
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())


    dalle_response_json["data"][0]["b64_json"] = ""
    openai_mock_return = MagicMock()
    openai_mock_return.text = json.dumps(dalle_response_json)

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock) as mock_request:

        mock_request.return_value = openai_mock_return

        with pytest.raises(EmptyResponseException) as e:
            await dalle_target.send_prompt_async(prompt_request=PromptRequestResponse([request]))
            assert str(e.value) == "Status Code: 204, Message: The chat returned an empty response."
            assert mock_request.call_count == os.getenv("RETRY_MAX_NUM_ATTEMPTS")


@pytest.mark.asyncio
async def test_send_prompt_async_rate_limit_exception(
    dalle_target: OpenAIDALLETarget, sample_conversations: list[PromptRequestPiece]
):
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    response = MagicMock()
    response.status_code = 429
    mock_image_resp_async = AsyncMock(
        side_effect=RateLimitError("Rate Limit Reached", response=response, body="Rate limit reached")
    )
    setattr(dalle_target, "_generate_image_response_async", mock_image_resp_async)

    with pytest.raises(RateLimitError):
        await dalle_target.send_prompt_async(prompt_request=PromptRequestResponse([request]))
        assert mock_image_resp_async.call_count == os.getenv("RETRY_MAX_NUM_ATTEMPTS")


@pytest.mark.asyncio
async def test_send_prompt_async_bad_request_error(
    dalle_target: OpenAIDALLETarget, sample_conversations: list[PromptRequestPiece]
):
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    response = MagicMock()
    response.status_code = 400
    mock_image_resp_async = AsyncMock(
        side_effect=BadRequestError("Bad Request Error", response=response, body="Bad Request")
    )
    setattr(dalle_target, "_generate_image_response_async", mock_image_resp_async)
    with pytest.raises(BadRequestError) as bre:
        await dalle_target.send_prompt_async(prompt_request=PromptRequestResponse([request]))
    assert str(bre.value) == "Bad Request Error"


@pytest.mark.asyncio
async def test_dalle_validate_request_length(
    dalle_target: OpenAIDALLETarget, sample_conversations: list[PromptRequestPiece]
):
    request = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await dalle_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_dalle_validate_prompt_type(
    dalle_target: OpenAIDALLETarget, sample_conversations: list[PromptRequestPiece]
):
    request_piece = sample_conversations[0]
    request_piece.converted_value_data_type = "image_path"
    request = PromptRequestResponse(request_pieces=[request_piece])
    with pytest.raises(ValueError, match="This target only supports text prompt input."):
        await dalle_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_dalle_send_prompt_file_save_async() -> None:

    request = PromptRequestPiece(
        role="user",
        original_value="draw me a test picture",
    ).to_prompt_request_response()

    mock_return = MagicMock()

    # "test image data" b64 encoded
    mock_return.model_dump_json.return_value = '{"data": [{"b64_json": "dGVzdCBpbWFnZSBkYXRh"}]}'
    mock_dalle_target = OpenAIDALLETarget(deployment_name="test", endpoint="test", api_key="test")
    mock_dalle_target._async_client.images = MagicMock()
    mock_dalle_target._async_client.images.generate = AsyncMock(return_value=mock_return)

    response = await mock_dalle_target.send_prompt_async(prompt_request=request)
    file_path = str(response.request_pieces[0].converted_value)
    assert file_path
    assert file_path.endswith(".png")

    assert os.path.exists(file_path)

    with open(file_path, "rb") as file:
        data = file.read()
        assert data == b"test image data"

    os.remove(file_path)


@pytest.mark.asyncio
async def test_send_prompt_async_empty_response_adds_memory() -> None:

    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()
    request = PromptRequestPiece(
        role="user",
        original_value="draw me a test picture",
    ).to_prompt_request_response()

    mock_return = MagicMock()

    # b64_json with empty response
    mock_return.model_dump_json.return_value = '{"data": [{"b64_json": ""}]}'
    mock_dalle_target = OpenAIDALLETarget(deployment_name="test", endpoint="test", api_key="test")
    mock_dalle_target._async_client.images = MagicMock()
    mock_dalle_target._async_client.images.generate = AsyncMock(return_value=mock_return)
    mock_dalle_target._memory = mock_memory
    with pytest.raises(EmptyResponseException) as e:
        await mock_dalle_target.send_prompt_async(prompt_request=request)
    assert str(e.value) == "Status Code: 204, Message: The chat returned an empty response."


@pytest.mark.asyncio
async def test_send_prompt_async_rate_limit_adds_memory() -> None:

    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()
    request = PromptRequestPiece(
        role="user",
        original_value="draw me a test picture",
    ).to_prompt_request_response()

    mock_dalle_target = OpenAIDALLETarget(deployment_name="test", endpoint="test", api_key="test")
    mock_dalle_target._memory = mock_memory

    # mocking openai.RateLimitError
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_generate_image_response_async = AsyncMock(
        side_effect=RateLimitError("Rate Limit Reached", response=mock_resp, body="Rate limit reached")
    )
    setattr(mock_dalle_target, "_generate_image_response_async", mock_generate_image_response_async)
    with pytest.raises(RateLimitError) as rle:
        await mock_dalle_target.send_prompt_async(prompt_request=request)
        mock_dalle_target._memory.add_request_response_to_memory.assert_called_once()
    assert str(rle.value) == "Rate Limit Reached"


@pytest.mark.asyncio
async def test_send_prompt_async_bad_request_adds_memory() -> None:

    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()
    request = PromptRequestPiece(
        role="user",
        original_value="draw me a test picture",
    ).to_prompt_request_response()
    with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
        mock_dalle_target = OpenAIDALLETarget(deployment_name="test", endpoint="test", api_key="test")
        mock_dalle_target._memory = mock_memory

        # mocking openai.BadRequestError
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_generate_image_response_async = AsyncMock(
            side_effect=BadRequestError("Bad Request", response=mock_resp, body="Bad Request")
        )

        setattr(mock_dalle_target, "_generate_image_response_async", mock_generate_image_response_async)
        with pytest.raises(BadRequestError) as bre:
            await mock_dalle_target.send_prompt_async(prompt_request=request)
            mock_dalle_target._memory.add_request_response_to_memory.assert_called_once()
        assert str(bre.value) == "Bad Request"


def test_is_json_response_supported():
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()

    mock_dalle_target = OpenAIDALLETarget(deployment_name="test", endpoint="test", api_key="test")
    assert mock_dalle_target.is_json_response_supported() is False
