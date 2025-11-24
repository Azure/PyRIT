# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import base64
import json
import os
import uuid
from pathlib import Path
from typing import MutableSequence
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import httpx
import pytest
from unit.mocks import get_image_message_piece, get_sample_conversations

from pyrit.exceptions.exception_classes import (
    EmptyResponseException,
    RateLimitException,
)
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import Message, MessagePiece
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
def sample_conversations() -> MutableSequence[MessagePiece]:
    conversations = get_sample_conversations()
    return Message.flatten_to_message_pieces(conversations)


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
    sample_conversations: MutableSequence[MessagePiece],
    dalle_response_json: dict,
):
    request = sample_conversations[0]

    # Mock SDK response
    mock_response = MagicMock()
    mock_image = MagicMock()
    mock_image.b64_json = "aGVsbG8="  # Base64 encoded "hello"
    mock_response.data = [mock_image]

    with patch.object(
        dalle_target._async_client.images, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = mock_response

        resp = await dalle_target.send_prompt_async(message=Message([request]))
        assert resp
        path = resp.message_pieces[0].original_value
        assert os.path.isfile(path)

        with open(path, "r") as file:
            data = file.read()
            assert data == "hello"

        os.remove(path)


@pytest.mark.asyncio
async def test_send_prompt_async_empty_response(
    dalle_target: OpenAIDALLETarget,
    sample_conversations: MutableSequence[MessagePiece],
    dalle_response_json: dict,
):
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    # Mock SDK response with empty b64_json
    mock_response = MagicMock()
    mock_image = MagicMock()
    mock_image.b64_json = ""  # Empty response
    mock_response.data = [mock_image]

    with patch.object(
        dalle_target._async_client.images, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = mock_response

        with pytest.raises(EmptyResponseException):
            await dalle_target.send_prompt_async(message=Message([request]))


@pytest.mark.asyncio
async def test_send_prompt_async_rate_limit_exception(
    dalle_target: OpenAIDALLETarget, sample_conversations: MutableSequence[MessagePiece]
):
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    # Import SDK exception
    from openai import RateLimitError

    with patch.object(
        dalle_target._async_client.images, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.side_effect = RateLimitError(
            "Rate Limit Reached", response=MagicMock(), body={}
        )

        with pytest.raises(RateLimitException):
            await dalle_target.send_prompt_async(message=Message([request]))


@pytest.mark.asyncio
async def test_send_prompt_async_bad_request_error(
    dalle_target: OpenAIDALLETarget, sample_conversations: MutableSequence[MessagePiece]
):
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    # Import SDK exception
    from openai import BadRequestError

    mock_response = MagicMock()
    mock_response.text = '{"error": {"message": "Bad Request Error"}}'

    # Create exception with proper status_code
    bad_request_error = BadRequestError(
        "Bad Request Error", response=mock_response, body={"error": {"message": "Bad Request Error"}}
    )
    bad_request_error.status_code = 400

    with patch.object(
        dalle_target._async_client.images, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.side_effect = bad_request_error

        # Non-content-filter BadRequestError should be re-raised (same as chat target behavior)
        with pytest.raises(Exception):
            await dalle_target.send_prompt_async(message=Message([request]))


@pytest.mark.asyncio
async def test_dalle_validate_request_length(dalle_target: OpenAIDALLETarget):
    request = Message(
        message_pieces=[
            MessagePiece(role="user", conversation_id="123", original_value="test"),
            MessagePiece(role="user", conversation_id="123", original_value="test2"),
        ]
    )

    with pytest.raises(ValueError, match="This target only supports a single message piece."):
        await dalle_target.send_prompt_async(message=request)


@pytest.mark.asyncio
async def test_dalle_validate_prompt_type(dalle_target: OpenAIDALLETarget):
    request = Message(message_pieces=[get_image_message_piece()])
    with pytest.raises(ValueError, match="This target only supports text prompt input."):
        await dalle_target.send_prompt_async(message=request)


@pytest.mark.asyncio
async def test_send_prompt_async_empty_response_adds_memory(
    dalle_target: OpenAIDALLETarget,
    sample_conversations: MutableSequence[MessagePiece],
) -> None:

    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_message_to_memory = AsyncMock()

    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    # Mock SDK response with empty b64_json
    mock_response = MagicMock()
    mock_image = MagicMock()
    mock_image.b64_json = ""  # Empty response
    mock_response.data = [mock_image]

    with patch.object(
        dalle_target._async_client.images, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = mock_response
        dalle_target._memory = mock_memory

        with pytest.raises(EmptyResponseException):
            await dalle_target.send_prompt_async(message=Message([request]))


@pytest.mark.asyncio
async def test_send_prompt_async_rate_limit_adds_memory(
    dalle_target: OpenAIDALLETarget,
    sample_conversations: MutableSequence[MessagePiece],
) -> None:
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_message_to_memory = AsyncMock()

    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    # Import SDK exception
    from openai import RateLimitError

    with patch.object(
        dalle_target._async_client.images, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.side_effect = RateLimitError(
            "Rate Limit Reached", response=MagicMock(), body={}
        )
        dalle_target._memory = mock_memory

        with pytest.raises(RateLimitException):
            await dalle_target.send_prompt_async(message=Message([request]))


@pytest.mark.asyncio
async def test_send_prompt_async_bad_request_content_filter(
    dalle_target: OpenAIDALLETarget,
    sample_conversations: MutableSequence[MessagePiece],
) -> None:

    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    # Import SDK exception
    from openai import BadRequestError

    mock_response = MagicMock()
    mock_response.text = '{"error": {"code": "content_filter", "message": "Content filtered"}}'

    # Create exception with proper status_code
    bad_request_error = BadRequestError(
        "Bad Request Error",
        response=mock_response,
        body={"error": {"code": "content_filter", "message": "Content filtered"}}
    )
    bad_request_error.status_code = 400

    with patch.object(
        dalle_target._async_client.images, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.side_effect = bad_request_error
        result = await dalle_target.send_prompt_async(message=Message([request]))
        assert result.message_pieces[0].converted_value_data_type == "error"
        assert "content_filter" in result.message_pieces[0].converted_value


@pytest.mark.asyncio
async def test_send_prompt_async_bad_request_content_policy_violation(
    dalle_target: OpenAIDALLETarget,
    sample_conversations: MutableSequence[MessagePiece],
) -> None:

    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    # Import SDK exception
    from openai import BadRequestError

    mock_response = MagicMock()
    mock_response.text = '{"error": {"code": "content_policy_violation", "message": "Content blocked by policy"}}'

    # Create exception with proper status_code and inner_error structure
    bad_request_error = BadRequestError(
        "Content blocked by policy",
        response=mock_response,
        body={"error": {"code": "content_policy_violation", "message": "Content blocked by policy"}}
    )
    bad_request_error.status_code = 400

    with patch.object(
        dalle_target._async_client.images, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.side_effect = bad_request_error
        result = await dalle_target.send_prompt_async(message=Message([request]))
        assert result.message_pieces[0].response_error == "blocked"
        assert result.message_pieces[0].converted_value_data_type == "error"


def test_is_json_response_supported(patch_central_database):
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_message_to_memory = AsyncMock()

    mock_dalle_target = OpenAIDALLETarget(model_name="test", endpoint="test", api_key="test")
    assert mock_dalle_target.is_json_response_supported() is False
