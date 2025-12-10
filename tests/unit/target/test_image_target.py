# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import uuid
from typing import MutableSequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unit.mocks import get_image_message_piece, get_sample_conversations

from pyrit.exceptions.exception_classes import (
    EmptyResponseException,
    RateLimitException,
)
from pyrit.models import Message, MessagePiece
from pyrit.prompt_target import OpenAIImageTarget


@pytest.fixture
def image_target(patch_central_database) -> OpenAIImageTarget:
    return OpenAIImageTarget(
        model_name="dall-e-3",
        endpoint="test",
        api_key="test",
    )


@pytest.fixture
def image_response_json() -> dict:
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


def test_initialization_with_required_parameters(image_target: OpenAIImageTarget):
    assert image_target
    assert image_target._model_name == "dall-e-3"


@pytest.mark.asyncio
async def test_send_prompt_async(
    image_target: OpenAIImageTarget,
    sample_conversations: MutableSequence[MessagePiece],
    image_response_json: dict,
):
    request = sample_conversations[0]

    # Mock SDK response
    mock_response = MagicMock()
    mock_image = MagicMock()
    mock_image.b64_json = "aGVsbG8="  # Base64 encoded "hello"
    mock_response.data = [mock_image]

    with patch.object(image_target._async_client.images, "generate", new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_response

        resp = await image_target.send_prompt_async(message=Message([request]))
        assert len(resp) == 1
        assert resp
        path = resp[0].message_pieces[0].original_value
        assert os.path.isfile(path)

        with open(path, "r") as file:
            data = file.read()
            assert data == "hello"

        os.remove(path)


@pytest.mark.asyncio
async def test_send_prompt_async_empty_response(
    image_target: OpenAIImageTarget,
    sample_conversations: MutableSequence[MessagePiece],
    image_response_json: dict,
):
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    # Mock SDK response with empty b64_json
    mock_response = MagicMock()
    mock_image = MagicMock()
    mock_image.b64_json = ""  # Empty response
    mock_response.data = [mock_image]

    with patch.object(image_target._async_client.images, "generate", new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_response

        with pytest.raises(EmptyResponseException):
            await image_target.send_prompt_async(message=Message([request]))


@pytest.mark.asyncio
async def test_send_prompt_async_rate_limit_exception(
    image_target: OpenAIImageTarget, sample_conversations: MutableSequence[MessagePiece]
):
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    # Import SDK exception
    from openai import RateLimitError

    with patch.object(image_target._async_client.images, "generate", new_callable=AsyncMock) as mock_generate:
        mock_generate.side_effect = RateLimitError("Rate Limit Reached", response=MagicMock(), body={})

        with pytest.raises(RateLimitException):
            await image_target.send_prompt_async(message=Message([request]))


@pytest.mark.asyncio
async def test_send_prompt_async_bad_request_error(
    image_target: OpenAIImageTarget, sample_conversations: MutableSequence[MessagePiece]
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

    with patch.object(image_target._async_client.images, "generate", new_callable=AsyncMock) as mock_generate:
        mock_generate.side_effect = bad_request_error

        # Non-content-filter BadRequestError should be re-raised (same as chat target behavior)
        with pytest.raises(Exception):
            await image_target.send_prompt_async(message=Message([request]))


@pytest.mark.asyncio
async def test_image_validate_request_length(image_target: OpenAIImageTarget):
    request = Message(
        message_pieces=[
            MessagePiece(role="user", conversation_id="123", original_value="test"),
            MessagePiece(role="user", conversation_id="123", original_value="test2"),
        ]
    )

    with pytest.raises(ValueError, match="This target only supports a single message piece."):
        await image_target.send_prompt_async(message=request)


@pytest.mark.asyncio
async def test_image_validate_prompt_type(image_target: OpenAIImageTarget):
    request = Message(message_pieces=[get_image_message_piece()])
    with pytest.raises(ValueError, match="This target only supports text prompt input."):
        await image_target.send_prompt_async(message=request)


@pytest.mark.asyncio
async def test_send_prompt_async_empty_response_adds_memory(
    image_target: OpenAIImageTarget,
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

    with patch.object(image_target._async_client.images, "generate", new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_response
        image_target._memory = mock_memory

        with pytest.raises(EmptyResponseException):
            await image_target.send_prompt_async(message=Message([request]))


@pytest.mark.asyncio
async def test_send_prompt_async_rate_limit_adds_memory(
    image_target: OpenAIImageTarget,
    sample_conversations: MutableSequence[MessagePiece],
) -> None:
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_message_to_memory = AsyncMock()

    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    # Import SDK exception
    from openai import RateLimitError

    with patch.object(image_target._async_client.images, "generate", new_callable=AsyncMock) as mock_generate:
        mock_generate.side_effect = RateLimitError("Rate Limit Reached", response=MagicMock(), body={})
        image_target._memory = mock_memory

        with pytest.raises(RateLimitException):
            await image_target.send_prompt_async(message=Message([request]))


@pytest.mark.asyncio
async def test_send_prompt_async_bad_request_content_filter(
    image_target: OpenAIImageTarget,
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
        body={"error": {"code": "content_filter", "message": "Content filtered"}},
    )
    bad_request_error.status_code = 400

    with patch.object(image_target._async_client.images, "generate", new_callable=AsyncMock) as mock_generate:
        mock_generate.side_effect = bad_request_error
        result = await image_target.send_prompt_async(message=Message([request]))
        assert len(result) == 1
        assert result[0].message_pieces[0].converted_value_data_type == "error"
        assert "content_filter" in result[0].message_pieces[0].converted_value


@pytest.mark.asyncio
async def test_send_prompt_async_bad_request_content_policy_violation(
    image_target: OpenAIImageTarget,
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
        body={"error": {"code": "content_policy_violation", "message": "Content blocked by policy"}},
    )
    bad_request_error.status_code = 400

    with patch.object(image_target._async_client.images, "generate", new_callable=AsyncMock) as mock_generate:
        mock_generate.side_effect = bad_request_error
        result = await image_target.send_prompt_async(message=Message([request]))
        assert len(result) == 1
        assert result[0].message_pieces[0].response_error == "blocked"
        assert result[0].message_pieces[0].converted_value_data_type == "error"


def test_is_json_response_supported(patch_central_database):
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_message_to_memory = AsyncMock()

    mock_image_target = OpenAIImageTarget(model_name="test", endpoint="test", api_key="test")
    assert mock_image_target.is_json_response_supported() is False


@pytest.mark.asyncio
async def test_send_prompt_async_url_response_triggers_retry(
    image_target: OpenAIImageTarget,
    sample_conversations: MutableSequence[MessagePiece],
):
    """Test that when model returns URL instead of base64, it triggers retry with response_format."""
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    # First call returns URL (no b64_json)
    mock_response_url = MagicMock()
    mock_image_url = MagicMock()
    mock_image_url.b64_json = None
    mock_image_url.url = "https://example.com/image.png"
    mock_response_url.data = [mock_image_url]

    # Second call returns base64 (after retry with response_format)
    mock_response_b64 = MagicMock()
    mock_image_b64 = MagicMock()
    mock_image_b64.b64_json = "aGVsbG8="  # Base64 encoded "hello"
    mock_response_b64.data = [mock_image_b64]

    with patch.object(image_target._async_client.images, "generate", new_callable=AsyncMock) as mock_generate:
        # First call returns URL, second call returns base64
        mock_generate.side_effect = [mock_response_url, mock_response_b64]

        resp = await image_target.send_prompt_async(message=Message([request]))

        # Should have called generate twice (initial + retry)
        assert mock_generate.call_count == 2

        # First call should NOT have response_format
        first_call_kwargs = mock_generate.call_args_list[0][1]
        assert "response_format" not in first_call_kwargs

        # Second call should have response_format set to b64_json
        second_call_kwargs = mock_generate.call_args_list[1][1]
        assert second_call_kwargs["response_format"] == "b64_json"

        # Should have successfully returned the image
        assert len(resp) == 1
        path = resp[0].message_pieces[0].original_value
        assert os.path.isfile(path)

        with open(path, "r") as file:
            data = file.read()
            assert data == "hello"

        os.remove(path)


@pytest.mark.asyncio
async def test_send_prompt_async_url_response_sets_flag(
    image_target: OpenAIImageTarget,
    sample_conversations: MutableSequence[MessagePiece],
):
    """Test that _requires_response_format flag is set after detecting URL response."""
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    # Initially, flag should be False
    assert image_target._requires_response_format is False

    # First call returns URL (no b64_json)
    mock_response_url = MagicMock()
    mock_image_url = MagicMock()
    mock_image_url.b64_json = None
    mock_image_url.url = "https://example.com/image.png"
    mock_response_url.data = [mock_image_url]

    # Second call returns base64
    mock_response_b64 = MagicMock()
    mock_image_b64 = MagicMock()
    mock_image_b64.b64_json = "aGVsbG8="
    mock_response_b64.data = [mock_image_b64]

    with patch.object(image_target._async_client.images, "generate", new_callable=AsyncMock) as mock_generate:
        mock_generate.side_effect = [mock_response_url, mock_response_b64]
        await image_target.send_prompt_async(message=Message([request]))

        # After handling URL response, flag should be True
        assert image_target._requires_response_format is True


@pytest.mark.asyncio
async def test_send_prompt_async_uses_response_format_when_flag_set(
    image_target: OpenAIImageTarget,
    sample_conversations: MutableSequence[MessagePiece],
):
    """Test that once flag is set, response_format is always included."""
    request = sample_conversations[0]

    # Set the flag (simulating previous URL detection)
    image_target._requires_response_format = True

    mock_response = MagicMock()
    mock_image = MagicMock()
    mock_image.b64_json = "aGVsbG8="
    mock_response.data = [mock_image]

    with patch.object(image_target._async_client.images, "generate", new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_response
        await image_target.send_prompt_async(message=Message([request]))

        # Should include response_format in the call
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["response_format"] == "b64_json"
