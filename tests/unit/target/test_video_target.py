# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import uuid
from typing import MutableSequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unit.mocks import get_sample_conversations

from pyrit.exceptions import RateLimitException
from pyrit.models import Message, MessagePiece
from pyrit.prompt_target import OpenAIVideoTarget


@pytest.fixture
def sample_conversations() -> MutableSequence[MessagePiece]:
    conversations = get_sample_conversations()
    return Message.flatten_to_message_pieces(conversations)


@pytest.fixture
def video_target(patch_central_database) -> OpenAIVideoTarget:
    return OpenAIVideoTarget(
        endpoint="https://api.openai.com/v1",
        api_key="test",
        model_name="sora-2",
    )


def test_video_initializes(video_target: OpenAIVideoTarget):
    assert video_target
    assert video_target._model_name == "sora-2"


def test_video_initialization_invalid_resolution(patch_central_database):
    with pytest.raises(ValueError, match="Invalid resolution"):
        OpenAIVideoTarget(
            endpoint="https://api.openai.com/v1",
            api_key="test",
            resolution_dimensions="invalid",
        )


def test_video_initialization_invalid_duration(patch_central_database):
    with pytest.raises(ValueError, match="Invalid duration"):
        OpenAIVideoTarget(
            endpoint="https://api.openai.com/v1",
            api_key="test",
            n_seconds=20,  # Only 4, 8, 12 are supported
        )


def test_video_validate_request_length(video_target: OpenAIVideoTarget):
    with pytest.raises(ValueError, match="single message piece"):
        conversation_id = str(uuid.uuid4())
        msg1 = MessagePiece(
            role="user", original_value="test1", converted_value="test1", conversation_id=conversation_id
        )
        msg2 = MessagePiece(
            role="user", original_value="test2", converted_value="test2", conversation_id=conversation_id
        )
        video_target._validate_request(message=Message([msg1, msg2]))


def test_video_validate_prompt_type(video_target: OpenAIVideoTarget):
    with pytest.raises(ValueError, match="text prompt input"):
        msg = MessagePiece(
            role="user", original_value="test", converted_value="test", converted_value_data_type="image_path"
        )
        video_target._validate_request(message=Message([msg]))


def test_is_json_response_supported(patch_central_database):
    target = OpenAIVideoTarget(endpoint="test", api_key="test")
    assert target.is_json_response_supported() is False


@pytest.mark.asyncio
async def test_video_send_prompt_async_success(
    video_target: OpenAIVideoTarget, sample_conversations: MutableSequence[MessagePiece]
):
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    # Mock successful video generation
    mock_video = MagicMock()
    mock_video.id = "video_123"
    mock_video.status = "completed"
    mock_video.error = None

    # Mock video content as HttpxBinaryResponseContent
    mock_video_response = MagicMock()
    mock_video_response.content = b"video data content"

    # Mock data serializer
    mock_serializer = MagicMock()
    mock_serializer.value = "/path/to/video.mp4"
    mock_serializer.save_data = AsyncMock()

    with (
        patch.object(video_target._async_client.videos, "create_and_poll", new_callable=AsyncMock) as mock_create,
        patch.object(video_target._async_client.videos, "download_content", new_callable=AsyncMock) as mock_download,
        patch("pyrit.prompt_target.openai.openai_video_target.data_serializer_factory") as mock_factory,
    ):
        mock_create.return_value = mock_video
        mock_download.return_value = mock_video_response
        mock_factory.return_value = mock_serializer

        response = await video_target.send_prompt_async(message=Message([request]))

        # Verify SDK methods were called correctly
        mock_create.assert_called_once_with(
            model="sora-2",
            prompt="Hello, how are you?",
            size="1280x720",
            seconds="4",
        )
        mock_download.assert_called_once_with("video_123")
        mock_serializer.save_data.assert_called_once_with(data=b"video data content")

        # Verify response
        assert len(response) == 1
        assert len(response[0].message_pieces) == 1
        assert response[0].message_pieces[0].converted_value == "/path/to/video.mp4"
        assert response[0].message_pieces[0].converted_value_data_type == "video_path"


@pytest.mark.asyncio
async def test_video_send_prompt_async_failed_content_filter(
    video_target: OpenAIVideoTarget, sample_conversations: MutableSequence[MessagePiece]
):
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    # Mock failed video generation with output-side content filter
    mock_video = MagicMock()
    mock_video.id = "video_456"
    mock_video.status = "failed"
    mock_error = MagicMock()
    mock_error.code = "content_filter"
    mock_video.error = mock_error
    video_response_dict = {
        "id": "video_456",
        "status": "failed",
        "error": {"code": "content_filter"},
    }
    mock_video.model_dump.return_value = video_response_dict
    mock_video.model_dump_json.return_value = json.dumps(video_response_dict)

    with patch.object(video_target._async_client.videos, "create_and_poll", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_video

        response = await video_target.send_prompt_async(message=Message([request]))

        # Verify response is error with blocked status
        assert len(response) == 1
        assert len(response[0].message_pieces) == 1
        assert response[0].message_pieces[0].response_error == "blocked"
        assert (
            "content_filter" in response[0].message_pieces[0].converted_value.lower()
            or "blocked" in response[0].message_pieces[0].converted_value.lower()
        )


@pytest.mark.asyncio
async def test_video_send_prompt_async_failed_processing_error(
    video_target: OpenAIVideoTarget, sample_conversations: MutableSequence[MessagePiece]
):
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    # Mock failed video generation with processing error
    mock_video = MagicMock()
    mock_video.id = "video_789"
    mock_video.status = "failed"
    mock_error = MagicMock()
    mock_error.code = "internal_error"
    mock_video.error = mock_error

    with patch.object(video_target._async_client.videos, "create_and_poll", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_video

        response = await video_target.send_prompt_async(message=Message([request]))

        # Verify response is processing error
        assert len(response) == 1
        assert len(response[0].message_pieces) == 1
        assert response[0].message_pieces[0].response_error == "processing"


@pytest.mark.asyncio
async def test_video_send_prompt_async_bad_request_exception(
    video_target: OpenAIVideoTarget, sample_conversations: MutableSequence[MessagePiece]
):
    from openai import BadRequestError

    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    # Mock BadRequestError with content filter
    mock_response = MagicMock()
    mock_response.text = '{"error": {"code": "content_policy_violation", "message": "Content blocked"}}'

    bad_request_error = BadRequestError(
        "Content blocked",
        response=mock_response,
        body={"error": {"code": "content_policy_violation", "message": "Content blocked"}},
    )
    bad_request_error.status_code = 400

    with patch.object(video_target._async_client.videos, "create_and_poll", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = bad_request_error

        response = await video_target.send_prompt_async(message=Message([request]))

        # Verify response is error with blocked status (content filter)
        assert len(response) == 1
        assert len(response[0].message_pieces) == 1
        assert response[0].message_pieces[0].response_error == "blocked"


@pytest.mark.asyncio
async def test_video_send_prompt_async_rate_limit_exception(
    video_target: OpenAIVideoTarget, sample_conversations: MutableSequence[MessagePiece]
):
    from openai import RateLimitError

    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    # Mock RateLimitError
    mock_response = MagicMock()
    mock_response.text = "Rate limit exceeded"

    rate_limit_error = RateLimitError("Rate limit exceeded", response=mock_response, body={})

    with patch.object(video_target._async_client.videos, "create_and_poll", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = rate_limit_error

        with pytest.raises(RateLimitException):
            await video_target.send_prompt_async(message=Message([request]))


@pytest.mark.asyncio
async def test_video_send_prompt_async_api_error(
    video_target: OpenAIVideoTarget, sample_conversations: MutableSequence[MessagePiece]
):
    from openai import APIStatusError

    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    # Mock APIStatusError
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal server error"

    api_error = APIStatusError("Internal server error", response=mock_response, body={})

    with patch.object(video_target._async_client.videos, "create_and_poll", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = api_error

        with pytest.raises(APIStatusError):
            await video_target.send_prompt_async(message=Message([request]))


@pytest.mark.asyncio
async def test_video_send_prompt_async_unexpected_status(
    video_target: OpenAIVideoTarget, sample_conversations: MutableSequence[MessagePiece]
):
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    # Mock video with unexpected status
    mock_video = MagicMock()
    mock_video.id = "video_unexpected"
    mock_video.status = "pending"  # Unexpected status
    mock_video.error = None

    with patch.object(video_target._async_client.videos, "create_and_poll", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_video

        response = await video_target.send_prompt_async(message=Message([request]))

        # Verify response is error with unknown status
        assert len(response) == 1
        assert len(response[0].message_pieces) == 1
        assert response[0].message_pieces[0].response_error == "unknown"
        assert "unexpected status: pending" in response[0].message_pieces[0].converted_value


# Unit tests for override methods


def test_check_content_filter_detects_output_content_filter(video_target: OpenAIVideoTarget):
    """Test _check_content_filter detects output-side content_filter error."""
    mock_video = MagicMock()
    mock_video.status = "failed"
    mock_error = MagicMock()
    mock_error.code = "content_filter"
    mock_video.error = mock_error
    mock_video.model_dump.return_value = {"error": {"code": "content_filter"}}

    assert video_target._check_content_filter(mock_video) is True


def test_check_content_filter_detects_moderation_blocked(video_target: OpenAIVideoTarget):
    """Test _check_content_filter detects moderation_blocked error."""
    mock_video = MagicMock()
    mock_video.status = "failed"
    mock_error = MagicMock()
    mock_error.code = "moderation_blocked"
    mock_video.error = mock_error
    mock_video.model_dump.return_value = {"error": {"code": "moderation_blocked"}}

    assert video_target._check_content_filter(mock_video) is True


def test_check_content_filter_completed_status(video_target: OpenAIVideoTarget):
    """Test _check_content_filter returns False for completed videos."""
    mock_video = MagicMock()
    mock_video.status = "completed"
    mock_video.error = None

    assert video_target._check_content_filter(mock_video) is False


def test_check_content_filter_different_error(video_target: OpenAIVideoTarget):
    """Test _check_content_filter returns False for non-content-filter errors."""
    mock_video = MagicMock()
    mock_video.status = "failed"
    mock_error = MagicMock()
    mock_error.code = "processing_error"
    mock_video.error = mock_error
    mock_video.model_dump.return_value = {"error": {"code": "processing_error"}}

    assert video_target._check_content_filter(mock_video) is False


def test_check_content_filter_no_error_object(video_target: OpenAIVideoTarget):
    """Test _check_content_filter returns False when no error object."""
    mock_video = MagicMock()
    mock_video.status = "failed"
    mock_video.error = None

    assert video_target._check_content_filter(mock_video) is False
