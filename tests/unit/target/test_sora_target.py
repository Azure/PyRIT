# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from typing import MutableSequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unit.mocks import get_sample_conversations

from pyrit.exceptions import RateLimitException
from pyrit.models import Message, MessagePiece
from pyrit.prompt_target import OpenAISoraTarget


@pytest.fixture
def sample_conversations() -> MutableSequence[MessagePiece]:
    conversations = get_sample_conversations()
    return Message.flatten_to_message_pieces(conversations)


@pytest.fixture
def sora_target(patch_central_database) -> OpenAISoraTarget:
    return OpenAISoraTarget(
        endpoint="https://api.openai.com/v1",
        api_key="test",
        model_name="sora-2",
    )


def test_sora_initializes(sora_target: OpenAISoraTarget):
    assert sora_target
    assert sora_target._model_name == "sora-2"


def test_sora_initialization_invalid_resolution(patch_central_database):
    with pytest.raises(ValueError, match="Invalid resolution"):
        OpenAISoraTarget(
            endpoint="https://api.openai.com/v1",
            api_key="test",
            resolution_dimensions="invalid",
        )


def test_sora_initialization_invalid_duration(patch_central_database):
    with pytest.raises(ValueError, match="Invalid duration"):
        OpenAISoraTarget(
            endpoint="https://api.openai.com/v1",
            api_key="test",
            n_seconds=20,  # Only 4, 8, 12 are supported
        )


def test_sora_validate_request_length(sora_target: OpenAISoraTarget):
    with pytest.raises(ValueError, match="single message piece"):
        conversation_id = str(uuid.uuid4())
        msg1 = MessagePiece(role="user", original_value="test1", converted_value="test1", conversation_id=conversation_id)
        msg2 = MessagePiece(role="user", original_value="test2", converted_value="test2", conversation_id=conversation_id)
        sora_target._validate_request(message=Message([msg1, msg2]))


def test_sora_validate_prompt_type(sora_target: OpenAISoraTarget):
    with pytest.raises(ValueError, match="text prompt input"):
        msg = MessagePiece(
            role="user",
            original_value="test",
            converted_value="test",
            converted_value_data_type="image_path"
        )
        sora_target._validate_request(message=Message([msg]))


def test_is_json_response_supported(patch_central_database):
    target = OpenAISoraTarget(endpoint="test", api_key="test")
    assert target.is_json_response_supported() is False


@pytest.mark.asyncio
async def test_sora_send_prompt_async_success(
    sora_target: OpenAISoraTarget, sample_conversations: MutableSequence[MessagePiece]
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
    
    with patch.object(
        sora_target._async_client.videos, "create_and_poll", new_callable=AsyncMock
    ) as mock_create, \
    patch.object(
        sora_target._async_client.videos, "download_content", new_callable=AsyncMock
    ) as mock_download, \
    patch("pyrit.prompt_target.openai.openai_sora_target.data_serializer_factory") as mock_factory:
        mock_create.return_value = mock_video
        mock_download.return_value = mock_video_response
        mock_factory.return_value = mock_serializer
        
        response = await sora_target.send_prompt_async(message=Message([request]))
        
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
        assert len(response.message_pieces) == 1
        assert response.message_pieces[0].converted_value == "/path/to/video.mp4"
        assert response.message_pieces[0].converted_value_data_type == "video_path"


@pytest.mark.asyncio
async def test_sora_send_prompt_async_failed_content_filter(
    sora_target: OpenAISoraTarget, sample_conversations: MutableSequence[MessagePiece]
):
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())
    
    # Mock failed video generation with content filter
    mock_video = MagicMock()
    mock_video.id = "video_456"
    mock_video.status = "failed"
    mock_error = MagicMock()
    mock_error.code = "input_moderation"
    mock_error.__str__ = lambda self: "Content policy violation"
    mock_video.error = mock_error
    
    with patch.object(
        sora_target._async_client.videos, "create_and_poll", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_video
        
        response = await sora_target.send_prompt_async(message=Message([request]))
        
        # Verify response is error with blocked status
        assert len(response.message_pieces) == 1
        assert response.message_pieces[0].response_error == "blocked"
        assert "Content policy violation" in response.message_pieces[0].converted_value


@pytest.mark.asyncio
async def test_sora_send_prompt_async_failed_processing_error(
    sora_target: OpenAISoraTarget, sample_conversations: MutableSequence[MessagePiece]
):
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())
    
    # Mock failed video generation with processing error
    mock_video = MagicMock()
    mock_video.id = "video_789"
    mock_video.status = "failed"
    mock_error = MagicMock()
    mock_error.code = "internal_error"
    mock_error.__str__ = lambda self: "Internal processing error"
    mock_video.error = mock_error
    
    with patch.object(
        sora_target._async_client.videos, "create_and_poll", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_video
        
        response = await sora_target.send_prompt_async(message=Message([request]))
        
        # Verify response is processing error
        assert len(response.message_pieces) == 1
        assert response.message_pieces[0].response_error == "processing"
        assert "Internal processing error" in response.message_pieces[0].converted_value


@pytest.mark.asyncio
async def test_sora_send_prompt_async_bad_request_exception(
    sora_target: OpenAISoraTarget, sample_conversations: MutableSequence[MessagePiece]
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
        body={"error": {"code": "content_policy_violation", "message": "Content blocked"}}
    )
    bad_request_error.status_code = 400
    
    with patch.object(
        sora_target._async_client.videos, "create_and_poll", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = bad_request_error
        
        response = await sora_target.send_prompt_async(message=Message([request]))
        
        # Verify response is error with blocked status (content filter)
        assert len(response.message_pieces) == 1
        assert response.message_pieces[0].response_error == "blocked"


@pytest.mark.asyncio
async def test_sora_send_prompt_async_rate_limit_exception(
    sora_target: OpenAISoraTarget, sample_conversations: MutableSequence[MessagePiece]
):
    from openai import RateLimitError
    
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())
    
    # Mock RateLimitError
    mock_response = MagicMock()
    mock_response.text = "Rate limit exceeded"
    
    rate_limit_error = RateLimitError("Rate limit exceeded", response=mock_response, body={})
    
    with patch.object(
        sora_target._async_client.videos, "create_and_poll", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = rate_limit_error
        
        with pytest.raises(RateLimitException):
            await sora_target.send_prompt_async(message=Message([request]))


@pytest.mark.asyncio
async def test_sora_send_prompt_async_api_error(
    sora_target: OpenAISoraTarget, sample_conversations: MutableSequence[MessagePiece]
):
    from openai import APIStatusError
    
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())
    
    # Mock APIStatusError
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal server error"
    
    api_error = APIStatusError("Internal server error", response=mock_response, body={})
    
    with patch.object(
        sora_target._async_client.videos, "create_and_poll", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = api_error
        
        with pytest.raises(APIStatusError):
            await sora_target.send_prompt_async(message=Message([request]))


@pytest.mark.asyncio
async def test_sora_send_prompt_async_unexpected_status(
    sora_target: OpenAISoraTarget, sample_conversations: MutableSequence[MessagePiece]
):
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())
    
    # Mock video with unexpected status
    mock_video = MagicMock()
    mock_video.id = "video_unexpected"
    mock_video.status = "pending"  # Unexpected status
    mock_video.error = None
    
    with patch.object(
        sora_target._async_client.videos, "create_and_poll", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_video
        
        response = await sora_target.send_prompt_async(message=Message([request]))
        
        # Verify response is error with unknown status
        assert len(response.message_pieces) == 1
        assert response.message_pieces[0].response_error == "unknown"
        assert "unexpected status: pending" in response.message_pieces[0].converted_value
