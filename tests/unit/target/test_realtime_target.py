# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import re
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.frames import Close

from pyrit.exceptions.exception_classes import ServerErrorException
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target import RealtimeTarget
from pyrit.prompt_target.openai.openai_realtime_target import RealtimeTargetResult


@pytest.fixture
def target(duckdb_instance):
    return RealtimeTarget(api_key="test_key", endpoint="wss://test_url", model_name="test", api_version="v1")


@pytest.fixture
def target_with_aad(duckdb_instance):
    target = RealtimeTarget(endpoint="wss://test_url", api_key="test_api_key")
    target._azure_auth = MagicMock()
    target._azure_auth.refresh_token = MagicMock(return_value="test_access_token")
    return target


@pytest.mark.asyncio
async def test_connect_success(target):
    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
        await target.connect()
        mock_connect.assert_called_once_with(
            "wss://test_url?deployment=test&OpenAI-Beta=realtime%3Dv1&api-key=test_key&api-version=v1"
        )
    await target.cleanup_target()


@pytest.mark.asyncio
async def test_send_prompt_async(target):
    # Mock the necessary methods
    target.connect = AsyncMock(return_value=AsyncMock())
    target.send_config = AsyncMock()
    result = RealtimeTargetResult(audio_bytes=b"file", transcripts=["hello"])
    target.send_text_async = AsyncMock(return_value=("output.wav", result))
    target.set_system_prompt = MagicMock()

    # Create a mock PromptRequestResponse with a valid data type
    request_piece = PromptRequestPiece(
        original_value="Hello",
        original_value_data_type="text",
        converted_value="Hello",
        converted_value_data_type="text",
        role="user",
        conversation_id="test_conversation_id",
    )
    prompt_request = PromptRequestResponse(request_pieces=[request_piece])

    # Call the send_prompt_async method
    response = await target.send_prompt_async(prompt_request=prompt_request)

    assert response

    target.send_text_async.assert_called_once_with(
        text="Hello",
        conversation_id="test_conversation_id",
    )
    assert response.get_value() == "hello"
    assert response.get_value(1) == "output.wav"

    # Clean up the WebSocket connections
    await target.cleanup_target()


@pytest.mark.asyncio
async def test_send_prompt_async_adds_system_prompt_to_memory(target):

    # Mock the necessary methods
    target.connect = AsyncMock(return_value=AsyncMock())
    target.send_config = AsyncMock()
    result = RealtimeTargetResult(audio_bytes=b"event1", transcripts=["event2"])
    target.send_text_async = AsyncMock(return_value=("output_audio_path", result))
    target.set_system_prompt = MagicMock()

    # Create a mock PromptRequestResponse with a valid data type
    request_piece = PromptRequestPiece(
        original_value="Hello",
        original_value_data_type="text",
        converted_value="Hello",
        converted_value_data_type="text",
        role="user",
        conversation_id="new_conversation_id",
    )
    prompt_request = PromptRequestResponse(request_pieces=[request_piece])

    # Call the send_prompt_async method
    await target.send_prompt_async(prompt_request=prompt_request)

    # Assert that set_system_prompt was called with the correct arguments
    target.set_system_prompt.assert_called_once_with(
        system_prompt=target.system_prompt,
        conversation_id="new_conversation_id",
        orchestrator_identifier=target.get_identifier(),
    )

    # Assert that the system_prompt is the default value
    assert target.system_prompt == "You are a helpful AI assistant"

    await target.cleanup_target()


@pytest.mark.asyncio
async def test_multiple_websockets_created_for_multiple_conversations(target):
    # Mock the necessary methods
    target.connect = AsyncMock(return_value=AsyncMock())
    target.send_config = AsyncMock()
    result = RealtimeTargetResult(audio_bytes=b"event1", transcripts=["event2"])
    target.send_text_async = AsyncMock(return_value=("output_audio_path", result))
    target.set_system_prompt = MagicMock()

    # Create mock PromptRequestResponses for two different conversations
    request_piece_1 = PromptRequestPiece(
        original_value="Hello",
        original_value_data_type="text",
        converted_value="Hello",
        converted_value_data_type="text",
        role="user",
        conversation_id="conversation_1",
    )
    prompt_request_1 = PromptRequestResponse(request_pieces=[request_piece_1])

    request_piece_2 = PromptRequestPiece(
        original_value="Hi",
        original_value_data_type="text",
        converted_value="Hi",
        converted_value_data_type="text",
        role="user",
        conversation_id="conversation_2",
    )
    prompt_request_2 = PromptRequestResponse(request_pieces=[request_piece_2])

    # Call the send_prompt_async method for both conversations
    await target.send_prompt_async(prompt_request=prompt_request_1)
    await target.send_prompt_async(prompt_request=prompt_request_2)

    # Assert that two different WebSocket connections were created
    assert "conversation_1" in target._existing_conversation
    assert "conversation_2" in target._existing_conversation

    # Clean up the WebSocket connections
    await target.cleanup_target()
    assert target._existing_conversation == {}


@pytest.mark.asyncio
async def test_send_prompt_async_invalid_request(target):

    # Create a mock PromptRequestResponse with an invalid data type
    request_piece = PromptRequestPiece(
        original_value="Invalid",
        original_value_data_type="image_path",
        converted_value="Invalid",
        converted_value_data_type="image_path",
        role="user",
    )
    prompt_request = PromptRequestResponse(request_pieces=[request_piece])
    with pytest.raises(ValueError) as excinfo:
        target._validate_request(prompt_request=prompt_request)

    assert "This target only supports text and audio_path prompt input." == str(excinfo.value)


@pytest.mark.asyncio
async def test_realtime_target_no_api_version(target):
    target._api_version = None  # No API version set
    target._existing_conversation.clear()  # Ensure no conversation exists

    # Mock necessary methods
    target.send_config = AsyncMock()
    target.set_system_prompt = MagicMock()
    result = RealtimeTargetResult(audio_bytes=b"file", transcripts=["hello"])
    target.send_text_async = AsyncMock(return_value=("output.wav", result))

    with patch("websockets.connect", new_callable=AsyncMock) as mock_websocket_connect:
        mock_websocket = AsyncMock()
        mock_websocket_connect.return_value = mock_websocket

        # Create a mock request
        request_piece = PromptRequestPiece(
            original_value="Hello",
            original_value_data_type="text",
            converted_value="Hello",
            converted_value_data_type="text",
            role="user",
            conversation_id="test_conversation_id",
        )
        prompt_request = PromptRequestResponse(request_pieces=[request_piece])

        # Call the method
        response = await target.send_prompt_async(prompt_request=prompt_request)

        assert response

        # Ensure `websockets.connect()` was called and capture the WebSocket URL
        mock_websocket_connect.assert_called_once()
        called_url = mock_websocket_connect.call_args[0][0]

        # Parse the query parameters from the URL
        parsed_url = urlparse(called_url)
        query_params = parse_qs(parsed_url.query)

        # Ensure API version is NOT in the request
        assert "api-version" not in query_params


@pytest.mark.asyncio
async def test_realtime_target_default_api_version(target):
    # Explicitly set default API version
    target._api_version = "2024-06-01"

    # Ensure no conversation exists
    target._existing_conversation.clear()

    # Mock necessary methods
    target.send_config = AsyncMock()
    target.set_system_prompt = MagicMock()

    result = RealtimeTargetResult(audio_bytes=b"file", transcripts=["hello"])
    target.send_text_async = AsyncMock(return_value=("output.wav", result))

    with patch("websockets.connect", new_callable=AsyncMock) as mock_websocket_connect:
        mock_websocket = AsyncMock()
        mock_websocket_connect.return_value = mock_websocket

        # Create a mock request
        request_piece = PromptRequestPiece(
            original_value="Hello",
            original_value_data_type="text",
            converted_value="Hello",
            converted_value_data_type="text",
            role="user",
            conversation_id="test_conversation_id",
        )
        prompt_request = PromptRequestResponse(request_pieces=[request_piece])

        # Call the method
        response = await target.send_prompt_async(prompt_request=prompt_request)

        assert response

        # Ensure `websockets.connect()` was called and capture the WebSocket URL
        mock_websocket_connect.assert_called_once()
        called_url = mock_websocket_connect.call_args[0][0]

        # Parse the query parameters from the URL
        parsed_url = urlparse(called_url)
        query_params = parse_qs(parsed_url.query)

        # Ensure API version IS in the request
        assert "api-version" in query_params
        assert query_params["api-version"][0] == "2024-06-01"


def test_add_auth_param_to_query_params_with_api_key(target_with_aad):
    query_params = {}
    target_with_aad._add_auth_param_to_query_params(query_params)
    assert query_params["api-key"] == "test_api_key"


def test_add_auth_param_to_query_params_with_azure_auth(target_with_aad):
    query_params = {}
    target_with_aad._add_auth_param_to_query_params(query_params)
    assert query_params["access_token"] == "test_access_token"


def test_add_auth_param_to_query_params_with_both_auth_methods(target_with_aad):
    query_params = {}
    target_with_aad._add_auth_param_to_query_params(query_params)
    assert query_params["api-key"] == "test_api_key"
    assert query_params["access_token"] == "test_access_token"


@pytest.mark.asyncio
async def test_receive_events_empty_output(target: RealtimeTarget):
    """Test handling of response.done event with empty output array."""
    mock_websocket = AsyncMock()
    conversation_id = "test_empty_output"
    target._existing_conversation[conversation_id] = mock_websocket

    empty_output_response = {
        "type": "response.done",
        "event_id": "event_123",
        "response": {
            "status": "failed",
            "status_details": {
                "type": "failed",
                "error": {"type": "server_error", "message": "The server had an error processing your request"},
            },
            "output": [],
            "conversation_id": conversation_id,
        },
    }

    # mock websocket yield our test response
    mock_websocket.__aiter__.return_value = [json.dumps(empty_output_response)]
    with pytest.raises(ServerErrorException, match="The server had an error processing your request"):
        await target.receive_events(conversation_id)


@pytest.mark.asyncio
async def test_receive_events_missing_content(target):
    """Test handling of response.done event with output but missing content."""
    mock_websocket = AsyncMock()
    conversation_id = "test_missing_content"
    target._existing_conversation[conversation_id] = mock_websocket

    missing_content_response = {
        "type": "response.done",
        "event_id": "event_456",
        "response": {
            "status": "success",
            "output": [
                {
                    # missing 'content' field
                    "type": "text"
                }
            ],
            "conversation_id": conversation_id,
        },
    }

    # mock websocket to yield test response
    mock_websocket.__aiter__.return_value = [json.dumps(missing_content_response)]
    first_output = missing_content_response["response"]["output"][0]
    with pytest.raises(ValueError, match=re.escape(f"Missing or invalid 'content' in: {first_output}")):
        await target.receive_events(conversation_id)


@pytest.mark.asyncio
async def test_receive_events_missing_transcript(target):
    """Test handling of response.done event with content but missing transcript."""
    mock_websocket = AsyncMock()
    conversation_id = "test_missing_transcript"
    target._existing_conversation[conversation_id] = mock_websocket

    missing_transcript_response = {
        "type": "response.done",
        "event_id": "event_789",
        "response": {
            "status": "success",
            "output": [
                {
                    "content": [
                        {
                            # missing 'transcript' field
                            "type": "text"
                        }
                    ]
                }
            ],
            "conversation_id": conversation_id,
        },
    }

    # mock websocket to yield test response
    mock_websocket.__aiter__.return_value = [json.dumps(missing_transcript_response)]

    content = missing_transcript_response["response"]["output"][0]["content"]
    with pytest.raises(ValueError, match=f"Missing 'transcript' in: {content}"):
        await target.receive_events(conversation_id)


@pytest.mark.asyncio
async def test_receive_events_audio_buffer_only(target):
    """Test receiving only audio data with no transcript."""
    mock_websocket = AsyncMock()
    conversation_id = "test_audio_only"
    target._existing_conversation[conversation_id] = mock_websocket

    # create audio delta and done events with no transcript
    audio_delta_event = {
        "type": "response.audio.delta",
        # base64 encoded "dummyaudio"
        "delta": "ZHVtbXlhdWRpbw==",
    }

    audio_done_event = {"type": "response.audio.done", "event_id": "event_abc"}

    # mock websocket to yield both events
    mock_websocket.__aiter__.return_value = [json.dumps(audio_delta_event), json.dumps(audio_done_event)]

    result = await target.receive_events(conversation_id)

    # it should have the audio buffer in the result
    assert len(result.transcripts) == 0
    assert result.audio_bytes == b"dummyaudio"


@pytest.mark.asyncio
async def test_receive_events_error_event(target):
    """Test handling of direct error event."""
    mock_websocket = AsyncMock()
    conversation_id = "test_error_event"
    target._existing_conversation[conversation_id] = mock_websocket

    error_event = {"type": "error", "error": {"type": "invalid_request_error", "message": "Invalid request"}}

    # mock websocket to yield test response
    mock_websocket.__aiter__.return_value = [json.dumps(error_event)]

    result = await target.receive_events(conversation_id)
    assert len(result.transcripts) == 0
    assert result.audio_bytes == b""


@pytest.mark.asyncio
async def test_receive_events_connection_closed(target):
    """Test handling of WebSocket connection closing unexpectedly."""
    mock_websocket = AsyncMock()
    conversation_id = "test_connection_closed"
    target._existing_conversation[conversation_id] = mock_websocket

    # create Close objects for the rcvd and sent parameters
    close_frame = Close(1000, "Normal closure")

    # forcing the websocket to raise a ConnectionClosed when iterated
    class FailingAsyncIterator:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise ConnectionClosed(rcvd=close_frame, sent=None)

    mock_websocket.__aiter__.side_effect = lambda: FailingAsyncIterator()
    result = await target.receive_events(conversation_id)
    assert len(result.transcripts) == 0
    assert result.audio_bytes == b""


@pytest.mark.asyncio
async def test_receive_events_with_audio_and_transcript(target):
    """Test successful processing of both audio data and transcript."""
    mock_websocket = AsyncMock()
    conversation_id = "test_success"
    target._existing_conversation[conversation_id] = mock_websocket

    audio_delta_event = {
        "type": "response.audio.delta",
        # base64 encoded "dummyaudio"
        "delta": "ZHVtbXlhdWRpbw==",
    }

    audio_done_event = {"type": "response.audio.done", "event_id": "event_def"}

    transcript_event = {
        "type": "response.done",
        "event_id": "event_ghi",
        "response": {
            "status": "success",
            "output": [{"content": [{"transcript": "Hello, this is a test transcript."}]}],
            "conversation_id": conversation_id,
        },
    }

    # mock websocket to yield all events
    mock_websocket.__aiter__.return_value = [
        json.dumps(audio_delta_event),
        json.dumps(audio_done_event),
        json.dumps(transcript_event),
    ]

    result = await target.receive_events(conversation_id)

    # result should have both the audio buffer and transcript
    assert len(result.transcripts) == 1
    assert result.audio_bytes == b"dummyaudio"
    assert result.transcripts[0] == "Hello, this is a test transcript."
