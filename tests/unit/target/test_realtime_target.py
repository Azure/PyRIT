# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target import RealtimeTarget


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
    target.send_text_async = AsyncMock(return_value=("output.wav", ["file", "hello"]))
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
    target.send_text_async = AsyncMock(return_value=("output_audio_path", ["event1", "event2"]))
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
    target.send_text_async = AsyncMock(return_value=("output_audio_path", ["event1", "event2"]))
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
    target.send_text_async = AsyncMock(return_value=("output.wav", ["file", "hello"]))

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
    target.send_text_async = AsyncMock(return_value=("output.wav", ["file", "hello"]))

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
