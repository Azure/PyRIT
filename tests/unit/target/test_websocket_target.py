# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from typing import Callable, List
from unittest.mock import AsyncMock, patch

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.frames import Close

from pyrit.models import Message, MessagePiece
from pyrit.prompt_target.websocket_target import WebsocketTarget


@pytest.fixture
def mock_initialization_strings() -> List[str]:
    return ["connect_message", "authenticate_message"]


@pytest.fixture
def mock_response_parser() -> Callable:
    def response_parser(text: str):
        json_body = json.loads(text)
        if "message" in json_body.keys():
            return json_body["message"]

    return response_parser


@pytest.fixture
def mock_message_builder() -> Callable:
    def message_builder(prompt: str):
        message_format = f"""{{"message":"{{PROMPT}}"}}"""

        message_w_prompt = message_format.replace("{PROMPT}", prompt)
        return message_w_prompt

    return message_builder


@pytest.fixture
def mock_websocket_target(
    mock_initialization_strings, mock_response_parser, mock_message_builder, sqlite_instance
) -> WebsocketTarget:
    endpoint = "wss://example.com"
    return WebsocketTarget(
        endpoint=endpoint,
        initialization_strings=mock_initialization_strings,
        response_parser=mock_response_parser,
        message_builder=mock_message_builder,
    )


@pytest.mark.asyncio
async def test_connect_success(mock_websocket_target):
    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
        await mock_websocket_target.connect()
        mock_connect.assert_called_once_with(uri="wss://example.com")
    await mock_websocket_target.cleanup_target()


@pytest.mark.asyncio
async def test_connect_success_w_kwargs():
    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
        # Create target with websockets.connect() keyword argument "proxy"
        target = WebsocketTarget(
            endpoint="wss://example.com",
            initialization_strings=mock_initialization_strings,
            response_parser=mock_response_parser,
            message_builder=mock_message_builder,
            proxy="http://example.proxy.com",
        )
        await target.connect()
        mock_connect.assert_called_once_with(uri="wss://example.com", proxy="http://example.proxy.com")
    await target.cleanup_target()


@pytest.mark.asyncio
async def test_send_prompt_async(mock_websocket_target):
    # Mock the necessary methods
    mock_websocket_target.connect = AsyncMock(return_value=AsyncMock())
    result = "Hi!"
    mock_websocket_target.send_text_async = AsyncMock(return_value=result)

    # Create a mock Message with a valid data type
    message_piece = MessagePiece(
        original_value="Hello",
        original_value_data_type="text",
        converted_value="Hello",
        converted_value_data_type="text",
        role="user",
        conversation_id="test_conversation_id",
    )
    message = Message(message_pieces=[message_piece])

    # Call the send_prompt_async method
    response = await mock_websocket_target.send_prompt_async(message=message)

    assert len(response) == 1
    assert response

    mock_websocket_target.send_text_async.assert_called_once_with(
        text="Hello",
        conversation_id="test_conversation_id",
    )
    assert response[0].get_value() == "Hi!"

    # Clean up the WebSocket connections
    await mock_websocket_target.cleanup_target()


@pytest.mark.asyncio
async def test_multiple_websockets_created_for_multiple_conversations(mock_websocket_target):
    # Mock the necessary methods
    mock_websocket_target.connect = AsyncMock(return_value=AsyncMock())
    result = "event2"
    mock_websocket_target.send_text_async = AsyncMock(return_value=result)

    # Create mock Messages for two different conversations
    message_piece_1 = MessagePiece(
        original_value="Hello",
        original_value_data_type="text",
        converted_value="Hello",
        converted_value_data_type="text",
        role="user",
        conversation_id="conversation_1",
    )
    message_1 = Message(message_pieces=[message_piece_1])

    message_piece_2 = MessagePiece(
        original_value="Hi",
        original_value_data_type="text",
        converted_value="Hi",
        converted_value_data_type="text",
        role="user",
        conversation_id="conversation_2",
    )
    message_2 = Message(message_pieces=[message_piece_2])

    # Call the send_prompt_async method for both conversations
    await mock_websocket_target.send_prompt_async(message=message_1)
    await mock_websocket_target.send_prompt_async(message=message_2)

    # Assert that two different WebSocket connections were created
    assert "conversation_1" in mock_websocket_target._existing_conversation
    assert "conversation_2" in mock_websocket_target._existing_conversation

    # Clean up the WebSocket connections
    await mock_websocket_target.cleanup_target()
    assert mock_websocket_target._existing_conversation == {}


@pytest.mark.asyncio
async def test_send_prompt_async_invalid_request(mock_websocket_target):
    # Create a mock Message with an invalid data type
    message_piece = MessagePiece(
        original_value="Invalid",
        original_value_data_type="image_path",
        converted_value="Invalid",
        converted_value_data_type="image_path",
        role="user",
    )
    message = Message(message_pieces=[message_piece])
    with pytest.raises(ValueError) as excinfo:
        mock_websocket_target._validate_request(message=message)

    assert "This target only supports text prompt input. Received: image_path." == str(excinfo.value)


@pytest.mark.asyncio
async def test_receive_messages_connection_closed(mock_websocket_target):
    """Test handling of WebSocket connection closing unexpectedly."""
    mock_websocket = AsyncMock()
    conversation_id = "test_connection_closed"
    mock_websocket_target._existing_conversation[conversation_id] = mock_websocket

    # create Close objects for the rcvd and sent parameters
    close_frame = Close(1000, "Normal closure")

    # forcing the websocket to raise a ConnectionClosed when iterated
    class FailingAsyncIterator:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise ConnectionClosed(rcvd=close_frame, sent=None)

    mock_websocket.__aiter__.side_effect = lambda: FailingAsyncIterator()
    result = await mock_websocket_target.receive_messages(conversation_id)
    assert result == ""


@pytest.mark.asyncio
async def test_receive_messages_with_text(mock_websocket_target):
    """Test successful processing of text message."""
    mock_websocket = AsyncMock()
    conversation_id = "test_success"
    mock_websocket_target._existing_conversation[conversation_id] = mock_websocket

    websocket_message = f"""{{"message":"test message"}}"""

    # mock websocket to yield all events
    mock_websocket.__aiter__.return_value = [websocket_message]

    result = await mock_websocket_target.receive_messages(conversation_id)

    assert result == "test message"
