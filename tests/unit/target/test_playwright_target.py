# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import MutableSequence
from unittest.mock import AsyncMock

import pytest

from pyrit.models import (
    Message,
    MessagePiece,
    construct_response_from_request,
)
from pyrit.prompt_target import PlaywrightTarget


@pytest.fixture
def sample_conversations() -> MutableSequence[MessagePiece]:
    conversation_1 = MessagePiece(
        role="user",
        converted_value="Hello",
        original_value="Hello",
        original_value_data_type="text",
        converted_value_data_type="text",
    )
    conversation_2 = MessagePiece(
        role="assistant",
        converted_value="World",
        original_value="World",
        original_value_data_type="text",
        converted_value_data_type="text",
    )
    return [conversation_1, conversation_2]


@pytest.fixture
def mock_interaction_func():
    async def interaction_func(page, message_piece):
        return f"Processed: {message_piece.converted_value}"

    return AsyncMock(side_effect=interaction_func)


@pytest.fixture
def mock_page():
    page = AsyncMock(name="MockPage")
    page.url = "https://example.com/test"
    return page


def test_playwright_initializes(mock_interaction_func, mock_page):
    target = PlaywrightTarget(interaction_func=mock_interaction_func, page=mock_page)
    assert target._interaction_func == mock_interaction_func


def test_playwright_sets_endpoint_and_rate_limit(mock_interaction_func, mock_page):
    target = PlaywrightTarget(interaction_func=mock_interaction_func, page=mock_page, max_requests_per_minute=20)
    identifier = target.get_identifier()
    assert identifier["endpoint"] == "https://example.com/test"
    assert target._max_requests_per_minute == 20


@pytest.mark.asyncio
async def test_playwright_validate_request_length(mock_interaction_func, mock_page):
    target = PlaywrightTarget(interaction_func=mock_interaction_func, page=mock_page)
    request = Message(
        message_pieces=[
            MessagePiece(role="user", conversation_id="123", original_value="test1"),
            MessagePiece(role="user", conversation_id="123", original_value="test2"),
        ]
    )
    with pytest.raises(ValueError, match="This target only supports a single message piece."):
        await target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_playwright_send_prompt_async(mock_interaction_func, mock_page):
    target = PlaywrightTarget(interaction_func=mock_interaction_func, page=mock_page)
    message_piece = MessagePiece(
        role="user",
        converted_value="Hello",
        original_value="Hello",
        original_value_data_type="text",
        converted_value_data_type="text",
    )
    request = Message(message_pieces=[message_piece])

    response = await target.send_prompt_async(prompt_request=request)
    # Assert that the response contains the assistant's message
    assert len(response.message_pieces) == 1  # Only assistant's response in this response
    assert response.message_pieces[0].role == "assistant"
    assert response.get_value() == "Processed: Hello"

    expected_response = construct_response_from_request(
        request=message_piece,
        response_text_pieces=[response.get_value()],
    )
    assert response.message_pieces[0].original_value == expected_response.message_pieces[0].original_value
    # Verify that the interaction function was called with the correct arguments
    mock_interaction_func.assert_awaited_once_with(mock_page, message_piece)


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_playwright_interaction_func_exception(mock_page):
    async def failing_interaction_func(page, message_piece):
        raise Exception("Interaction failed")

    target = PlaywrightTarget(interaction_func=failing_interaction_func, page=mock_page)
    message_piece = MessagePiece(
        role="user",
        converted_value="Hello",
        original_value="Hello",
        original_value_data_type="text",
        converted_value_data_type="text",
    )
    request = Message(message_pieces=[message_piece])

    with pytest.raises(RuntimeError, match="An error occurred during interaction: Interaction failed"):
        await target.send_prompt_async(prompt_request=request)
