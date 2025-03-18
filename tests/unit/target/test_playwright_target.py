# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import MutableSequence
from unittest.mock import AsyncMock

import pytest

from pyrit.models import (
    PromptRequestPiece,
    PromptRequestResponse,
    construct_response_from_request,
)
from pyrit.prompt_target import PlaywrightTarget


@pytest.fixture
def sample_conversations() -> MutableSequence[PromptRequestPiece]:
    conversation_1 = PromptRequestPiece(
        role="user",
        converted_value="Hello",
        original_value="Hello",
        original_value_data_type="text",
        converted_value_data_type="text",
    )
    conversation_2 = PromptRequestPiece(
        role="assistant",
        converted_value="World",
        original_value="World",
        original_value_data_type="text",
        converted_value_data_type="text",
    )
    return [conversation_1, conversation_2]


@pytest.fixture
def mock_interaction_func():
    async def interaction_func(page, request_piece):
        return f"Processed: {request_piece.converted_value}"

    return AsyncMock(side_effect=interaction_func)


@pytest.fixture
def mock_page():
    return AsyncMock(name="MockPage")


def test_playwright_initializes(mock_interaction_func, mock_page):
    target = PlaywrightTarget(interaction_func=mock_interaction_func, page=mock_page)
    assert target._interaction_func == mock_interaction_func


@pytest.mark.asyncio
async def test_playwright_validate_request_length(sample_conversations, mock_interaction_func, mock_page):
    target = PlaywrightTarget(interaction_func=mock_interaction_func, page=mock_page)
    request = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_playwright_send_prompt_async(mock_interaction_func, mock_page):
    target = PlaywrightTarget(interaction_func=mock_interaction_func, page=mock_page)
    request_piece = PromptRequestPiece(
        role="user",
        converted_value="Hello",
        original_value="Hello",
        original_value_data_type="text",
        converted_value_data_type="text",
    )
    request = PromptRequestResponse(request_pieces=[request_piece])

    response = await target.send_prompt_async(prompt_request=request)
    # Assert that the response contains the assistant's message
    assert len(response.request_pieces) == 1  # Only assistant's response in this response
    assert response.request_pieces[0].role == "assistant"
    assert response.get_value() == "Processed: Hello"

    expected_response = construct_response_from_request(
        request=request_piece,
        response_text_pieces=[response.get_value()],
    )
    assert response.request_pieces[0].original_value == expected_response.request_pieces[0].original_value
    # Verify that the interaction function was called with the correct arguments
    mock_interaction_func.assert_awaited_once_with(mock_page, request_piece)


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_playwright_interaction_func_exception(mock_page):
    async def failing_interaction_func(page, request_piece):
        raise Exception("Interaction failed")

    target = PlaywrightTarget(interaction_func=failing_interaction_func, page=mock_page)
    request_piece = PromptRequestPiece(
        role="user",
        converted_value="Hello",
        original_value="Hello",
        original_value_data_type="text",
        converted_value_data_type="text",
    )
    request = PromptRequestResponse(request_pieces=[request_piece])

    with pytest.raises(RuntimeError, match="An error occurred during interaction: Interaction failed"):
        await target.send_prompt_async(prompt_request=request)
