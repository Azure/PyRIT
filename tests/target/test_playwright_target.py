# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from unittest.mock import AsyncMock

from pyrit.models import PromptRequestResponse, PromptRequestPiece, construct_response_from_request
from pyrit.prompt_target import PlaywrightTarget


@pytest.fixture
def sample_conversations() -> list[PromptRequestPiece]:
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


def test_playwright_initializes(mock_interaction_func):
    target = PlaywrightTarget(interaction_func=mock_interaction_func)
    assert target._interaction_func == mock_interaction_func
    assert target._page is None


@pytest.mark.asyncio
async def test_playwright_validate_request_length(sample_conversations, mock_interaction_func, mock_page):
    target = PlaywrightTarget(interaction_func=mock_interaction_func, page=mock_page)
    request = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_playwright_send_prompt_without_page(mock_interaction_func):
    target = PlaywrightTarget(interaction_func=mock_interaction_func)
    request_piece = PromptRequestPiece(
        role="user",
        converted_value="Hello",
        original_value="Hello",
        original_value_data_type="text",
        converted_value_data_type="text",
    )
    request = PromptRequestResponse(request_pieces=[request_piece])
    with pytest.raises(
        RuntimeError,
        match="Playwright page is not initialized. Please pass a Page object when initializing PlaywrightTarget.",
    ):
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
    assert response.request_pieces[0].converted_value == "Processed: Hello"

    expected_response = construct_response_from_request(
        request=request_piece,
        response_text_pieces=[response.request_pieces[0].converted_value],
    )
    assert response.request_pieces[0].original_value == expected_response.request_pieces[0].original_value
    # Verify that the interaction function was called with the correct arguments
    mock_interaction_func.assert_awaited_once_with(mock_page, request_piece)


@pytest.mark.asyncio
async def test_playwright_send_prompt_with_various_data_types(mock_interaction_func, mock_page):
    target = PlaywrightTarget(interaction_func=mock_interaction_func, page=mock_page)
    data_types = ["text", "image_path", "audio_path", "url", "error"]
    for data_type in data_types:
        request_piece = PromptRequestPiece(
            role="user",
            converted_value=f"{data_type}_value",
            original_value=f"{data_type}_value",
            original_value_data_type=data_type,
            converted_value_data_type=data_type,
        )
        request = PromptRequestResponse(request_pieces=[request_piece])

        response = await target.send_prompt_async(prompt_request=request)
        # Assert that the response contains the assistant's message
        assert len(response.request_pieces) == 1  # Only assistant's response in this response
        assert response.request_pieces[0].role == "assistant"
        assert response.request_pieces[0].converted_value == f"Processed: {data_type}_value"

        expected_response = construct_response_from_request(
            request=request_piece,
            response_text_pieces=[response.request_pieces[0].converted_value],
        )
        assert response.request_pieces[0].original_value == expected_response.request_pieces[0].original_value
        # Verify that the interaction function was called with the correct arguments
        mock_interaction_func.assert_any_await(mock_page, request_piece)


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