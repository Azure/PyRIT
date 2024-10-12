# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from unittest.mock import patch, MagicMock
from pyrit.prompt_target.http_target.http_target import HTTPTarget
from pyrit.prompt_target.http_target.http_target_callback_functions import (
    get_http_target_json_response_callback_function,
    get_http_target_regex_matching_callback_function,
)
from typing import Callable


@pytest.fixture
def mock_callback_function() -> Callable:
    parsing_function = get_http_target_json_response_callback_function(key="mock_key")
    return parsing_function


@pytest.fixture
def mock_http_target(mock_callback_function) -> HTTPTarget:
    sample_request = (
        'POST / HTTP/1.1\nHost: example.com\nContent-Type: application/json\n\n{"prompt": "{PLACEHOLDER_PROMPT}"}'
    )
    return HTTPTarget(
        http_request=sample_request,
        prompt_regex_string="{PLACEHOLDER_PROMPT}",
        callback_function=mock_callback_function,
    )


@pytest.fixture
def mock_http_response() -> MagicMock:
    mock_response = MagicMock()
    mock_response.content = b'{"mock_key": "value1"}'
    return mock_response


def test_initilization_with_parameters(mock_http_target, mock_callback_function):
    assert (
        mock_http_target.http_request
        == 'POST / HTTP/1.1\nHost: example.com\nContent-Type: application/json\n\n{"prompt": "{PLACEHOLDER_PROMPT}"}'
    )
    assert mock_http_target.prompt_regex_string == "{PLACEHOLDER_PROMPT}"
    assert mock_http_target.callback_function == mock_callback_function


@pytest.mark.asyncio
@patch("httpx.AsyncClient.request")
async def test_send_prompt_async(mock_request, mock_http_target, mock_http_response):
    prompt_request = MagicMock()
    prompt_request.request_pieces = [MagicMock(converted_value="test_prompt")]
    mock_request.return_value = mock_http_response
    response = await mock_http_target.send_prompt_async(prompt_request=prompt_request)
    assert response.request_pieces[0].converted_value == "value1"
    assert mock_request.call_count == 1
    mock_request.assert_called_with(
        method="POST",
        url="https://example.com/",
        headers={"Host": "example.com", "Content-Type": "application/json"},
        data='{"prompt": "test_prompt"}',
        follow_redirects=True,
    )


@pytest.mark.asyncio
async def test_send_prompt_async_validation(mock_http_target):
    # Create an invalid prompt request (missing request_pieces)
    invalid_prompt_request = MagicMock()
    invalid_prompt_request.request_pieces = []
    with pytest.raises(ValueError) as value_error:
        await mock_http_target.send_prompt_async(prompt_request=invalid_prompt_request)

    assert str(value_error.value) == "This target only supports a single prompt request piece."


@pytest.mark.asyncio
@patch("httpx.AsyncClient.request")
async def test_send_prompt_regex_parse_async(mock_request, mock_http_target):

    callback_function = get_http_target_regex_matching_callback_function(key=r"Match: (\d+)")
    mock_http_target.callback_function = callback_function

    prompt_request = MagicMock()
    prompt_request.request_pieces = [MagicMock(converted_value="test_prompt")]

    mock_response = MagicMock()
    mock_response.content = b"<html><body>Match: 1234</body></html>"
    mock_request.return_value = mock_response

    response = await mock_http_target.send_prompt_async(prompt_request=prompt_request)
    assert response.request_pieces[0].converted_value == "Match: 1234"
    assert mock_request.call_count == 1
    mock_request.assert_called_with(
        method="POST",
        url="https://example.com/",
        headers={"Host": "example.com", "Content-Type": "application/json"},
        data='{"prompt": "test_prompt"}',
        follow_redirects=True,
    )
