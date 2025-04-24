# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable
from unittest.mock import MagicMock, patch

import pytest

from pyrit.prompt_target.http_target.http_target import HTTPTarget
from pyrit.prompt_target.http_target.http_target_callback_functions import (
    get_http_target_json_response_callback_function,
    get_http_target_regex_matching_callback_function,
)


@pytest.fixture
def mock_callback_function() -> Callable:
    parsing_function = get_http_target_json_response_callback_function(key="mock_key")
    return parsing_function


@pytest.fixture
def mock_http_target(mock_callback_function, duckdb_instance) -> HTTPTarget:
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
    assert response.get_value() == "value1"
    assert mock_request.call_count == 1
    mock_request.assert_called_with(
        method="POST",
        url="https://example.com/",
        headers={"host": "example.com", "content-type": "application/json"},
        content='{"prompt": "test_prompt"}',
        follow_redirects=True,
    )


def test_parse_raw_http_request_ignores_content_length(patch_central_database):

    request = "POST / HTTP/1.1\nHost: example.com\nContent-Type: application/json\nContent-Length: 100\n\n"
    target = HTTPTarget(http_request=request)

    headers, _, _, _, _ = target.parse_raw_http_request(request)
    assert headers == {"host": "example.com", "content-type": "application/json"}


def test_parse_raw_http_respects_url_path(patch_central_database):

    request1 = (
        "POST https://diffsite.com/test/ HTTP/1.1\nHost: example.com\nContent-Type: "
        + "application/json\nContent-Length: 100\n\n"
    )
    target = HTTPTarget(http_request=request1)
    headers, _, url, _, _ = target.parse_raw_http_request(request1)
    assert url == "https://diffsite.com/test/"

    # The host header should still be example.com
    assert headers == {"host": "example.com", "content-type": "application/json"}


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_send_prompt_async_client_kwargs(mock_async_client):
    # Create httpx_client_kwargs to test
    httpx_client_kwargs = {"timeout": 10, "verify": False}
    sample_request = "GET /test HTTP/1.1\nHost: example.com\n\n"
    # Create instance of HTTPTarget with httpx_client_kwargs
    # Use **httpx_client_kwargs to pass them as keyword arguments
    http_target = HTTPTarget(http_request=sample_request, **httpx_client_kwargs)
    prompt_request = MagicMock()
    prompt_request.request_pieces = [MagicMock(converted_value="")]
    mock_response = MagicMock()
    mock_response.content = b"Response content"
    instance = mock_async_client.return_value.__aenter__.return_value
    instance.request.return_value = mock_response
    await http_target.send_prompt_async(prompt_request=prompt_request)

    mock_async_client.assert_called_with(http2=False, timeout=10, verify=False)


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
    assert response.get_value() == "Match: 1234"
    assert mock_request.call_count == 1
    mock_request.assert_called_with(
        method="POST",
        url="https://example.com/",
        headers={"host": "example.com", "content-type": "application/json"},
        content='{"prompt": "test_prompt"}',
        follow_redirects=True,
    )


@pytest.mark.asyncio
@patch("httpx.AsyncClient.request")
async def test_send_prompt_async_keeps_original_template(mock_request, mock_http_target, mock_http_response):

    original_http_request = mock_http_target.http_request
    mock_request.return_value = mock_http_response

    # Send first prompt
    prompt_request = MagicMock()
    prompt_request.request_pieces = [MagicMock(converted_value="test_prompt")]
    response = await mock_http_target.send_prompt_async(prompt_request=prompt_request)

    assert response.get_value() == "value1"
    assert mock_http_target.http_request == original_http_request

    assert mock_request.call_count == 1
    mock_request.assert_called_with(
        method="POST",
        url="https://example.com/",
        headers={"host": "example.com", "content-type": "application/json"},
        content='{"prompt": "test_prompt"}',
        follow_redirects=True,
    )

    # Send second prompt
    second_prompt_request = MagicMock()
    second_prompt_request.request_pieces = [MagicMock(converted_value="second_test_prompt")]
    await mock_http_target.send_prompt_async(prompt_request=second_prompt_request)

    # Assert that the original template is still the same
    assert mock_http_target.http_request == original_http_request

    assert mock_request.call_count == 2
    # Verify HTTP requests were made with the correct prompts

    mock_request.assert_any_call(
        method="POST",
        url="https://example.com/",
        headers={"host": "example.com", "content-type": "application/json"},
        content='{"prompt": "test_prompt"}',
        follow_redirects=True,
    )
    mock_request.assert_any_call(
        method="POST",
        url="https://example.com/",
        headers={"host": "example.com", "content-type": "application/json"},
        content='{"prompt": "second_test_prompt"}',
        follow_redirects=True,
    )
