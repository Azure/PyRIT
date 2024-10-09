# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from unittest.mock import patch, Mock
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
def mock_http_response() -> Mock:
    mock_response = Mock()
    mock_response.content = b'{"mock_key": "value1"}'
    return mock_response


def test_initilization_with_parameters(mock_http_target, mock_callback_function):
    assert (
        mock_http_target.http_request
        == 'POST / HTTP/1.1\nHost: example.com\nContent-Type: application/json\n\n{"prompt": "{PLACEHOLDER_PROMPT}"}'
    )
    assert mock_http_target.prompt_regex_string == "{PLACEHOLDER_PROMPT}"
    assert mock_http_target.callback_function == mock_callback_function


def test_parse_json_response_no_match(mock_http_response):
    parse_json_response = get_http_target_json_response_callback_function(key="nonexistant_key")
    result = parse_json_response(mock_http_response)
    assert result == ""


def test_parse_json_response_match(mock_http_response, mock_callback_function):
    result = mock_callback_function(mock_http_response)
    assert result == "value1"


@pytest.mark.asyncio
@patch("requests.request")
async def test_send_prompt_async(mock_request, mock_http_target, mock_http_response):
    prompt_request = Mock()
    prompt_request.request_pieces = [Mock(converted_value="test_prompt")]
    mock_request.return_value = mock_http_response
    response = await mock_http_target.send_prompt_async(prompt_request=prompt_request)
    assert response.request_pieces[0].converted_value == "value1"
    assert mock_request.call_count == 1


def test_parse_raw_http_request(mock_http_target):
    headers, body, url, method = mock_http_target.parse_raw_http_request()
    assert url == "https://example.com/"
    assert method == "POST"
    assert headers == {"Host": "example.com", "Content-Type": "application/json"}

    assert body == '{"prompt": "{PLACEHOLDER_PROMPT}"}'


def test_parse_regex_response_no_match():
    mock_response = Mock()
    mock_response.content = b"<html><body>No match here</body></html>"
    parse_html_function = get_http_target_regex_matching_callback_function(key=r'no_results\/[^\s"]+')
    result = parse_html_function(mock_response)
    assert result == "b'<html><body>No match here</body></html>'"


def test_parse_regex_response_match():
    mock_response = Mock()
    mock_response.content = b"<html><body>Match: 1234</body></html>"
    parse_html_response = get_http_target_regex_matching_callback_function(r"Match: (\d+)")
    result = parse_html_response(mock_response)
    assert result == "Match: 1234"
