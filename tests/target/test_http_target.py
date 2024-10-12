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


@pytest.mark.asyncio
@patch("requests.request")
async def test_send_prompt_async(mock_request, mock_http_target, mock_http_response):
    prompt_request = Mock()
    prompt_request.request_pieces = [Mock(converted_value="test_prompt")]
    mock_request.return_value = mock_http_response
    response = await mock_http_target.send_prompt_async(prompt_request=prompt_request)
    assert response.request_pieces[0].converted_value == "value1"
    assert mock_request.call_count == 1
