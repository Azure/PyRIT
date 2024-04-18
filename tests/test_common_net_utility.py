# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import httpx
import respx

from unittest.mock import patch, MagicMock
from tenacity import RetryError
from pyrit.common.net_utility import get_httpx_client, make_request_and_raise_if_error


@pytest.mark.parametrize(
    "use_async, expected_type",
    [
        (False, httpx.Client),
        (True, httpx.AsyncClient),
    ],
)
def test_get_httpx_client_type(use_async, expected_type):
    client = get_httpx_client(use_async=use_async)
    assert isinstance(client, expected_type)


@respx.mock
def test_make_request_and_raise_if_error_success():
    url = "http://testserver/api/test"
    method = "GET"
    mock_route = respx.get(url).respond(200, json={"status": "ok"})
    response = make_request_and_raise_if_error(endpoint_uri=url, method=method)
    assert mock_route.called
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@respx.mock
def test_make_request_and_raise_if_error_failure():
    url = "http://testserver/api/fail"
    method = "GET"
    mock_route = respx.get(url).respond(500)

    with pytest.raises(RetryError) as retry_error:
        make_request_and_raise_if_error(endpoint_uri=url, method=method)
    assert mock_route.called

    last_exception = retry_error.value.last_attempt.exception()
    assert isinstance(last_exception, httpx.HTTPStatusError)


@respx.mock
def test_make_request_and_raise_if_error_retries():
    url = "http://testserver/api/retry"
    method = "GET"
    call_count = 0

    def response_callback(request):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return httpx.Response(500)
        return httpx.Response(200, json={"status": "ok"})

    mock_route = respx.route(method=method, url=url).mock(side_effect=response_callback)

    with pytest.raises(RetryError):
        make_request_and_raise_if_error(endpoint_uri=url, method=method)
    assert call_count == 2, "The request should have been retried exactly once."
    assert mock_route.called


def test_debug_is_false_by_default():
    with patch("pyrit.common.net_utility.get_httpx_client") as mock_get_httpx_client:
        mock_client_instance = MagicMock()
        mock_get_httpx_client.return_value = mock_client_instance

        make_request_and_raise_if_error(endpoint_uri="http://example.com", method="GET")

        mock_get_httpx_client.assert_called_with(debug=False)
