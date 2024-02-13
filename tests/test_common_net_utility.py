import json
from unittest.mock import patch, MagicMock
import pytest

from pyrit.common.net_utility import make_request_and_raise_if_error


@pytest.fixture
def mock_http():
    return MagicMock()


@pytest.fixture
def mock_response():
    response = MagicMock()
    response.status = 200
    response.reason = "OK"
    response.data = b"Response data"
    return response


def test_make_request_and_raise_if_error_http_error(mock_http, mock_response):
    mock_response.status = 404
    mock_response.reason = "Not Found"
    mock_response.data = b"Error message"
    mock_http.request.return_value = mock_response

    with patch("pyrit.common.net_utility.get_pool_manager", return_value=mock_http):
        with pytest.raises(RuntimeError) as exc_info:
            make_request_and_raise_if_error(
                endpoint_uri="http://example.com",
                method="GET",
                request_body=None,
                headers={},
                retries=3,
                use_proxy=False,
            )
        assert str(exc_info.value) == "HTTP error: Not Found\nError message."


def test_make_request_and_raise_if_error_with_json_body(mock_http, mock_response):
    mock_http.request.return_value = mock_response

    with patch("pyrit.common.net_utility.get_pool_manager", return_value=mock_http):
        response = make_request_and_raise_if_error(
            endpoint_uri="http://example.com",
            method="POST",
            request_body={"key": "value"},
            headers={},
            retries=3,
            use_proxy=False,
        )
        assert response == mock_response
        assert mock_http.request.call_args[1]["body"] == json.dumps({"key": "value"}).encode("utf-8")
        assert mock_http.request.call_args[1]["headers"]["Content-Type"] == "application/json"
