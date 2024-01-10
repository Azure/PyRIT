# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from unittest.mock import Mock, patch

from pyrit.auth.auth_config import REFRESH_TOKEN_BEFORE_MSEC
from pyrit.auth.azure_auth import AzureAuth

curr_epoch_time = int(time.time())
mock_token = "fake token"


def test_get_token_on_init():
    with patch("azure.identity.AzureCliCredential.get_token") as mock_get_token:
        mock_get_token.return_value = Mock(token=mock_token)
        test_instance = AzureAuth(token_scope="https://mocked_endpoint.azure.com")
        assert test_instance.token == mock_token


def test_refresh_no_expiration():
    # Token not expired so not reset
    with patch("azure.identity.AzureCliCredential.get_token") as mock_get_token:
        mock_get_token.return_value = Mock(token=mock_token, expires_on=curr_epoch_time + REFRESH_TOKEN_BEFORE_MSEC)
        test_instance = AzureAuth(token_scope="https://mocked_endpoint.azure.com")
        token = test_instance.refresh_token()
        assert token == mock_token
        mock_get_token.assert_called_once()


def test_refresh_expiration():
    # Token expired and reset
    with patch("azure.identity.AzureCliCredential.get_token") as mock_get_token:
        mock_get_token.return_value = Mock(token=mock_token, expires_on=curr_epoch_time)
        test_instance = AzureAuth(token_scope="https://mocked_endpoint.azure.com")
        token = test_instance.refresh_token()
        assert token
        assert mock_get_token.call_count == 2
