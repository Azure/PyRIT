# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time

from azure.core.credentials import AccessToken
from azure.identity import AzureCliCredential

from pyrit.auth.auth_config import REFRESH_TOKEN_BEFORE_MSEC
from pyrit.interfaces import Authenticator


class AzureAuth(Authenticator):
    """
    Azure CLI Authentication.
    """

    _access_token: AccessToken
    _tenant_id: str
    _token_scope: str

    def __init__(self, token_scope: str, tenant_id: str = ""):
        self._tenant_id = tenant_id
        self._token_scope = token_scope
        azure_creds = AzureCliCredential(tenant_id=tenant_id)
        self._access_token = azure_creds.get_token(self._token_scope)
        # Make the token available to the user
        self.token = self._access_token.token

    def refresh_token(self) -> str:
        """Refresh the access token if it is expired.

        Returns:
            A token

        """
        curr_epoch_time_in_ms = int(time.time()) * 1_000
        access_token_epoch_expiration_time_in_ms = int(self._access_token.expires_on) * 1_000
        # Adjust the expiration time to be before the actual expiration time so that user can use the token
        # for a while before it expires. This improves user experience. The token is refreshed REFRESH_TOKEN_BEFORE_MSEC
        # before it expires.
        token_expires_on_in_ms = access_token_epoch_expiration_time_in_ms - REFRESH_TOKEN_BEFORE_MSEC
        if token_expires_on_in_ms <= curr_epoch_time_in_ms:
            # Token is expired, generate a new one
            azure_creds = AzureCliCredential(tenant_id=self._tenant_id)
            self._access_token = azure_creds.get_token(self._token_scope)
            self.token = self._access_token.token
        return self.token

    def get_token(self) -> str:
        """
        Get the current token.

        Returns: The current token

        """
        return self.token
