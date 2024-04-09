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

    def connect_openai_msi(self, client_id):
        """Connect to to AOAI endpoint via managed identity credential attached to an Azure resource.  For proper setup and configuration of MSI https://learn.microsoft.com/en-us/entra/identity/managed-identities-azure-resources/overview.

        Args:
            client id of the service

        Returns:
            authentication token
        """
        try:
            credential = azure.identity.ManagedIdentityCredential(client_id=client_id)
            token = credential.get_token("https://cognitiveservices.azure.com/.default")
            return token.token
        except Exception as e:
            return e

    def connect_msa_interactive_login(self, client_id):
        """uses MSA account to connect to an AOAI endpoint via interactive login. A browser window will open and ask for login credentials.

        Args:
            client id

        Returns:
            authentication token
        """
        try:
            app = msal.PublicClientApplication(client_id)
            result = app.acquire_token_interactive(scopes=["https://cognitiveservices.azure.com/.default"])
            print(result)
            return result["access_token"]
        except Exception as e:
            return e

    def connect_openai_interactive_login(self):
        """connects to an OpenAI endpoint with an interactive login from Azure. A browser window will open and ask for login credentials.  The token will be scoped for Azure Cognitive services.

        Returns:
            authentication token
        """
        try:
            token_provider = azure.identity.get_bearer_token_provider(
                azure.identity.InteractiveBrowserCredential(),
                "https://cognitiveservices.azure.com/.default"
            )
            return token_provider()
        except Exception as e:
            return e     