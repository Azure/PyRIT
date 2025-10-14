# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import time
from typing import Callable, Union
from urllib.parse import urlparse

import msal
from azure.core.credentials import AccessToken
from azure.identity import (
    AzureCliCredential,
    DefaultAzureCredential,
    InteractiveBrowserCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)

from pyrit.auth.auth_config import REFRESH_TOKEN_BEFORE_MSEC
from pyrit.auth.authenticator import Authenticator

logger = logging.getLogger(__name__)


class AzureAuth(Authenticator):
    """
    Azure CLI Authentication.
    """

    access_token: AccessToken
    _token_scope: str

    def __init__(self, token_scope: str, tenant_id: str = ""):
        self._tenant_id = tenant_id
        self._token_scope = token_scope
        self._set_default_token()

    def _set_default_token(self) -> None:
        self.azure_creds = DefaultAzureCredential()
        self.access_token = self.azure_creds.get_token(self._token_scope)
        self.token = self.access_token.token

    def refresh_token(self) -> str:
        """Refresh the access token if it is expired.

        Returns:
            A token

        """
        curr_epoch_time_in_ms = int(time.time()) * 1_000
        access_token_epoch_expiration_time_in_ms = int(self.access_token.expires_on) * 1_000
        # Adjust the expiration time to be before the actual expiration time so that user can use the token
        # for a while before it expires. This improves user experience. The token is refreshed REFRESH_TOKEN_BEFORE_MSEC
        # before it expires.
        token_expires_on_in_ms = access_token_epoch_expiration_time_in_ms - REFRESH_TOKEN_BEFORE_MSEC
        if token_expires_on_in_ms <= curr_epoch_time_in_ms:
            # Token is expired, generate a new one
            self._set_default_token()
        return self.token

    def get_token(self) -> str:
        """
        Get the current token.

        Returns: The current token

        """
        return self.token


def get_access_token_from_azure_cli(*, scope: str, tenant_id: str = ""):
    try:
        credential = AzureCliCredential(tenant_id=tenant_id)
        token = credential.get_token(scope)
        return token.token
    except Exception as e:
        logger.error(f"Failed to obtain token for '{scope}' with tenant ID '{tenant_id}': {e}")
        raise


def get_access_token_from_azure_msi(*, client_id: str, scope: str):
    """Connect to an AOAI endpoint via managed identity credential attached to an Azure resource.
    For proper setup and configuration of MSI
    https://learn.microsoft.com/en-us/entra/identity/managed-identities-azure-resources/overview.

    Args:
        client_id (str): The client ID of the service
        scope (str): The scope to request

    Returns:
        Authentication token
    """
    try:
        credential = ManagedIdentityCredential(client_id=client_id)
        token = credential.get_token(scope)
        return token.token
    except Exception as e:
        logger.error(f"Failed to obtain token for '{scope}' with client ID '{client_id}': {e}")
        raise


def get_access_token_from_msa_public_client(*, client_id: str, scope: str):
    """Uses MSA account to connect to an AOAI endpoint via interactive login. A browser window
    will open and ask for login credentials.

    Args:
        client_id (str): The client ID of the service
        scope (str): The scope to request

    Returns:
        Authentication token
    """
    try:
        app = msal.PublicClientApplication(client_id)
        result = app.acquire_token_interactive(scopes=[scope])
        return result["access_token"]
    except Exception as e:
        logger.error(f"Failed to obtain token for '{scope}' with client ID '{client_id}': {e}")
        raise


def get_access_token_from_interactive_login(scope: str) -> str:
    """Connects to an OpenAI endpoint with an interactive login from Azure. A browser window will
    open and ask for login credentials.  The token will be scoped for Azure Cognitive services.

    Returns:
        Authentication token
    """
    try:
        token_provider = get_bearer_token_provider(InteractiveBrowserCredential(), scope)
        return token_provider()
    except Exception as e:
        logger.error(f"Failed to obtain token for '{scope}': {e}")
        raise


def get_token_provider_from_default_azure_credential(scope: str) -> Callable[[], str]:
    """Connect to an AOAI endpoint via default Azure credential.

    Returns:
        Authentication token provider
    """
    try:
        token_provider = get_bearer_token_provider(DefaultAzureCredential(), scope)
        return token_provider
    except Exception as e:
        logger.error(f"Failed to obtain token for '{scope}': {e}")
        raise


def get_default_scope(endpoint: str) -> str:
    """Get the default scope for the given endpoint.

    Args:
        endpoint (str): The endpoint to get the scope for.

    Returns:
        The default scope for the given endpoint.
    """
    try:
        parsed_uri = urlparse(endpoint)
        if parsed_uri.hostname and parsed_uri.hostname.lower().endswith(".ai.azure.com"):
            return "https://ml.azure.com/.default"
    except Exception:
        pass

    return "https://cognitiveservices.azure.com/.default"


def get_speech_config(resource_id: Union[str, None], key: Union[str, None], region: str):
    """
    Get the speech config using key/region pair (for key auth scenarios) or resource_id/region pair
    (for Entra auth scenarios)

    Args:
        resource_id (Union[str, None]): The resource ID to get the token for.
        key (Union[str, None]): The Azure Speech key
        region (str): The region to get the token for.
    Returns:
        The speech config based on passed in args

    Raises:
        ModuleNotFoundError: If azure.cognitiveservices.speech is not installed.
        ValueError: If neither key/region nor resource_id/region is provided.
    """
    try:
        import azure.cognitiveservices.speech as speechsdk  # noqa: F811
    except ModuleNotFoundError as e:
        logger.error(
            "Could not import azure.cognitiveservices.speech. "
            + "You may need to install it via 'pip install pyrit[speech]'"
        )
        raise e

    if key and region:
        return speechsdk.SpeechConfig(
            subscription=key,
            region=region,
        )
    elif resource_id and region:
        return get_speech_config_from_default_azure_credential(
            resource_id=resource_id,
            region=region,
        )
    else:
        raise ValueError("Insufficient information provided for Azure Speech service.")


def get_speech_config_from_default_azure_credential(resource_id: str, region: str):
    """Get the speech config for the given resource ID and region.

    Args:
        resource_id (str): The resource ID to get the token for.
        region (str): The region to get the token for.

    Returns:
        The speech config for the given resource ID and region.
    """
    try:
        import azure.cognitiveservices.speech as speechsdk  # noqa: F811
    except ModuleNotFoundError as e:
        logger.error(
            "Could not import azure.cognitiveservices.speech. "
            + "You may need to install it via 'pip install pyrit[speech]'"
        )
        raise e

    try:
        azure_auth = AzureAuth(token_scope=get_default_scope(""))
        token = azure_auth.get_token()
        authorization_token = "aad#" + resource_id + "#" + token
        speech_config = speechsdk.SpeechConfig(
            auth_token=authorization_token,
            region=region,
        )
        return speech_config
    except Exception as e:
        logger.error(f"Failed to get speech config for resource ID '{resource_id}' and region '{region}': {e}")
        raise
