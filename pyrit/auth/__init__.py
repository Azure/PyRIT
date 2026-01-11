# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Authentication functionality for a variety of services.
"""

from pyrit.auth.authenticator import Authenticator
from pyrit.auth.azure_auth import (
    AzureAuth,
    TokenProviderCredential,
    get_azure_async_token_provider,
    get_azure_openai_auth,
    get_azure_token_provider,
    get_default_azure_scope,
)
from pyrit.auth.azure_storage_auth import AzureStorageAuth
from pyrit.auth.copilot_authenticator import CopilotAuthenticator

__all__ = [
    "Authenticator",
    "AzureAuth",
    "AzureStorageAuth",
    "CopilotAuthenticator",
    "TokenProviderCredential",
    "get_azure_token_provider",
    "get_azure_async_token_provider",
    "get_default_azure_scope",
    "get_azure_openai_auth",
]
