# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Authentication functionality for a variety of services.
"""

from pyrit.auth.authenticator import Authenticator
from pyrit.auth.azure_auth import AzureAuth
from pyrit.auth.azure_storage_auth import AzureStorageAuth

__all__ = ["Authenticator", "AzureAuth", "AzureStorageAuth"]
