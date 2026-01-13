# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# The standard pattern for refreshing a token is to trigger 300 ms before the token expires to prevent users from
# having an expired token.
REFRESH_TOKEN_BEFORE_MSEC: int = 300
AZURE_AI_SERVICES_DEFAULT_SCOPE: list[str] = [
    "https://cognitiveservices.azure.com/.default",
    "https://ml.azure.com/.default",
]
