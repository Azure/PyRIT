# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from unittest.mock import MagicMock, patch

import pytest

from pyrit.embedding import AzureTextEmbedding


def test_valid_init():
    os.environ[AzureTextEmbedding.API_KEY_ENVIRONMENT_VARIABLE] = ""
    completion = AzureTextEmbedding(api_key="xxxxx", endpoint="https://mock.azure.com/", deployment="gpt-4")

    assert completion is not None


def test_valid_init_env():
    os.environ[AzureTextEmbedding.API_KEY_ENVIRONMENT_VARIABLE] = "xxxxx"
    os.environ[AzureTextEmbedding.ENDPOINT_URI_ENVIRONMENT_VARIABLE] = "https://testcompletionendpoint"
    os.environ[AzureTextEmbedding.DEPLOYMENT_ENVIRONMENT_VARIABLE] = "testcompletiondeployment"

    completion = AzureTextEmbedding()
    assert completion is not None


def test_invalid_key_raises():
    os.environ[AzureTextEmbedding.API_KEY_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        AzureTextEmbedding(
            api_key="",
            endpoint="https://mock.azure.com/",
            deployment="gpt-4",
            api_version="some_version",
        )


def test_invalid_endpoint_raises():
    os.environ[AzureTextEmbedding.ENDPOINT_URI_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        AzureTextEmbedding(
            api_key="xxxxxx",
            deployment="gpt-4",
            api_version="some_version",
        )


def test_invalid_deployment_raises():
    os.environ[AzureTextEmbedding.DEPLOYMENT_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        AzureTextEmbedding(
            api_key="",
            endpoint="https://mock.azure.com/",
        )


def test_use_entra_auth_true_with_api_key_raises_error():
    """Test that use_entra_auth=True with api_key raises ValueError."""
    os.environ[AzureTextEmbedding.ENDPOINT_URI_ENVIRONMENT_VARIABLE] = "https://mock.azure.com/"
    os.environ[AzureTextEmbedding.DEPLOYMENT_ENVIRONMENT_VARIABLE] = "text-embedding"

    with pytest.raises(ValueError, match="If using Entra ID auth, please do not specify api_key"):
        AzureTextEmbedding(api_key="some_key", use_entra_auth=True)


@patch("pyrit.embedding.azure_text_embedding.AzureAuth")
@patch("pyrit.embedding.azure_text_embedding.get_default_scope")
@patch("pyrit.embedding.azure_text_embedding.AzureOpenAI")
def test_use_entra_auth_default_false_uses_api_key(mock_azure_openai, mock_get_default_scope, mock_azure_auth):
    """Test that default behavior (use_entra_auth=False) uses API key."""
    mock_client = MagicMock()
    mock_azure_openai.return_value = mock_client

    # Set required environment variables
    os.environ[AzureTextEmbedding.API_KEY_ENVIRONMENT_VARIABLE] = "env_api_key"
    os.environ[AzureTextEmbedding.ENDPOINT_URI_ENVIRONMENT_VARIABLE] = "https://mock.azure.com/"
    os.environ[AzureTextEmbedding.DEPLOYMENT_ENVIRONMENT_VARIABLE] = "text-embedding"

    # Create instance without specifying use_entra_auth (should default to False)
    embedding = AzureTextEmbedding()

    # Verify Azure Auth was NOT used
    mock_get_default_scope.assert_not_called()
    mock_azure_auth.assert_not_called()

    # Verify AzureOpenAI client was created with API key from environment
    mock_azure_openai.assert_called_once_with(
        api_key="env_api_key",
        api_version="2024-02-01",
        azure_endpoint="https://mock.azure.com/",
        azure_deployment="text-embedding",
    )

    assert embedding._client == mock_client
