# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

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
