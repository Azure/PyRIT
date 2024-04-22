# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pytest

from pyrit.completion import AzureCompletion


def test_valid_init():
    os.environ[AzureCompletion.API_KEY_ENVIRONMENT_VARIABLE] = ""
    completion = AzureCompletion(api_key="xxxxx", endpoint="https://mock.azure.com/", deployment="gpt-4")

    assert completion is not None


def test_valid_init_env():
    os.environ[AzureCompletion.API_KEY_ENVIRONMENT_VARIABLE] = "xxxxx"
    os.environ[AzureCompletion.ENDPOINT_URI_ENVIRONMENT_VARIABLE] = "https://testcompletionendpoint"
    os.environ[AzureCompletion.DEPLOYMENT_ENVIRONMENT_VARIABLE] = "testcompletiondeployment"

    completion = AzureCompletion()
    assert completion is not None


def test_invalid_key_raises():
    os.environ[AzureCompletion.API_KEY_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        AzureCompletion(
            api_key="",
            endpoint="https://mock.azure.com/",
            deployment="gpt-4",
            api_version="some_version",
        )


def test_invalid_endpoint_raises():
    os.environ[AzureCompletion.ENDPOINT_URI_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        AzureCompletion(
            api_key="xxxxxx",
            deployment="gpt-4",
            api_version="some_version",
        )


def test_invalid_deployment_raises():
    os.environ[AzureCompletion.DEPLOYMENT_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        AzureCompletion(
            api_key="",
            endpoint="https://mock.azure.com/",
        )
