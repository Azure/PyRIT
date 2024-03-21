# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import os

from contextlib import AbstractAsyncContextManager
from unittest.mock import AsyncMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from pyrit.prompt_target import AzureOpenAIChatTarget, OpenAIChatTarget
from pyrit.models import ChatMessage


@pytest.fixture
def openai_mock_return() -> ChatCompletion:
    return ChatCompletion(
        id="12345678-1a2b-3c4e5f-a123-12345678abcd",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hi"),
                finish_reason="stop",
                logprobs=None,
            )
        ],
        created=1629389505,
        model="gpt-4",
    )


@pytest.fixture
def azure_chat_engine() -> AzureOpenAIChatTarget:
    return AzureOpenAIChatTarget(
        deployment_name="gpt-4",
        endpoint="https://mock.azure.com/",
        api_key="mock-api-key",
        api_version="some_version",
    )

@pytest.fixture
def openai_chat_engine() -> OpenAIChatTarget:
    return OpenAIChatTarget(
        deployment_name="gpt-4",
        endpoint="https://mock.azure.com/",
        api_key="mock-api-key",
    )


class MockChatCompletionsAsync(AbstractAsyncContextManager):
    async def __call__(self, *args, **kwargs):
        self.mock_chat_completion = ChatCompletion(
            id="12345678-1a2b-3c4e5f-a123-12345678abcd",
            object="chat.completion",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="hi"),
                    finish_reason="stop",
                    logprobs=None,
                )
            ],
            created=1629389505,
            model="gpt-4",
        )
        return self.mock_chat_completion

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        pass


@patch(
    "openai.resources.chat.AsyncCompletions.create",
    new_callable=lambda: MockChatCompletionsAsync(),
)

@pytest.mark.asyncio
async def test_azure_complete_chat_async_return(openai_mock_return: ChatCompletion, azure_chat_engine: AzureOpenAIChatTarget):
    with patch("openai.resources.chat.Completions.create") as mock_create:
        mock_create.return_value = openai_mock_return
        ret = await azure_chat_engine.complete_chat_async(messages=[ChatMessage(role="user", content="hello")])
        assert ret == "hi"

@pytest.mark.asyncio
async def test_openai_complete_chat_async_return(openai_mock_return: ChatCompletion, openai_chat_engine: OpenAIChatTarget):
    with patch("openai.resources.chat.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = openai_mock_return
        ret = await openai_chat_engine.complete_chat_async(messages=[ChatMessage(role="user", content="hello")])
        assert ret == "hi"

def test_azure_complete_chat_return(openai_mock_return: ChatCompletion, azure_chat_engine: AzureOpenAIChatTarget):
    with patch("openai.resources.chat.Completions.create") as mock_create:
        mock_create.return_value = openai_mock_return
        ret = azure_chat_engine.complete_chat(messages=[ChatMessage(role="user", content="hello")])
        assert ret == "hi"

def test_openai_complete_chat_return(openai_mock_return: ChatCompletion, openai_chat_engine: OpenAIChatTarget):
    with patch("openai.resources.chat.Completions.create") as mock_create:
        mock_create.return_value = openai_mock_return
        ret = openai_chat_engine.complete_chat(messages=[ChatMessage(role="user", content="hello")])
        assert ret == "hi"


def test_azure_invalid_key_raises():
    os.environ[AzureOpenAIChatTarget.API_KEY_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        AzureOpenAIChatTarget(
            deployment_name="gpt-4",
            endpoint="https://mock.azure.com/",
            api_key="",
            api_version="some_version",
        )

def test_openai_invalid_key_raises():
    os.environ[OpenAIChatTarget.API_KEY_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        OpenAIChatTarget(
            deployment_name="gpt-4",
            endpoint="https://mock.azure.com/",
            api_key="",
        )


def test_azure_initialization_with_no_deployment_raises():
    os.environ[AzureOpenAIChatTarget.DEPLOYMENT_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        AzureOpenAIChatTarget()

def test_openai_initialization_with_no_deployment_raises():
    os.environ[OpenAIChatTarget.DEPLOYMENT_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        OpenAIChatTarget()


def test_azure_invalid_endpoint_raises():
    os.environ[AzureOpenAIChatTarget.ENDPOINT_URI_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        AzureOpenAIChatTarget(
            deployment_name="gpt-4",
            endpoint="",
            api_key="xxxxx",
            api_version="some_version",
        )

def test_openai_invalid_endpoint_raises():
    os.environ[OpenAIChatTarget.ENDPOINT_URI_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        OpenAIChatTarget(
            deployment_name="gpt-4",
            endpoint="",
            api_key="xxxxx",
        )
