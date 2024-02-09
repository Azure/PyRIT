# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import os

from contextlib import AbstractAsyncContextManager
from unittest.mock import AsyncMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from pyrit.chat.azure_openai_chat import AzureOpenAIChat
from pyrit.models import ChatMessage

_loop = asyncio.get_event_loop()


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
def chat_engine() -> AzureOpenAIChat:
    return AzureOpenAIChat(
        deployment_name="gpt-4",
        endpoint="https://mock.azure.com/",
        api_key="mock-api-key",
        api_version="some_version",
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
def test_complete_chat_async_return(mock_chat_create: AsyncMock, chat_engine: AzureOpenAIChat):
    ret = _loop.run_until_complete(
        chat_engine.complete_chat_async(messages=[ChatMessage(role="user", content="hello")])
    )
    assert ret == "hi"


def test_complete_chat_return(openai_mock_return: ChatCompletion, chat_engine: AzureOpenAIChat):
    with patch("openai.resources.chat.Completions.create") as mock_create:
        mock_create.return_value = openai_mock_return
        ret = chat_engine.complete_chat(messages=[ChatMessage(role="user", content="hello")])
        assert ret == "hi"


def test_invalid_key_raises():
    os.environ[AzureOpenAIChat.API_KEY_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        AzureOpenAIChat(
            deployment_name="gpt-4",
            endpoint="https://mock.azure.com/",
            api_key="",
            api_version="some_version",
        )


def test_invalid_endpoint_raises():
    os.environ[AzureOpenAIChat.ENDPOINT_URI_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        AzureOpenAIChat(
            deployment_name="gpt-4",
            endpoint="",
            api_key="xxxxx",
            api_version="some_version",
        )
