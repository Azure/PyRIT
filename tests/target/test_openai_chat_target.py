# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pytest

from unittest.mock import AsyncMock, patch

from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.prompt_target.prompt_chat_target.openai_chat_target import OpenAIChatInterface
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from pyrit.prompt_target import AzureOpenAIChatTarget, OpenAIChatTarget
from tests.mocks import get_sample_conversations


@pytest.fixture
def sample_conversations() -> list[PromptRequestPiece]:
    return get_sample_conversations()


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
def azure_chat_target() -> AzureOpenAIChatTarget:
    return AzureOpenAIChatTarget(
        deployment_name="gpt-4",
        endpoint="https://mock.azure.com/",
        api_key="mock-api-key",
        api_version="some_version",
    )


@pytest.fixture
def openai_chat_target() -> OpenAIChatTarget:
    return OpenAIChatTarget(
        deployment_name="gpt-4",
        endpoint="https://mock.azure.com/",
        api_key="mock-api-key",
    )


@pytest.fixture
def prompt_request_response() -> PromptRequestResponse:
    return PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="user",
                conversation_id="1234",
                original_value="hello",
                converted_value="hello",
                prompt_target_identifier={"target": "target-identifier"},
                orchestrator_identifier={"test": "test"},
                labels={"test": "test"},
            )
        ]
    )


def execute_openai_send_prompt(
    target: OpenAIChatInterface,
    prompt_request_response: PromptRequestResponse,
    mock_return: ChatCompletion,
):
    with patch("openai.resources.chat.Completions.create") as mock_create:
        mock_create.return_value = mock_return
        response: PromptRequestResponse = target.send_prompt(prompt_request=prompt_request_response)
        assert len(response.request_pieces) == 1
        assert response.request_pieces[0].converted_value == "hi"


async def execute_openai_send_prompt_async(
    target: OpenAIChatInterface,
    prompt_request_response: PromptRequestResponse,
    mock_return: ChatCompletion,
):
    with patch("openai.resources.chat.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_return
        response: PromptRequestResponse = await target.send_prompt_async(prompt_request=prompt_request_response)
        assert len(response.request_pieces) == 1
        assert response.request_pieces[0].converted_value == "hi"


@pytest.mark.asyncio
async def test_azure_complete_chat_async_return(
    openai_mock_return: ChatCompletion,
    azure_chat_target: AzureOpenAIChatTarget,
    prompt_request_response: PromptRequestResponse,
):
    await execute_openai_send_prompt_async(azure_chat_target, prompt_request_response, openai_mock_return)


@pytest.mark.asyncio
async def test_openai_complete_chat_async_return(
    openai_mock_return: ChatCompletion,
    openai_chat_target: OpenAIChatTarget,
    prompt_request_response: PromptRequestResponse,
):
    await execute_openai_send_prompt_async(openai_chat_target, prompt_request_response, openai_mock_return)


def test_azure_complete_chat_return(
    openai_mock_return: ChatCompletion,
    azure_chat_target: AzureOpenAIChatTarget,
    prompt_request_response: PromptRequestResponse,
):
    execute_openai_send_prompt(azure_chat_target, prompt_request_response, openai_mock_return)


def test_openai_complete_chat_return(
    openai_mock_return: ChatCompletion,
    openai_chat_target: OpenAIChatTarget,
    prompt_request_response: PromptRequestResponse,
):
    execute_openai_send_prompt(openai_chat_target, prompt_request_response, openai_mock_return)


@pytest.mark.asyncio
async def test_openai_validate_request_length(
    openai_chat_target: OpenAIChatTarget, sample_conversations: list[PromptRequestPiece]
):
    request = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await openai_chat_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_azure_openai_validate_request_length(
    azure_chat_target: AzureOpenAIChatTarget, sample_conversations: list[PromptRequestPiece]
):
    request = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await azure_chat_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_openai_validate_prompt_type(
    openai_chat_target: OpenAIChatTarget, sample_conversations: list[PromptRequestPiece]
):
    request_piece = sample_conversations[0]
    request_piece.converted_value_data_type = "image_path"
    request = PromptRequestResponse(request_pieces=[request_piece])
    with pytest.raises(ValueError, match="This target only supports text prompt input."):
        await openai_chat_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_azure_openai_validate_prompt_type(
    azure_chat_target: AzureOpenAIChatTarget, sample_conversations: list[PromptRequestPiece]
):
    request_piece = sample_conversations[0]
    request_piece.converted_value_data_type = "image_path"
    request = PromptRequestResponse(request_pieces=[request_piece])
    with pytest.raises(ValueError, match="This target only supports text prompt input."):
        await azure_chat_target.send_prompt_async(prompt_request=request)


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
