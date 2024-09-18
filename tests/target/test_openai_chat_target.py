# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tests.mocks import get_sample_conversations

import os
import pytest

from unittest.mock import AsyncMock, MagicMock, patch
from openai import BadRequestError, RateLimitError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from pyrit.exceptions.exception_classes import EmptyResponseException
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import PromptRequestPiece
from pyrit.models import PromptRequestResponse
from pyrit.prompt_target.prompt_chat_target.openai_chat_target import OpenAIChatInterface
from pyrit.prompt_target import AzureOpenAITextChatTarget, OpenAIChatTarget


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
def azure_chat_target() -> AzureOpenAITextChatTarget:
    return AzureOpenAITextChatTarget(
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


@pytest.mark.asyncio
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
    azure_chat_target: AzureOpenAITextChatTarget,
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


@pytest.mark.asyncio
async def test_openai_validate_request_length(
    openai_chat_target: OpenAIChatTarget, sample_conversations: list[PromptRequestPiece]
):
    request = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await openai_chat_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_azure_openai_validate_request_length(
    azure_chat_target: AzureOpenAITextChatTarget, sample_conversations: list[PromptRequestPiece]
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
    azure_chat_target: AzureOpenAITextChatTarget, sample_conversations: list[PromptRequestPiece]
):
    request_piece = sample_conversations[0]
    request_piece.converted_value_data_type = "image_path"
    request = PromptRequestResponse(request_pieces=[request_piece])
    with pytest.raises(ValueError, match="This target only supports text prompt input."):
        await azure_chat_target.send_prompt_async(prompt_request=request)


def test_azure_invalid_key_raises():
    os.environ[AzureOpenAITextChatTarget.API_KEY_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        AzureOpenAITextChatTarget(
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
    os.environ[AzureOpenAITextChatTarget.DEPLOYMENT_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        AzureOpenAITextChatTarget()


def test_openai_initialization_with_no_deployment_raises():
    os.environ[OpenAIChatTarget.DEPLOYMENT_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        OpenAIChatTarget()


def test_azure_invalid_endpoint_raises():
    os.environ[AzureOpenAITextChatTarget.ENDPOINT_URI_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        AzureOpenAITextChatTarget(
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


@pytest.mark.asyncio
async def test_send_prompt_async_empty_response_retries(
    openai_mock_return: ChatCompletion, azure_chat_target: AzureOpenAITextChatTarget
):
    prompt_req_resp = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="user",
                conversation_id="12345679",
                original_value="hello",
                converted_value="hello",
                original_value_data_type="text",
                converted_value_data_type="text",
                prompt_target_identifier={"target": "target-identifier"},
                orchestrator_identifier={"test": "test"},
                labels={"test": "test"},
            )
        ]
    )
    # Make assistant response empty
    openai_mock_return.choices[0].message.content = ""
    with patch("openai.resources.chat.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = openai_mock_return
        azure_chat_target._memory = MagicMock(MemoryInterface)
        with pytest.raises(EmptyResponseException):
            await azure_chat_target.send_prompt_async(prompt_request=prompt_req_resp)
        assert mock_create.call_count == int(os.getenv("RETRY_MAX_NUM_ATTEMPTS"))


@pytest.mark.asyncio
async def test_send_prompt_async_rate_limit_exception_retries(azure_chat_target: AzureOpenAITextChatTarget):
    response = MagicMock()
    response.status_code = 429
    mock_complete_chat_async = AsyncMock(
        side_effect=RateLimitError("Rate Limit Reached", response=response, body="Rate limit reached")
    )
    setattr(azure_chat_target, "_complete_chat_async", mock_complete_chat_async)
    prompt_request = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="user", conversation_id="123", original_value="Hello")]
    )

    with pytest.raises(RateLimitError):
        await azure_chat_target.send_prompt_async(prompt_request=prompt_request)
        assert mock_complete_chat_async.call_count == int(os.getenv("RETRY_MAX_NUM_ATTEMPTS"))


@pytest.mark.asyncio
async def test_send_prompt_async_rate_limit_exception_adds_to_memory(azure_chat_target: AzureOpenAITextChatTarget):
    mock_memory = MagicMock()
    mock_memory.get_chat_messages_with_conversation_id.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()
    mock_memory.add_response_entries_to_memory = AsyncMock()

    azure_chat_target._memory = mock_memory

    response = MagicMock()
    response.status_code = 429
    mock_complete_chat_async = AsyncMock(
        side_effect=RateLimitError("Rate Limit Reached", response=response, body="Rate limit reached")
    )
    setattr(azure_chat_target, "_complete_chat_async", mock_complete_chat_async)
    prompt_request = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="user", conversation_id="123", original_value="Hello")]
    )

    with pytest.raises(RateLimitError) as rle:
        await azure_chat_target.send_prompt_async(prompt_request=prompt_request)
        azure_chat_target._memory.get_chat_messages_with_conversation_id.assert_called_once_with(conversation_id="123")
        azure_chat_target._memory.add_request_response_to_memory.assert_called_once_with(request=prompt_request)
        azure_chat_target._memory.add_response_entries_to_memory.assert_called_once()

    assert str(rle.value) == "Rate Limit Reached"


@pytest.mark.asyncio
async def test_send_prompt_async_bad_request_error(azure_chat_target: AzureOpenAITextChatTarget):
    response = MagicMock()
    response.status_code = 400
    mock_complete_chat_async = AsyncMock(
        side_effect=BadRequestError("Bad Request", response=response, body="Bad Request")
    )
    setattr(azure_chat_target, "_complete_chat_async", mock_complete_chat_async)

    prompt_request = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="user", conversation_id="123", original_value="Hello")]
    )
    with pytest.raises(BadRequestError) as bre:
        await azure_chat_target.send_prompt_async(prompt_request=prompt_request)
        assert str(bre.value == "Bad Request Error")


def test_parse_chat_completion_successful(azure_chat_target: AzureOpenAITextChatTarget):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "Test response message"
    result = azure_chat_target._parse_chat_completion(mock_response)
    assert result == "Test response message", "The response message was not parsed correctly"


def test_openai_parse_chat_completion_successful(openai_chat_target: OpenAIChatTarget):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "Test response message"
    result = openai_chat_target._parse_chat_completion(mock_response)
    assert result == "Test response message", "The response message was not parsed correctly"


def test_validate_request_too_many_request_pieces(azure_chat_target: AzureOpenAITextChatTarget):

    prompt_request = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(role="user", original_value="Hello", converted_value_data_type="text"),
            PromptRequestPiece(role="user", original_value="Second Request", converted_value_data_type="text"),
        ]
    )
    with pytest.raises(ValueError) as excinfo:
        azure_chat_target._validate_request(prompt_request=prompt_request)

    assert "target only supports a single prompt request piece" in str(
        excinfo.value
    ), "Error not raised for too many request pieces"


@pytest.mark.asyncio
async def test_send_prompt_async_empty_response_adds_to_memory(
    openai_mock_return: ChatCompletion, azure_chat_target: AzureOpenAITextChatTarget
):
    mock_memory = MagicMock()
    mock_memory.get_chat_messages_with_conversation_id.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()
    mock_memory.add_response_entries_to_memory = AsyncMock()

    azure_chat_target._memory = mock_memory

    prompt_req_resp = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="user",
                conversation_id="12345679",
                original_value="hello",
                converted_value="hello",
                original_value_data_type="text",
                converted_value_data_type="text",
                prompt_target_identifier={"target": "target-identifier"},
                orchestrator_identifier={"test": "test"},
                labels={"test": "test"},
            )
        ]
    )
    # Make assistant response empty
    openai_mock_return.choices[0].message.content = ""
    with patch("openai.resources.chat.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = openai_mock_return
        with pytest.raises(EmptyResponseException) as e:
            await azure_chat_target.send_prompt_async(prompt_request=prompt_req_resp)
            azure_chat_target._memory.get_chat_messages_with_conversation_id.assert_called_once_with(
                conversation_id="12345679"
            )
            azure_chat_target._memory.add_request_response_to_memory.assert_called_once_with(request=prompt_req_resp)
            azure_chat_target._memory.add_response_entries_to_memory.assert_called_once()
        assert str(e.value) == "Status Code: 204, Message: The chat returned an empty response."


@pytest.mark.asyncio
async def test_send_prompt_async_bad_request_error_adds_to_memory(azure_chat_target: AzureOpenAITextChatTarget):
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()
    mock_memory.add_response_entries_to_memory = AsyncMock()

    azure_chat_target._memory = mock_memory

    response = MagicMock()
    response.status_code = 400
    mock_complete_chat_async = AsyncMock(
        side_effect=BadRequestError("Bad Request", response=response, body="Bad Request")
    )
    setattr(azure_chat_target, "_complete_chat_async", mock_complete_chat_async)
    prompt_request = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="user", conversation_id="123", original_value="Hello")]
    )

    with pytest.raises(BadRequestError) as bre:
        await azure_chat_target.send_prompt_async(prompt_request=prompt_request)
        azure_chat_target._memory.get_conversation.assert_called_once_with(conversation_id="123")
        azure_chat_target._memory.add_request_response_to_memory.assert_called_once_with(request=prompt_request)
        azure_chat_target._memory.add_response_entries_to_memory.assert_called_once()

    assert str(bre.value) == "Bad Request"


@pytest.mark.asyncio
async def test_send_prompt_async(openai_mock_return: ChatCompletion, azure_chat_target: AzureOpenAITextChatTarget):
    prompt_req_resp = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="user",
                conversation_id="12345679",
                original_value="hello",
                converted_value="hello",
                original_value_data_type="text",
                converted_value_data_type="text",
                prompt_target_identifier={"target": "target-identifier"},
                orchestrator_identifier={"test": "test"},
                labels={"test": "test"},
            )
        ]
    )

    with patch("openai.resources.chat.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = openai_mock_return
        response: PromptRequestResponse = await azure_chat_target.send_prompt_async(prompt_request=prompt_req_resp)
        assert len(response.request_pieces) == 1
        assert response.request_pieces[0].converted_value == "hi"
