# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import HTTPStatusError
import os
from openai import RateLimitError
import pytest

from pyrit.exceptions import EmptyResponseException, RateLimitException
from pyrit.memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import PromptRequestResponse, PromptRequestPiece
from pyrit.prompt_target import AzureMLChatTarget
from pyrit.models import ChatMessage
from pyrit.chat_message_normalizer import ChatMessageNop, GenericSystemSquash, ChatMessageNormalizer
from tests.mocks import get_sample_conversations
from tests.mocks import get_memory_interface


@pytest.fixture
def sample_conversations() -> list[PromptRequestPiece]:
    return get_sample_conversations()


@pytest.fixture
def memory() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.fixture
def aml_online_chat(memory) -> AzureMLChatTarget:
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        aml_online_chat = AzureMLChatTarget(
            endpoint="http://aml-test-endpoint.com",
            api_key="valid_api_key",
            extra_param1="sample",
            extra_param2=1.0,
        )
        return aml_online_chat


def test_initialization_with_required_parameters(
    aml_online_chat: AzureMLChatTarget,
):
    assert aml_online_chat._endpoint == "http://aml-test-endpoint.com"
    assert aml_online_chat._api_key == "valid_api_key"


def test_initialization_with_extra_model_parameters(aml_online_chat: AzureMLChatTarget):
    assert aml_online_chat._extra_parameters == {"extra_param1": "sample", "extra_param2": 1.0}


def test_set_env_configuration_vars_with_default_env_vars(aml_online_chat: AzureMLChatTarget):
    with patch("pyrit.prompt_target.azure_ml_chat_target.AzureMLChatTarget._initialize_vars") as mock_initialize_vars:
        aml_online_chat._set_env_configuration_vars()
        assert aml_online_chat.endpoint_uri_environment_variable == "AZURE_ML_MANAGED_ENDPOINT"
        assert aml_online_chat.api_key_environment_variable == "AZURE_ML_KEY"
        mock_initialize_vars.assert_called_once_with()


def test_set_env_configuration_vars_initializes_vars(aml_online_chat: AzureMLChatTarget):
    with patch("pyrit.prompt_target.azure_ml_chat_target.AzureMLChatTarget._initialize_vars") as mock_initialize_vars:
        aml_online_chat._set_env_configuration_vars(
            endpoint_uri_environment_variable="CUSTOM_ENDPOINT_ENV_VAR",
            api_key_environment_variable="CUSTOM_API_KEY_ENV_VAR",
        )
        mock_initialize_vars.assert_called_once_with()


def test_set_model_parameters_with_defaults(aml_online_chat: AzureMLChatTarget):
    aml_online_chat._set_model_parameters()
    assert aml_online_chat._max_new_tokens == 400
    assert aml_online_chat._temperature == 1.0
    assert aml_online_chat._top_p == 1.0
    assert aml_online_chat._repetition_penalty == 1.0


def test_set_model_parameters_with_custom_values(aml_online_chat: AzureMLChatTarget):
    aml_online_chat._set_model_parameters(
        max_new_tokens=500, temperature=0.8, top_p=0.9, repetition_penalty=1.2, custom_param="custom_value"
    )
    assert aml_online_chat._max_new_tokens == 500
    assert aml_online_chat._temperature == 0.8
    assert aml_online_chat._top_p == 0.9
    assert aml_online_chat._repetition_penalty == 1.2
    assert aml_online_chat._extra_parameters == {"custom_param": "custom_value"}


def test_set_model_parameters_partial_update(aml_online_chat: AzureMLChatTarget):
    aml_online_chat._set_model_parameters(temperature=0.5, custom_param="custom_value")
    assert aml_online_chat._max_new_tokens == 400
    assert aml_online_chat._temperature == 0.5
    assert aml_online_chat._top_p == 1.0
    assert aml_online_chat._repetition_penalty == 1.0
    assert aml_online_chat._extra_parameters == {"custom_param": "custom_value"}


def test_initialization_with_no_key_raises():
    os.environ[AzureMLChatTarget.api_key_environment_variable] = ""
    with patch.object(CentralMemory, "get_memory_instance", return_value=MagicMock()):
        with pytest.raises(ValueError):
            AzureMLChatTarget(endpoint="http://aml-test-endpoint.com")


def test_initialization_with_no_api_raises():
    os.environ[AzureMLChatTarget.endpoint_uri_environment_variable] = ""
    with patch.object(CentralMemory, "get_memory_instance", return_value=MagicMock()):
        with pytest.raises(ValueError):
            AzureMLChatTarget(api_key="xxxxx")


def test_get_headers_with_valid_api_key(aml_online_chat: AzureMLChatTarget):
    expected_headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer valid_api_key",
    }
    assert aml_online_chat._get_headers() == expected_headers


@pytest.mark.asyncio
async def test_complete_chat_async(aml_online_chat: AzureMLChatTarget):
    messages = [
        ChatMessage(role="user", content="user content"),
    ]

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async") as mock:
        mock_response = MagicMock()
        mock_response.json.return_value = {"output": "extracted response"}
        mock.return_value = mock_response
        response = await aml_online_chat._complete_chat_async(messages)
        assert response == "extracted response"
        mock.assert_called_once()


# The None parameter checks the default is the same as ChatMessageNop
@pytest.mark.asyncio
@pytest.mark.parametrize("message_normalizer", [None, ChatMessageNop()])
async def test_complete_chat_async_with_nop_normalizer(
    aml_online_chat: AzureMLChatTarget, message_normalizer: ChatMessageNormalizer
):
    if message_normalizer:
        aml_online_chat.chat_message_normalizer = message_normalizer

    messages = [
        ChatMessage(role="system", content="system content"),
        ChatMessage(role="user", content="user content"),
    ]

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock) as mock:
        mock_response = MagicMock()
        mock_response.json.return_value = {"output": "extracted response"}
        mock.return_value = mock_response
        response = await aml_online_chat._complete_chat_async(messages)
        assert response == "extracted response"

        args, kwargs = mock.call_args
        body = kwargs["request_body"]

        assert body
        assert len(body["input_data"]["input_string"]) == 2
        assert body["input_data"]["input_string"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_complete_chat_async_with_squashnormalizer(aml_online_chat: AzureMLChatTarget):
    aml_online_chat.chat_message_normalizer = GenericSystemSquash()

    messages = [
        ChatMessage(role="system", content="system content"),
        ChatMessage(role="user", content="user content"),
    ]

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock) as mock:
        mock_response = MagicMock()
        mock_response.json.return_value = {"output": "extracted response"}
        mock.return_value = mock_response
        response = await aml_online_chat._complete_chat_async(messages)
        assert response == "extracted response"

        args, kwargs = mock.call_args
        body = kwargs["request_body"]

        assert body
        assert len(body["input_data"]["input_string"]) == 1
        assert body["input_data"]["input_string"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_complete_chat_async_bad_json_response(aml_online_chat: AzureMLChatTarget):
    messages = [
        ChatMessage(role="user", content="user content"),
    ]

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock) as mock:
        mock_response = MagicMock()
        mock_response.json.return_value = {"bad response"}
        mock.return_value = mock_response
        with pytest.raises(TypeError):
            await aml_online_chat._complete_chat_async(messages)


@pytest.mark.asyncio
async def test_azure_ml_validate_request_length(
    aml_online_chat: AzureMLChatTarget, sample_conversations: list[PromptRequestPiece]
):
    request = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await aml_online_chat.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_azure_ml_validate_prompt_type(
    aml_online_chat: AzureMLChatTarget, sample_conversations: list[PromptRequestPiece]
):
    request_piece = sample_conversations[0]
    request_piece.converted_value_data_type = "image_path"
    request = PromptRequestResponse(request_pieces=[request_piece])
    with pytest.raises(ValueError, match="This target only supports text prompt input."):
        await aml_online_chat.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_send_prompt_async_bad_request_error_adds_to_memory(aml_online_chat: AzureMLChatTarget):
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()

    aml_online_chat._memory = mock_memory

    response = MagicMock()
    response.status_code = 400
    response.text = "Bad Request"
    mock_complete_chat_async = AsyncMock(
        side_effect=HTTPStatusError(message="Bad Request", request=MagicMock(), response=response)
    )
    setattr(aml_online_chat, "_complete_chat_async", mock_complete_chat_async)
    prompt_request = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="user", conversation_id="123", original_value="Hello")]
    )

    with pytest.raises(HTTPStatusError) as bre:
        await aml_online_chat.send_prompt_async(prompt_request=prompt_request)
        aml_online_chat._memory.get_conversation.assert_called_once_with(conversation_id="123")
        aml_online_chat._memory.add_request_response_to_memory.assert_called_once_with(request=prompt_request)

    assert str(bre.value) == "Bad Request"


@pytest.mark.asyncio
async def test_send_prompt_async_rate_limit_exception_adds_to_memory(aml_online_chat: AzureMLChatTarget):
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()

    aml_online_chat._memory = mock_memory

    response = MagicMock()
    response.status_code = 429
    mock_complete_chat_async = AsyncMock(
        side_effect=HTTPStatusError(message="Rate Limit Reached", request=MagicMock(), response=response)
    )
    setattr(aml_online_chat, "_complete_chat_async", mock_complete_chat_async)
    prompt_request = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="user", conversation_id="123", original_value="Hello")]
    )

    with pytest.raises(RateLimitException) as rle:
        await aml_online_chat.send_prompt_async(prompt_request=prompt_request)
        aml_online_chat._memory.get_conversation.assert_called_once_with(conversation_id="123")
        aml_online_chat._memory.add_request_response_to_memory.assert_called_once_with(request=prompt_request)

    assert str(rle.value) == "Status Code: 429, Message: Rate Limit Exception"


@pytest.mark.asyncio
async def test_send_prompt_async_rate_limit_exception_retries(aml_online_chat: AzureMLChatTarget):

    response = MagicMock()
    response.status_code = 429
    mock_complete_chat_async = AsyncMock(
        side_effect=RateLimitError("Rate Limit Reached", response=response, body="Rate limit reached")
    )
    setattr(aml_online_chat, "_complete_chat_async", mock_complete_chat_async)
    prompt_request = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="user", conversation_id="12345", original_value="Hello")]
    )

    with pytest.raises(RateLimitError):
        await aml_online_chat.send_prompt_async(prompt_request=prompt_request)
        assert mock_complete_chat_async.call_count == os.getenv("RETRY_MAX_NUM_ATTEMPTS")


@pytest.mark.asyncio
async def test_send_prompt_async_empty_response_retries(aml_online_chat: AzureMLChatTarget):

    response = MagicMock()
    response.status_code = 429
    mock_complete_chat_async = AsyncMock()
    mock_complete_chat_async.return_value = None

    setattr(aml_online_chat, "_complete_chat_async", mock_complete_chat_async)
    prompt_request = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="user", conversation_id="12345", original_value="Hello")]
    )

    with pytest.raises(EmptyResponseException):
        await aml_online_chat.send_prompt_async(prompt_request=prompt_request)
        assert mock_complete_chat_async.call_count == os.getenv("RETRY_MAX_NUM_ATTEMPTS")
