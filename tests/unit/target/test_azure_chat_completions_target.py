import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from azure.core.exceptions import HttpResponseError
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target.azure_chat_completions_target import AzureChatCompletionsTarget
from pyrit.exceptions.exception_classes import EmptyResponseException, RETRY_MAX_NUM_ATTEMPTS

@pytest.fixture
def sample_conversations() -> list[PromptRequestPiece]:
    return [
        PromptRequestPiece(
            role="user",
            original_value="Test prompt",
            conversation_id=str(uuid.uuid4()),
        )
    ]

@pytest.fixture
def chat_target(patch_central_database) -> AzureChatCompletionsTarget:
    return AzureChatCompletionsTarget(endpoint="test", api_key="test")

def test_chat_target_initializes(chat_target: AzureChatCompletionsTarget):
    assert chat_target

def test_chat_target_initializes_calls_get_required_parameters(patch_central_database):
    with patch("pyrit.common.default_values.get_required_value") as mock_get_required:

        mock_get_required.return_value = "test"
        target = AzureChatCompletionsTarget(endpoint="endpointtest", api_key="keytest")

        assert mock_get_required.call_count == 2

        mock_get_required.assert_any_call(
            env_var_name=target.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value="endpointtest"
        )
        mock_get_required.assert_any_call(
            env_var_name=target.API_KEY_ENVIRONMENT_VARIABLE, passed_value="keytest"
        )

@pytest.mark.asyncio
async def test_chat_target_validate_request_length(
    chat_target: AzureChatCompletionsTarget, sample_conversations: list[PromptRequestPiece]
):
    request = PromptRequestResponse(request_pieces=sample_conversations * 2)
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await chat_target.send_prompt_async(prompt_request=request)

@pytest.mark.asyncio
async def test_chat_target_validate_prompt_type(
    chat_target: AzureChatCompletionsTarget, sample_conversations: list[PromptRequestPiece]
):
    request_piece = sample_conversations[0]
    request_piece.converted_value_data_type = "image"
    request = PromptRequestResponse(request_pieces=[request_piece])
    with pytest.raises(ValueError, match="This target only supports text prompt input."):
        await chat_target.send_prompt_async(prompt_request=request)

@pytest.mark.asyncio
async def test_chat_target_send_prompt_async_filter_no_exception(
    chat_target: AzureChatCompletionsTarget, sample_conversations: list[PromptRequestPiece]
):
    with patch.object(chat_target._client, "complete", new_callable=AsyncMock) as mock_complete:
        response_message = "Bad Request content_filter"
        mock_error = HttpResponseError(message=response_message)
        mock_error.error = MagicMock()
        mock_error.error.innererror = {"content_filter_result": {}}

        mock_complete.side_effect = mock_error

        request_piece = sample_conversations[0]
        request = PromptRequestResponse(request_pieces=[request_piece])

        response = await chat_target.send_prompt_async(prompt_request=request)

        assert response_message in response.request_pieces[0].converted_value
        assert response.request_pieces[0].converted_value_data_type == "error"

        mock_complete.assert_called_once()

@pytest.mark.asyncio
async def test_chat_target_send_prompt_async_non_filter_exception(
    chat_target: AzureChatCompletionsTarget, sample_conversations: list[PromptRequestPiece]
):
    with patch.object(chat_target._client, "complete", new_callable=AsyncMock) as mock_complete:
        response_message = "Bad Request unrelated to filter"
        mock_error = HttpResponseError(message=response_message)
        mock_error.error = MagicMock()

        mock_complete.side_effect = mock_error

        request_piece = sample_conversations[0]
        request = PromptRequestResponse(request_pieces=[request_piece])

        with pytest.raises(HttpResponseError):
            await chat_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_chat_target_send_prompt_async_empty_response_exception(
    chat_target: AzureChatCompletionsTarget, sample_conversations: list[PromptRequestPiece]
):
    with patch.object(chat_target._client, "complete", new_callable=AsyncMock) as mock_complete:

        mock_return = MagicMock()
        mock_return.choices = [MagicMock()]
        mock_return.choices[0].message = MagicMock()
        mock_return.choices[0].message.content = ""

        mock_complete.return_value = mock_return

        request_piece = sample_conversations[0]
        request = PromptRequestResponse(request_pieces=[request_piece])

        with pytest.raises(EmptyResponseException):
            await chat_target.send_prompt_async(prompt_request=request)

        # assert it has retried
        assert mock_complete.call_count == RETRY_MAX_NUM_ATTEMPTS


@pytest.mark.asyncio
async def test_chat_target_send_prompt_async_success(
    chat_target: AzureChatCompletionsTarget, sample_conversations: list[PromptRequestPiece]
):
    with patch.object(chat_target._client, "complete", new_callable=AsyncMock) as mock_complete:

        mock_return = MagicMock()
        mock_return.choices = [MagicMock()]
        mock_return.choices[0].message = MagicMock()
        mock_return.choices[0].message.content = "response"
        mock_return.choices[0].finish_reason = "stop"


        mock_complete.return_value = mock_return

        request_piece = sample_conversations[0]
        request = PromptRequestResponse(request_pieces=[request_piece])

        response = await chat_target.send_prompt_async(prompt_request=request)

        assert mock_complete.call_count == 1
        assert response.request_pieces[0].converted_value == "response"
        assert response.request_pieces[0].converted_value_data_type == "text"


@pytest.mark.asyncio
async def test_chat_target_send_prompt_async_error(
    chat_target: AzureChatCompletionsTarget, sample_conversations: list[PromptRequestPiece]
):
    with patch.object(chat_target._client, "complete", new_callable=AsyncMock) as mock_complete:

        mock_return = MagicMock()
        mock_return.choices = [MagicMock()]
        mock_return.choices[0].message = MagicMock()
        mock_return.choices[0].message.content = "response"
        mock_return.choices[0].finish_reason = "not_stop"


        mock_complete.return_value = mock_return

        request_piece = sample_conversations[0]
        request = PromptRequestResponse(request_pieces=[request_piece])

        response = await chat_target.send_prompt_async(prompt_request=request)

        assert mock_complete.call_count == 1
        assert response.request_pieces[0].converted_value == "response"
        assert response.request_pieces[0].converted_value_data_type == "error"



def test_is_json_response_supported(chat_target: AzureChatCompletionsTarget):
    assert chat_target.is_json_response_supported() is True
