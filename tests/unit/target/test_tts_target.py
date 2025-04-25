# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import uuid
from typing import MutableSequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import HTTPStatusError
from openai import RateLimitError
from unit.mocks import get_sample_conversations

from pyrit.common import net_utility
from pyrit.exceptions import RateLimitException
from pyrit.memory import MemoryInterface
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target import OpenAITTSTarget
from pyrit.prompt_target.openai.openai_tts_target import TTSResponseFormat


@pytest.fixture
def sample_conversations() -> MutableSequence[PromptRequestPiece]:
    return get_sample_conversations()


@pytest.fixture
def tts_target(patch_central_database) -> OpenAITTSTarget:
    return OpenAITTSTarget(model_name="test", endpoint="https://test.com", api_key="test")


def test_tts_initializes(tts_target: OpenAITTSTarget):
    assert tts_target


def test_tts_initializes_calls_get_required_parameters(patch_central_database):
    with patch("pyrit.common.default_values.get_required_value") as mock_get_required:
        target = OpenAITTSTarget(
            model_name="deploymenttest",
            endpoint="endpointtest",
            api_key="keytest",
        )

        assert mock_get_required.call_count == 1

        mock_get_required.assert_any_call(
            env_var_name=target.endpoint_environment_variable, passed_value="endpointtest"
        )


@pytest.mark.asyncio
async def test_tts_validate_request_length(
    tts_target: OpenAITTSTarget, sample_conversations: MutableSequence[PromptRequestPiece]
):
    request = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await tts_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_tts_validate_prompt_type(
    tts_target: OpenAITTSTarget, sample_conversations: MutableSequence[PromptRequestPiece]
):
    request_piece = sample_conversations[0]
    request_piece.converted_value_data_type = "image_path"
    request = PromptRequestResponse(request_pieces=[request_piece])
    with pytest.raises(ValueError, match="This target only supports text prompt input."):
        await tts_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_tts_validate_previous_conversations(
    tts_target: OpenAITTSTarget, sample_conversations: MutableSequence[PromptRequestPiece]
):
    request_piece = sample_conversations[0]

    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = sample_conversations
    mock_memory.add_request_response_to_memory = AsyncMock()

    tts_target._memory = mock_memory

    request = PromptRequestResponse(request_pieces=[request_piece])

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async") as mock_request:
        mock_request.return_value = MagicMock(content=b"audio data")
        with pytest.raises(ValueError, match="This target only supports a single turn conversation."):
            await tts_target.send_prompt_async(prompt_request=request)


@pytest.mark.parametrize("response_format", ["mp3", "ogg"])
@pytest.mark.asyncio
async def test_tts_send_prompt_file_save_async(
    patch_central_database,
    sample_conversations: MutableSequence[PromptRequestPiece],
    response_format: TTSResponseFormat,
) -> None:
    tts_target = OpenAITTSTarget(model_name="test", endpoint="test", api_key="test", response_format=response_format)

    request_piece = sample_conversations[0]
    request_piece.conversation_id = str(uuid.uuid4())
    request = PromptRequestResponse(request_pieces=[request_piece])
    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:
        return_value = MagicMock()
        return_value.content = b"audio data"
        mock_request.return_value = return_value
        response = await tts_target.send_prompt_async(prompt_request=request)

        file_path = response.get_value()
        assert file_path
        assert file_path.endswith(f".{response_format}")
        assert os.path.exists(file_path)
        data = open(file_path, "rb").read()
        assert data == b"audio data"
        os.remove(file_path)


testdata = [(400, "Bad Request", HTTPStatusError), (429, "Rate Limit Reached", RateLimitException)]


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code, error_text, exception_class", testdata)
async def test_tts_send_prompt_async_exception_adds_to_memory(
    tts_target: OpenAITTSTarget,
    sample_conversations: MutableSequence[PromptRequestPiece],
    status_code: int,
    error_text: str,
    exception_class: type[BaseException],
):
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()

    tts_target._memory = mock_memory

    response = MagicMock()
    response.status_code = status_code
    response.text = error_text
    mock_response_async = AsyncMock(
        side_effect=HTTPStatusError(message=response.text, request=MagicMock(), response=response)
    )

    setattr(net_utility, "make_request_and_raise_if_error_async", mock_response_async)

    request_piece = sample_conversations[0]
    request_piece.conversation_id = str(uuid.uuid4())
    request = PromptRequestResponse(request_pieces=[request_piece])

    with pytest.raises((exception_class)) as exc:
        await tts_target.send_prompt_async(prompt_request=request)
        tts_target._memory.get_conversation.assert_called_once_with(conversation_id=request_piece.conversation_id)

        tts_target._memory.add_request_response_to_memory.assert_called_once_with(request=request)

        assert response.text in str(exc.value)


@pytest.mark.asyncio
async def test_tts_send_prompt_async_rate_limit_exception_retries(
    tts_target: OpenAITTSTarget, sample_conversations: MutableSequence[PromptRequestPiece]
):
    response = MagicMock()
    response.status_code = 429
    response.text = "Rate Limit Reached"
    mock_response_async = AsyncMock(
        side_effect=RateLimitError(message=response.text, response=response, body="Rate limit reached")
    )

    setattr(net_utility, "make_request_and_raise_if_error_async", mock_response_async)
    request_piece = sample_conversations[0]
    request = PromptRequestResponse(request_pieces=[request_piece])

    with pytest.raises(RateLimitError):
        await tts_target.send_prompt_async(prompt_request=request)
        assert mock_response_async.call_count == os.getenv("RETRY_MAX_NUM_ATTEMPTS")


def test_is_json_response_supported(tts_target: OpenAITTSTarget):
    assert tts_target.is_json_response_supported() is False


@pytest.mark.asyncio
async def test_tts_target_no_api_version(sample_conversations: MutableSequence[PromptRequestPiece]):
    target = OpenAITTSTarget(
        api_key="test_key", endpoint="https://mock.azure.com", model_name="tts-model", api_version=None
    )
    request = PromptRequestResponse([sample_conversations[0]])

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"audio data"

        mock_request.return_value = mock_response

        await target.send_prompt_async(prompt_request=request)

        called_params = mock_request.call_args[1]["params"]
        assert "api-version" not in called_params


@pytest.mark.asyncio
async def test_tts_target_default_api_version(sample_conversations: MutableSequence[PromptRequestPiece]):
    target = OpenAITTSTarget(api_key="test_key", endpoint="https://mock.azure.com", model_name="tts-model")
    request = PromptRequestResponse([sample_conversations[0]])

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"audio data"

        mock_request.return_value = mock_response

        await target.send_prompt_async(prompt_request=request)

        called_params = mock_request.call_args[1]["params"]

        assert "api-version" in called_params
        assert called_params["api-version"] == "2025-02-01-preview"


@pytest.mark.asyncio
async def test_send_prompt_async_calls_refresh_auth_headers(tts_target):
    mock_memory = MagicMock(spec=MemoryInterface)
    mock_memory.get_conversation.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()

    tts_target._memory = mock_memory

    tts_target.refresh_auth_headers = MagicMock()
    tts_target._validate_request = MagicMock()
    tts_target._construct_request_body = AsyncMock(return_value={})

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async") as mock_make_request:
        mock_response = MagicMock()
        mock_response.content = b"audio data"
        mock_make_request.return_value = mock_response

        prompt_request = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="user",
                    original_value="test prompt",
                    converted_value="test prompt",
                    converted_value_data_type="text",
                )
            ]
        )
        await tts_target.send_prompt_async(prompt_request=prompt_request)

        tts_target.refresh_auth_headers.assert_called_once()
