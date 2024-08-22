# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from httpx import HTTPStatusError
from openai import RateLimitError
import os
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from pyrit.common import net_utility
from pyrit.exceptions import RateLimitException
from pyrit.models import PromptRequestResponse, PromptRequestPiece
from pyrit.prompt_target import AzureTTSTarget

from pyrit.prompt_target.tts_target import TTSResponseFormat
from tests.mocks import get_sample_conversations


@pytest.fixture
def sample_conversations() -> list[PromptRequestPiece]:
    return get_sample_conversations()


@pytest.fixture
def tts_target() -> AzureTTSTarget:
    return AzureTTSTarget(deployment_name="test", endpoint="test", api_key="test")


def test_tts_initializes(tts_target: AzureTTSTarget):
    assert tts_target


def test_tts_initializes_calls_get_required_parameters():
    with patch("pyrit.common.default_values.get_required_value") as mock_get_required:
        AzureTTSTarget(
            deployment_name="deploymenttest",
            endpoint="endpointtest",
            api_key="keytest",
        )

        assert mock_get_required.call_count == 3

        mock_get_required.assert_any_call(
            env_var_name=AzureTTSTarget.DEPLOYMENT_ENVIRONMENT_VARIABLE, passed_value="deploymenttest"
        )

        mock_get_required.assert_any_call(
            env_var_name=AzureTTSTarget.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value="endpointtest"
        )

        mock_get_required.assert_any_call(
            env_var_name=AzureTTSTarget.API_KEY_ENVIRONMENT_VARIABLE, passed_value="keytest"
        )


@pytest.mark.asyncio
async def test_tts_validate_request_length(tts_target: AzureTTSTarget, sample_conversations: list[PromptRequestPiece]):
    request = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await tts_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_tts_validate_prompt_type(tts_target: AzureTTSTarget, sample_conversations: list[PromptRequestPiece]):
    request_piece = sample_conversations[0]
    request_piece.converted_value_data_type = "image_path"
    request = PromptRequestResponse(request_pieces=[request_piece])
    with pytest.raises(ValueError, match="This target only supports text prompt input."):
        await tts_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_tts_validate_previous_conversations(
    tts_target: AzureTTSTarget, sample_conversations: list[PromptRequestPiece]
):
    request_piece = sample_conversations[0]
    tts_target._memory.add_request_response_to_memory(request=PromptRequestResponse(request_pieces=[request_piece]))
    request = PromptRequestResponse(request_pieces=[request_piece])

    with pytest.raises(ValueError, match="This target only supports a single turn conversation."):
        await tts_target.send_prompt_async(prompt_request=request)


@pytest.mark.parametrize("response_format", ["mp3", "ogg"])
@pytest.mark.asyncio
async def test_tts_send_prompt_file_save_async(
    sample_conversations: list[PromptRequestPiece], response_format: TTSResponseFormat
) -> None:

    tts_target = AzureTTSTarget(
        deployment_name="test", endpoint="test", api_key="test", response_format=response_format
    )

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

        file_path = response.request_pieces[0].converted_value
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
    tts_target: AzureTTSTarget,
    sample_conversations: list[PromptRequestPiece],
    status_code: int,
    error_text: str,
    exception_class: type[BaseException],
):
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()
    mock_memory.add_response_entries_to_memory = AsyncMock()

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
        tts_target._memory.add_response_entries_to_memory.assert_called_once()

        assert response.text in str(exc.value)


@pytest.mark.asyncio
async def test_tts_send_prompt_async_rate_limit_exception_retries(
    tts_target: AzureTTSTarget, sample_conversations: list[PromptRequestPiece]
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
