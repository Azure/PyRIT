# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.prompt_target import AzureTTSTarget

from tests.mocks import get_sample_conversations


@pytest.fixture
def sample_conversations() -> list[PromptRequestPiece]:
    return get_sample_conversations()

@pytest.fixture
def tts_target() -> AzureTTSTarget:
    return AzureTTSTarget(
        deployment_name="test",
        endpoint="test",
        api_key="test"
    )

    
def test_tts_initializes(tts_target: AzureTTSTarget):
    assert tts_target

def test_tts_initializes_calls_get_required_parameters():
    with patch('pyrit.common.default_values.get_required_value') as mock_get_required:
        AzureTTSTarget(
            deployment_name="deploymenttest",
            endpoint="endpointtest",
            api_key="keytest",   
        )

        assert mock_get_required.call_count == 3

        mock_get_required.assert_any_call(
            env_var_name=AzureTTSTarget.DEPLOYMENT_ENVIRONMENT_VARIABLE,
            passed_value='deploymenttest'
        )

        mock_get_required.assert_any_call(
            env_var_name=AzureTTSTarget.ENDPOINT_URI_ENVIRONMENT_VARIABLE,
            passed_value='endpointtest'
        )

        mock_get_required.assert_any_call(
            env_var_name=AzureTTSTarget.API_KEY_ENVIRONMENT_VARIABLE,
            passed_value='keytest'
        )

@pytest.mark.asyncio
async def test_tts_validate_request_length(tts_target: AzureTTSTarget, sample_conversations: list[PromptRequestPiece]):
    request = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await tts_target.send_prompt_async(prompt_request=request)

@pytest.mark.asyncio
async def test_tts_validate_prompt_type(tts_target: AzureTTSTarget, sample_conversations: list[PromptRequestPiece]):
    request_piece = sample_conversations[0]
    request_piece.converted_prompt_data_type = "image_path"
    request = PromptRequestResponse(request_pieces=[request_piece])
    with pytest.raises(ValueError, match="This target only supports text prompt input."):
        await tts_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_tts_validate_previous_conversations(tts_target: AzureTTSTarget, sample_conversations: list[PromptRequestPiece]):
    tts_target._memory.add_request_pieces_to_memory(request_pieces=sample_conversations)
    request_piece = sample_conversations[0]
    request = PromptRequestResponse(request_pieces=[request_piece])

    with pytest.raises(ValueError, match="This target only supports a single turn conversation."):
        await tts_target.send_prompt_async(prompt_request=request)

@pytest.mark.parametrize("response_format", ["mp3", "ogg"])
@pytest.mark.asyncio
async def test_tts_send_prompt_file_save_async(
    sample_conversations: list[PromptRequestPiece],
    response_format: str
) -> None:

    tts_target = AzureTTSTarget(
        deployment_name="test",
        endpoint="test",
        api_key="test",
        response_format=response_format
    )

    request_piece = sample_conversations[0]
    request = PromptRequestResponse(request_pieces=[request_piece])
    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock) as mock_request:
        return_value = MagicMock()
        return_value.content = b"audio data"
        mock_request.return_value = return_value
        response = await tts_target.send_prompt_async(prompt_request=request)
        assert response
        assert response.request_pieces[0].converted_prompt_data_type == "audio_path"
        file_path = response.request_pieces[0].converted_prompt_text
        assert file_path
        assert file_path.endswith(f".{response_format}")
        assert os.path.exists(file_path)
        data = open(file_path, "rb").read()
        assert data == b"audio data"
        os.remove(file_path)

@pytest.mark.asyncio
async def test_tts_send_prompt_adds_memory_async(sample_conversations: list[PromptRequestPiece]) -> None:

    mock_memory = MagicMock()
    tts_target = AzureTTSTarget(
        deployment_name="test",
        endpoint="test",
        api_key="test",
        memory=mock_memory
    )

    request_piece = sample_conversations[0]
    request = PromptRequestResponse(request_pieces=[request_piece])
    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock) as mock_request:
        return_value = MagicMock()
        return_value.content = b"audio data"
        mock_request.return_value = return_value

        await tts_target.send_prompt_async(prompt_request=request)

        assert mock_memory.add_request_pieces_to_memory.called, "Request and Response need to be added to memory"
        assert mock_memory.add_response_entries_to_memory.called, "Request and Response need to be added to memory"