# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, patch

import pytest

from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target import RealtimeTarget


@pytest.fixture
def target(duckdb_instance):
    return RealtimeTarget(
        api_key="test_key", endpoint="wss://test_url", deployment_name="test_deployment", api_version="v1"
    )


@pytest.mark.asyncio
async def test_connect_success(target):
    # Mock the websockets.connect method
    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
        await target.connect()
        mock_connect.assert_called_once_with(
            "wss://test_url/openai/realtime?api-version=v1&deployment=test_deployment&api-key=test_key",
            extra_headers={"Authorization": "Bearer test_key", "OpenAI-Beta": "realtime=v1"},
        )
        assert target.websocket == mock_connect.return_value


@pytest.mark.asyncio
async def test_send_prompt_async(target):
    request_piece = PromptRequestPiece(
        original_value="Hello",
        original_value_data_type="text",
        converted_value="Hello",
        converted_value_data_type="text",
        role="user",
    )
    prompt_request = PromptRequestResponse(request_pieces=[request_piece])

    with (
        patch("pyrit.prompt_target.openai.openai_realtime_target.RealtimeTarget.connect", next_callable=AsyncMock) as mock_connect,
        patch(
            "pyrit.prompt_target.openai.openai_realtime_target.RealtimeTarget.receive_events", new_callable=AsyncMock
        ) as mock_receive_events,
        patch(
            "pyrit.prompt_target.openai.openai_realtime_target.RealtimeTarget.send_response_create", new_callable=AsyncMock
        ) as mock_send_response_create,
        patch("pyrit.prompt_target.openai.openai_realtime_target.RealtimeTarget.send_text", new_callable=AsyncMock) as mock_send_text,
        patch(
            "pyrit.prompt_target.openai.openai_realtime_target.RealtimeTarget.save_audio", new_callable=AsyncMock
        ) as mock_save_audio,
        patch(
            "pyrit.prompt_target.openai.openai_realtime_target.RealtimeTarget.send_event", new_callable=AsyncMock
        ) as mock_send_event,
    ):

        mock_receive_events.return_value = ["", "Hello"]
        mock_save_audio.return_value = "response_audio.wav"

        response = await target.send_prompt_async(prompt_request=prompt_request)

        mock_connect.assert_called_once()
        mock_send_response_create.assert_called_once()
        mock_send_text.assert_called_once_with("Hello")
        mock_receive_events.assert_called_once()
        mock_save_audio.assert_called_once()
        mock_send_event.assert_called()

        assert response

        assert response.request_pieces[0].converted_value == "Hello"
        assert response.request_pieces[1].converted_value == "response_audio.wav"
        


@pytest.mark.asyncio
async def test_send_prompt_async_invalid_request(target):

    # Create a mock PromptRequestResponse with an invalid data type
    request_piece = PromptRequestPiece(
        original_value="Invalid",
        original_value_data_type="image_path",
        converted_value="Invalid",
        converted_value_data_type="image_path",
        role="user",
    )
    prompt_request = PromptRequestResponse(request_pieces=[request_piece])

    with pytest.raises(ValueError) as excinfo:
        target._validate_request(prompt_request=prompt_request)

    assert "This target only supports text and audio_path prompt input." == str(excinfo.value)
