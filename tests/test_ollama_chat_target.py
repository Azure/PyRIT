# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from unittest.mock import AsyncMock, patch

import pytest
import httpx
from pyrit.prompt_target import OllamaChatTarget
from pyrit.models import PromptRequestResponse, PromptRequestPiece
from pyrit.models import ChatMessage
from tests.mocks import get_sample_conversations


@pytest.fixture
def sample_conversations() -> list[PromptRequestPiece]:
    return get_sample_conversations()


@pytest.fixture
def ollama_mock_return() -> dict:
    return {
        "model": "mistral",
        "created_at": "2024-04-13T16:14:52.69602Z",
        "message": {"role": "assistant", "content": " Hello."},
        "done": True,
        "total_duration": 254579625,
        "load_duration": 276542,
        "prompt_eval_count": 20,
        "prompt_eval_duration": 222911000,
        "eval_count": 3,
        "eval_duration": 30879000,
    }


@pytest.fixture
def ollama_chat_engine() -> OllamaChatTarget:
    return OllamaChatTarget(
        endpoint="http://mock.ollama.com:11434/api/chat",
        model_name="mistral",
    )


@pytest.mark.asyncio
async def test_ollama_complete_chat_async_return(ollama_chat_engine: OllamaChatTarget):
    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = httpx.Response(
            200,
            json={
                "model": "mistral",
                "created_at": "2024-04-13T16:14:52.69602Z",
                "message": {"role": "assistant", "content": " Hello."},
                "done": True,
                "total_duration": 254579625,
                "load_duration": 276542,
                "prompt_eval_count": 20,
                "prompt_eval_duration": 222911000,
                "eval_count": 3,
                "eval_duration": 30879000,
            },
        )
        ret = await ollama_chat_engine._complete_chat_async(messages=[ChatMessage(role="user", content="hello")])
        assert ret == " Hello."


def test_ollama_invalid_model_raises():
    os.environ[OllamaChatTarget.MODEL_NAME_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        OllamaChatTarget(
            endpoint="http://mock.ollama.com:11434/api/chat",
            model_name="",
        )


def test_ollama_invalid_endpoint_raises():
    os.environ[OllamaChatTarget.ENDPOINT_URI_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        OllamaChatTarget(
            endpoint="",
            model_name="mistral",
        )


@pytest.mark.asyncio
async def test_ollama_validate_request_length(
    ollama_chat_engine: OllamaChatTarget, sample_conversations: list[PromptRequestPiece]
):
    request = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await ollama_chat_engine.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_ollama_validate_prompt_type(
    ollama_chat_engine: OllamaChatTarget, sample_conversations: list[PromptRequestPiece]
):
    request_piece = sample_conversations[0]
    request_piece.converted_value_data_type = "image_path"
    request = PromptRequestResponse(request_pieces=[request_piece])
    with pytest.raises(ValueError, match="This target only supports text prompt input."):
        await ollama_chat_engine.send_prompt_async(prompt_request=request)
