# Copyright (c) Adriano Maia <adriano@drstrange.wtf>
# Licensed under the MIT license.

import os

from unittest.mock import AsyncMock, patch

import pytest
import httpx
from pyrit.prompt_target import OllamaChatTarget
from pyrit.models import ChatMessage

@pytest.fixture
def ollama_mock_return() -> dict:
    return {"model":"mistral","created_at":"2024-04-13T16:14:52.69602Z","message":{"role":"assistant","content":" Hello."},"done":True,"total_duration":254579625,"load_duration":276542,"prompt_eval_count":20,"prompt_eval_duration":222911000,"eval_count":3,"eval_duration":30879000}

@pytest.fixture
def ollama_chat_engine() -> OllamaChatTarget:
    return OllamaChatTarget(
        endpoint_uri="http://mock.ollama.com:11434/api/chat",
        model_name="mistral",
    )

@pytest.mark.asyncio
async def test_ollama_complete_chat_async_return(ollama_chat_engine: OllamaChatTarget):
    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = httpx.Response(200, json={"model":"mistral","created_at":"2024-04-13T16:14:52.69602Z","message":{"role":"assistant","content":" Hello."},"done":True,"total_duration":254579625,"load_duration":276542,"prompt_eval_count":20,"prompt_eval_duration":222911000,"eval_count":3,"eval_duration":30879000})
        ret = await ollama_chat_engine._complete_chat_async(messages=[ChatMessage(role="user", content="hello")])
        assert ret == " Hello."

def test_ollama_complete_chat_return(ollama_chat_engine: OllamaChatTarget):
    with patch("pyrit.common.net_utility.make_request_and_raise_if_error") as mock_create:
        mock_create.return_value = httpx.Response(200, json={"model":"mistral","created_at":"2024-04-13T16:14:52.69602Z","message":{"role":"assistant","content":" Hello."},"done":True,"total_duration":254579625,"load_duration":276542,"prompt_eval_count":20,"prompt_eval_duration":222911000,"eval_count":3,"eval_duration":30879000})
        ret = ollama_chat_engine._complete_chat(messages=[ChatMessage(role="user", content="hello")])
        assert ret == " Hello."

def test_ollama_invalid_model_raises():
    os.environ[OllamaChatTarget.MODEL_NAME_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        OllamaChatTarget(
            endpoint_uri="http://mock.ollama.com:11434/api/chat",
            model_name="",
        )   

def test_ollama_invalid_endpoint_uri_raises():
    os.environ[OllamaChatTarget.ENDPOINT_URI_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        OllamaChatTarget(
            endpoint_uri="",
            model_name="mistral",
        )

