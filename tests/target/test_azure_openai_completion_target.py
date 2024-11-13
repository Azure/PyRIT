# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

from pyrit.memory.central_memory import CentralMemory
from pyrit.models import PromptRequestPiece, PromptRequestResponse
import pytest
from openai.types.completion import Completion
from pyrit.memory.memory_interface import MemoryInterface
from openai.types.completion_choice import CompletionChoice
from openai.types.completion_usage import CompletionUsage

from pyrit.prompt_target import OpenAICompletionTarget
from tests.mocks import get_sample_conversations
from tests.mocks import get_memory_interface


@pytest.fixture
def openai_mock_return() -> Completion:
    return Completion(
        id="12345678-1a2b-3c4e5f-a123-12345678abcd",
        object="text_completion",
        choices=[
            CompletionChoice(
                index=0,
                text="hi",
                finish_reason="stop",
                logprobs=None,
            )
        ],
        created=1629389505,
        model="gpt-35-turbo",
        usage=CompletionUsage(
            prompt_tokens=1,
            total_tokens=2,
            completion_tokens=1,
        ),
    )


@pytest.fixture
def memory() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.fixture
def azure_completion_target(memory) -> OpenAICompletionTarget:
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        return OpenAICompletionTarget(
            deployment_name="gpt-35-turbo",
            endpoint="https://mock.azure.com/",
            api_key="mock-api-key",
            api_version="some_version",
        )


@pytest.fixture
def sample_conversations() -> list[PromptRequestPiece]:
    return get_sample_conversations()


@pytest.mark.asyncio
async def test_azure_completion_validate_request_length(
    azure_completion_target: OpenAICompletionTarget, sample_conversations: list[PromptRequestPiece]
):
    request = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await azure_completion_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_azure_completion_validate_prompt_type(
    azure_completion_target: OpenAICompletionTarget, sample_conversations: list[PromptRequestPiece]
):
    request_piece = sample_conversations[0]
    request_piece.converted_value_data_type = "image_path"
    request = PromptRequestResponse(request_pieces=[request_piece])
    with pytest.raises(ValueError, match="This target only supports text prompt input."):
        await azure_completion_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_azure_completion_validate_prev_convs(
    azure_completion_target: OpenAICompletionTarget, sample_conversations: list[PromptRequestPiece]
):
    request_piece = sample_conversations[0]
    azure_completion_target._memory.add_request_response_to_memory(
        request=PromptRequestResponse(request_pieces=[request_piece])
    )
    request = PromptRequestResponse(request_pieces=[request_piece])

    with pytest.raises(ValueError, match="This target only supports a single turn conversation."):
        await azure_completion_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_azure_complete_async_return(
    openai_mock_return: Completion,
    azure_completion_target: OpenAICompletionTarget,
    sample_conversations: list[PromptRequestPiece],
):
    request_piece = sample_conversations[0]
    request = PromptRequestResponse(request_pieces=[request_piece])
    with patch("openai.resources.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = openai_mock_return
        response: PromptRequestResponse = await azure_completion_target.send_prompt_async(prompt_request=request)
        assert len(response.request_pieces) == 1
        assert response.request_pieces[0].converted_value == "hi"


def test_azure_invalid_key_raises():
    with patch.object(CentralMemory, "get_memory_instance", return_value=MagicMock()):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                OpenAICompletionTarget(
                    deployment_name="gpt-4",
                    endpoint="https://mock.azure.com/",
                    api_key="",
                    api_version="some_version",
                )


def test_azure_initialization_with_no_deployment_raises():
    with patch.object(CentralMemory, "get_memory_instance", return_value=MagicMock()):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                OpenAICompletionTarget()


def test_azure_invalid_endpoint_raises():
    with patch.object(CentralMemory, "get_memory_instance", return_value=MagicMock()):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                OpenAICompletionTarget(
                    deployment_name="gpt-4",
                    endpoint="",
                    api_key="xxxxx",
                    api_version="some_version",
                )
