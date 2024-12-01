# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Generator
from unittest.mock import AsyncMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from pyrit.memory.central_memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import PromptRequestResponse, PromptRequestPiece
from pyrit.orchestrator.orchestrator_class import Orchestrator
from pyrit.prompt_target import OpenAIChatTarget

from tests.mocks import get_memory_interface
from tests.mocks import get_sample_conversations


@pytest.fixture
def memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.fixture
def sample_entries() -> list[PromptRequestPiece]:
    return get_sample_conversations()


@pytest.fixture
def openai_mock_return() -> ChatCompletion:
    return ChatCompletion(
        id="12345678-1a2b-3c4e5f-a123-12345678abcd",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hi, I'm adversary chat."),
                finish_reason="stop",
                logprobs=None,
            )
        ],
        created=1629389505,
        model="gpt-4",
    )


@pytest.fixture
def chat_completion_engine(memory_interface: MemoryInterface) -> OpenAIChatTarget:
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):
        return OpenAIChatTarget(deployment_name="test", endpoint="test", api_key="test")


@pytest.fixture
def azure_openai_target(memory_interface: MemoryInterface):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):
        return OpenAIChatTarget(
            deployment_name="test",
            endpoint="test",
            api_key="test",
        )


def test_set_system_prompt(azure_openai_target: OpenAIChatTarget):
    azure_openai_target.set_system_prompt(
        system_prompt="system prompt",
        conversation_id="1",
        orchestrator_identifier=Orchestrator().get_identifier(),
        labels={},
    )

    chats = azure_openai_target._memory._get_prompt_pieces_with_conversation_id(conversation_id="1")
    assert len(chats) == 1, f"Expected 1 chat, got {len(chats)}"
    assert chats[0].role == "system"
    assert chats[0].converted_value == "system prompt"


@pytest.mark.asyncio
async def test_send_prompt_user_no_system(
    azure_openai_target: OpenAIChatTarget,
    openai_mock_return: ChatCompletion,
    sample_entries: list[PromptRequestPiece],
):

    with patch("openai.resources.chat.AsyncCompletions.create", new_callable=AsyncMock) as mock:
        mock.return_value = openai_mock_return

        request = sample_entries[0]
        request.converted_value = "hi, I am a victim chatbot, how can I help?"

        await azure_openai_target.send_prompt_async(prompt_request=PromptRequestResponse(request_pieces=[request]))

        mock.assert_called_once()

        _, kwargs = mock.call_args

        assert kwargs["messages"]
        assert len(kwargs["messages"]) == 1
        assert kwargs["messages"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_set_system_prompt_adds_memory(azure_openai_target: OpenAIChatTarget):
    azure_openai_target.set_system_prompt(
        system_prompt="system prompt",
        conversation_id="1",
        orchestrator_identifier=Orchestrator().get_identifier(),
        labels={},
    )

    chats = azure_openai_target._memory._get_prompt_pieces_with_conversation_id(conversation_id="1")
    assert len(chats) == 1, f"Expected 1 chats, got {len(chats)}"
    assert chats[0].role == "system"


@pytest.mark.asyncio
async def test_send_prompt_with_system_calls_chat_complete(
    azure_openai_target: OpenAIChatTarget,
    openai_mock_return: ChatCompletion,
    sample_entries: list[PromptRequestPiece],
):

    with patch("openai.resources.chat.AsyncCompletions.create", new_callable=AsyncMock) as mock:
        mock.return_value = openai_mock_return

        azure_openai_target.set_system_prompt(
            system_prompt="system prompt",
            conversation_id="1",
            orchestrator_identifier=Orchestrator().get_identifier(),
            labels={},
        )

        request = sample_entries[0]
        request.converted_value = "hi, I am a victim chatbot, how can I help?"
        request.conversation_id = "1"

        await azure_openai_target.send_prompt_async(prompt_request=PromptRequestResponse(request_pieces=[request]))

        mock.assert_called_once()


@pytest.mark.asyncio
async def test_send_prompt_async_with_delay(
    azure_openai_target: OpenAIChatTarget,
    openai_mock_return: ChatCompletion,
    sample_entries: list[PromptRequestPiece],
):
    azure_openai_target._max_requests_per_minute = 10

    with patch("openai.resources.chat.AsyncCompletions.create", new_callable=AsyncMock) as mock:
        with patch("asyncio.sleep") as mock_sleep:
            mock.return_value = openai_mock_return

            request = sample_entries[0]
            request.converted_value = "hi, I am a victim chatbot, how can I help?"

            await azure_openai_target.send_prompt_async(prompt_request=PromptRequestResponse(request_pieces=[request]))

            mock.assert_called_once()
            mock_sleep.assert_called_once_with(6)  # 60/max_requests_per_minute
