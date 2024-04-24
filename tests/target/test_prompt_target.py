# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Generator
from unittest.mock import patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import PromptRequestResponse, PromptRequestPiece
from pyrit.orchestrator.orchestrator_class import Orchestrator
from pyrit.prompt_target import AzureOpenAIChatTarget

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
def chat_completion_engine() -> AzureOpenAIChatTarget:
    return AzureOpenAIChatTarget(deployment_name="test", endpoint="test", api_key="test")


@pytest.fixture
def azure_openai_target(memory_interface: MemoryInterface):

    return AzureOpenAIChatTarget(
        deployment_name="test",
        endpoint="test",
        api_key="test",
        memory=memory_interface,
    )


def test_set_system_prompt(azure_openai_target: AzureOpenAIChatTarget):
    azure_openai_target.set_system_prompt(
        system_prompt="system prompt",
        conversation_id="1",
        orchestrator_identifier=Orchestrator().get_identifier(),
        labels={},
    )

    chats = azure_openai_target._memory._get_prompt_pieces_with_conversation_id(conversation_id="1")
    assert len(chats) == 1, f"Expected 1 chat, got {len(chats)}"
    assert chats[0].role == "system"
    assert chats[0].converted_prompt_text == "system prompt"


def test_send_prompt_user_no_system(
    azure_openai_target: AzureOpenAIChatTarget,
    openai_mock_return: ChatCompletion,
    sample_entries: list[PromptRequestPiece],
):

    with patch("openai.resources.chat.Completions.create") as mock:
        mock.return_value = openai_mock_return

        request = sample_entries[0]
        request.converted_prompt_text = "hi, I am a victim chatbot, how can I help?"

        azure_openai_target.send_prompt(prompt_request=PromptRequestResponse(request_pieces=[request]))

        chats = azure_openai_target._memory._get_prompt_pieces_with_conversation_id(
            conversation_id=request.conversation_id
        )
        assert len(chats) == 2, f"Expected 2 chats, got {len(chats)}"
        assert chats[0].role == "user"
        assert chats[1].role == "assistant"


def test_send_prompt_with_system(
    azure_openai_target: AzureOpenAIChatTarget,
    openai_mock_return: ChatCompletion,
    sample_entries: list[PromptRequestPiece],
):

    with patch("openai.resources.chat.Completions.create") as mock:
        mock.return_value = openai_mock_return

        azure_openai_target.set_system_prompt(
            system_prompt="system prompt",
            conversation_id="1",
            orchestrator_identifier=Orchestrator().get_identifier(),
            labels={},
        )

        request = sample_entries[0]
        request.converted_prompt_text = "hi, I am a victim chatbot, how can I help?"
        request.conversation_id = "1"

        azure_openai_target.send_prompt(prompt_request=PromptRequestResponse(request_pieces=[request]))

        chats = azure_openai_target._memory._get_prompt_pieces_with_conversation_id(conversation_id="1")
        assert len(chats) == 3, f"Expected 3 chats, got {len(chats)}"
        assert chats[0].role == "system"
        assert chats[1].role == "user"


def test_send_prompt_with_system_calls_chat_complete(
    azure_openai_target: AzureOpenAIChatTarget,
    openai_mock_return: ChatCompletion,
    sample_entries: list[PromptRequestPiece],
):

    with patch("openai.resources.chat.Completions.create") as mock:
        mock.return_value = openai_mock_return

        azure_openai_target.set_system_prompt(
            system_prompt="system prompt",
            conversation_id="1",
            orchestrator_identifier=Orchestrator().get_identifier(),
            labels={},
        )

        request = sample_entries[0]
        request.converted_prompt_text = "hi, I am a victim chatbot, how can I help?"
        request.conversation_id = "1"

        azure_openai_target.send_prompt(prompt_request=PromptRequestResponse(request_pieces=[request]))

        mock.assert_called_once()
