# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
from unittest.mock import patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from pyrit.chat import AzureOpenAIChat
from pyrit.memory import FileMemory
from pyrit.prompt_target import AzureOpenAIChatTarget


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
def chat_completion_engine() -> AzureOpenAIChat:
    return AzureOpenAIChat(deployment_name="test", endpoint="test", api_key="test")


@pytest.fixture
def azure_openai_target(chat_completion_engine: AzureOpenAIChat, tmp_path: pathlib.Path):
    file_memory = FileMemory(filepath=tmp_path / "target_test.json.memory")

    return AzureOpenAIChatTarget(
        deployment_name="test",
        endpoint="test",
        api_key="test",
        memory=file_memory,
    )


def test_set_system_prompt(azure_openai_target: AzureOpenAIChatTarget):
    azure_openai_target.set_system_prompt(prompt="system prompt", conversation_id="1", normalizer_id="2")

    chats = azure_openai_target.memory.get_memories_with_conversation_id(conversation_id="1")
    assert len(chats) == 1, f"Expected 1 chat, got {len(chats)}"
    assert chats[0].role == "system"
    assert chats[0].content == "system prompt"


def test_complete_chat_user_no_system(azure_openai_target: AzureOpenAIChatTarget, openai_mock_return: ChatCompletion):
    with patch("openai.resources.chat.Completions.create") as mock:
        mock.return_value = openai_mock_return
        azure_openai_target.send_prompt(
            normalized_prompt="hi, I am a victim chatbot, how can I help?", conversation_id="1", normalizer_id="2"
        )

        chats = azure_openai_target.memory.get_memories_with_conversation_id(conversation_id="1")
        assert len(chats) == 2, f"Expected 2 chats, got {len(chats)}"
        assert chats[0].role == "user"
        assert chats[1].role == "assistant"


def test_complete_chat_user_with_system(azure_openai_target: AzureOpenAIChatTarget, openai_mock_return: ChatCompletion):
    with patch("openai.resources.chat.Completions.create") as mock:
        mock.return_value = openai_mock_return

        azure_openai_target.set_system_prompt(prompt="system prompt", conversation_id="1", normalizer_id="2")

        azure_openai_target.send_prompt(
            normalized_prompt="hi, I am a victim chatbot, how can I help?", conversation_id="1", normalizer_id="2"
        )

        chats = azure_openai_target.memory.get_memories_with_conversation_id(conversation_id="1")
        assert len(chats) == 3, f"Expected 3 chats, got {len(chats)}"
        assert chats[0].role == "system"
        assert chats[1].role == "user"


def test_complete_chat_user_with_system_calls_chat_complete(
    azure_openai_target: AzureOpenAIChatTarget, openai_mock_return: ChatCompletion
):
    with patch("openai.resources.chat.Completions.create") as mock:
        mock.return_value = openai_mock_return

        azure_openai_target.set_system_prompt(prompt="system prompt", conversation_id="1", normalizer_id="2")

        azure_openai_target.send_prompt(
            normalized_prompt="hi, I am a victim chatbot, how can I help?", conversation_id="1", normalizer_id="2"
        )

        mock.assert_called_once()
