# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
from unittest.mock import patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

import pyrit.agent.red_teaming_bot
from pyrit.agent import RedTeamingBot
from pyrit.chat import AzureOpenAIChat
from pyrit.models import PromptTemplate
from pyrit.memory import FileMemory
from pyrit.common.path import HOME_PATH


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
def red_teaming_bot(chat_completion_engine: AzureOpenAIChat, tmp_path: pathlib.Path):
    attack_strategy = PromptTemplate.from_yaml_file(
        pathlib.Path(HOME_PATH)
        / "datasets"
        / "attack_strategies"
        / "multi_turn_chat"
        / "red_team_chatbot_with_objective.yaml"
    )

    file_memory = FileMemory(filepath=tmp_path / "test.json.memory")

    return RedTeamingBot(
        conversation_objective="Do bad stuff",
        chat_engine=chat_completion_engine,
        memory=file_memory,
        attack_strategy=attack_strategy,
    )


def test_complete_chat_user(red_teaming_bot: RedTeamingBot):
    with patch.object(red_teaming_bot._chat_engine, "complete_chat") as mock:
        mock.return_value = "Hello, this is a message sent by the assistant. How can i help you?"
        red_teaming_bot.complete_chat_user("hi, I am a victim chatbot, how can I help?")
        chats = red_teaming_bot.get_conversation_chat_messages()
        assert len(chats) == 3, f"Expected 3 chats, got {len(chats)}"
        red_teaming_bot.complete_chat_user("hi, I'm a new chat")
        chats = red_teaming_bot.get_conversation_chat_messages()
        assert chats is not None
        assert len(chats) == 5, f"Expected 5 chats, got {len(chats)}"
        assert chats[0].role == "system", "First chat should be system instructions"
        assert chats[1].role == "user"
        assert chats[2].role == "assistant"
        assert "Do bad stuff" in chats[0].content
        assert "Instructions" in chats[0].content


def test_complete_chat_user_calls_complete_chat(red_teaming_bot: RedTeamingBot):
    with patch.object(red_teaming_bot._chat_engine, "complete_chat") as mock:
        mock.return_value = "Hello, this is a message sent by the assistant. How can i help you?"
        red_teaming_bot.complete_chat_user("new chat")

        args, kwargs = mock.call_args_list[0]
        assert kwargs["messages"] is not None
        assert kwargs["messages"][0].role == "system"
        assert kwargs["messages"][1].content == "new chat"

        red_teaming_bot.complete_chat_user("new chat2")
        args2, kwargs2 = mock.call_args_list[1]

        assert kwargs2["messages"][0].role == "system"
        assert kwargs2["messages"][1].content == "new chat"
        assert kwargs2["messages"][1].role == "user"
        assert kwargs2["messages"][2].content == "Hello, this is a message sent by the assistant. How can i help you?"
        assert kwargs2["messages"][2].role == "assistant"
        assert kwargs2["messages"][3].content == "new chat2"
        assert kwargs2["messages"][3].role == "user"


def test_is_conversation_complete_false(red_teaming_bot: RedTeamingBot):
    with patch.object(red_teaming_bot._chat_engine, "complete_chat") as mock:
        mock.return_value = "Hello, this is a message sent by the assistant. How can i help you?"
        red_teaming_bot.complete_chat_user("hi, I am a victim chatbot, how can I help?")
        red_teaming_bot.complete_chat_user("hi, I am a victim chatbot, how can I help?")

        assert (
            red_teaming_bot.is_conversation_complete() is False
        ), "Conversation should not be complete, objective not realized"


def test_is_conversation_complete_emptyhistory_false(red_teaming_bot: RedTeamingBot):
    with patch.object(red_teaming_bot._chat_engine, "complete_chat") as mock:
        mock.return_value = "Hello, this is a message sent by the assistant. How can i help you?"
        assert red_teaming_bot.is_conversation_complete() is False, "Conversation is empty, objective not realized"


def test_is_conversation_complete_true(red_teaming_bot: RedTeamingBot):
    with patch.object(red_teaming_bot._chat_engine, "complete_chat") as mock:
        mock.return_value = "end token <|done|>"
        red_teaming_bot.complete_chat_user(message="bad stuff is done")
        assert (
            red_teaming_bot.is_conversation_complete() is True
        ), "Conversation should be complete, objective is realized"


def test_default_attack_strategy_set_with_end_token(chat_completion_engine: AzureOpenAIChat, tmp_path: pathlib.Path):
    file_memory = FileMemory(filepath=tmp_path / "test.json.memory")

    bot = RedTeamingBot(conversation_objective="Do bad stuff", chat_engine=chat_completion_engine, memory=file_memory)

    assert bot._attack_strategy is not None
    assert pyrit.agent.red_teaming_bot.RED_TEAM_CONVERSATION_END_TOKEN in bot._attack_strategy.template


def test_attack_strategy_without_token_raises(chat_completion_engine: AzureOpenAIChat, tmp_path: pathlib.Path):
    file_memory = FileMemory(filepath=tmp_path / "test.json.memory")

    invalid_strategy = PromptTemplate(template="This is a bad strategy")

    with pytest.raises(ValueError):
        RedTeamingBot(
            conversation_objective="Do bad stuff",
            attack_strategy=invalid_strategy,
            chat_engine=chat_completion_engine,
            memory=file_memory,
        )
