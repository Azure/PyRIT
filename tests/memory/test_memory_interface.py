# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from string import ascii_lowercase

from pyrit.memory import MemoryInterface
from pyrit.models import ChatMessage

from tests.mocks import memory_fixture as memory # noqa: F401


def generate_random_string(length: int = 10) -> str:
    return "".join(random.choice(ascii_lowercase) for _ in range(length))


def test_memory(memory_fixture: MemoryInterface):
    assert memory_fixture


def test_conversation_memory_empty_by_default(memory_fixture: MemoryInterface):
    expected_count = 0
    c = memory_fixture.get_all_prompt_entries()
    assert len(c) == expected_count


def test_count_of_memories_matches_number_of_conversations_added_1(
    memory_fixture: MemoryInterface,
):
    expected_count = 1
    message = ChatMessage(role="user", content="Hello")
    memory_fixture.add_chat_message_to_memory(conversation=message, conversation_id="1", labels={})
    c = memory_fixture.get_all_prompt_entries()
    assert len(c) == expected_count


def test_add_chat_message_to_memory_added(memory_fixture: MemoryInterface):
    expected_count = 3
    memory_fixture.add_chat_message_to_memory(conversation=ChatMessage(role="user", content="Hello 1"), conversation_id="1")
    memory_fixture.add_chat_message_to_memory(conversation=ChatMessage(role="user", content="Hello 2"), conversation_id="1")
    memory_fixture.add_chat_message_to_memory(conversation=ChatMessage(role="user", content="Hello 3"), conversation_id="1")
    assert len(memory_fixture.get_all_prompt_entries()) == expected_count


def test_add_chat_messages_to_memory_added(memory_fixture: MemoryInterface):
    messages = [
        ChatMessage(role="user", content="Hello 1"),
        ChatMessage(role="user", content="Hello 2"),
    ]

    memory_fixture.add_chat_messages_to_memory(conversations=messages, conversation_id="1")
    assert len(memory_fixture.get_all_prompt_entries()) == len(messages)
