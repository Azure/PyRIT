# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from string import ascii_lowercase

import pytest
from sqlalchemy import inspect

from pyrit.memory import DuckDBMemory, MemoryInterface
from pyrit.models import ChatMessage
from pyrit.memory.memory_models import ConversationData


@pytest.fixture
def memory() -> MemoryInterface:  # type: ignore
    # Create an in-memory DuckDB engine
    duckdb_memory = DuckDBMemory(db_path=":memory:")

    # Reset the database to ensure a clean state
    duckdb_memory.reset_database()
    inspector = inspect(duckdb_memory.engine)

    # Verify that tables are created as expected
    assert "ConversationStore" in inspector.get_table_names(), "ConversationStore table not created."
    assert "EmbeddingStore" in inspector.get_table_names(), "EmbeddingStore table not created."

    yield duckdb_memory
    duckdb_memory.dispose_engine()


def generate_random_string(length: int = 10) -> str:
    return "".join(random.choice(ascii_lowercase) for _ in range(length))


def test_memory(memory: MemoryInterface):
    assert memory


def test_conversation_memory_empty_by_default(memory: MemoryInterface):
    expected_count = 0
    c = memory.get_all_memory(ConversationData)
    assert len(c) == expected_count


def test_count_of_memories_matches_number_of_conversations_added_1(
    memory: MemoryInterface,
):
    expected_count = 1
    message = ChatMessage(role="user", content="Hello")
    memory.add_chat_message_to_memory(conversation=message, conversation_id="1", labels=[])
    c = memory.get_all_memory(ConversationData)
    assert len(c) == expected_count


def test_add_chate_message_to_memory_added(memory: MemoryInterface):
    expected_count = 3
    memory.add_chat_message_to_memory(conversation=ChatMessage(role="user", content="Hello 1"), conversation_id="1")
    memory.add_chat_message_to_memory(conversation=ChatMessage(role="user", content="Hello 2"), conversation_id="1")
    memory.add_chat_message_to_memory(conversation=ChatMessage(role="user", content="Hello 3"), conversation_id="1")
    assert len(memory.get_all_memory(ConversationData)) == expected_count


def test_add_chate_messages_to_memory_added(memory: MemoryInterface):
    messages = [
        ChatMessage(role="user", content="Hello 1"),
        ChatMessage(role="user", content="Hello 2"),
    ]

    memory.add_chat_messages_to_memory(conversations=messages, conversation_id="1")
    assert len(memory.get_all_memory(ConversationData)) == len(messages)
