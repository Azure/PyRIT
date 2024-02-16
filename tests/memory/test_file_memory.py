# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import pathlib
from unittest.mock import patch
import pytest
import uuid

from tempfile import NamedTemporaryFile
from pathlib import Path

from pyrit.embedding.azure_text_embedding import AzureTextEmbedding

from pyrit.memory.file_memory import FileMemory
from pyrit.memory.memory_models import (
    ConversationMemoryEntry,
    ConversationMemoryEntryList,
)
from pyrit.models import ChatMessage
from pyrit.common.path import RESULTS_PATH

MEMORY_FILE_EXTENSION = ".json.memory"


@pytest.fixture
def chat_memory_json() -> dict:
    return {
        "conversations": [
            {
                "role": "user",
                "content": "Hello 1",
                "conversation_id": "1",
                "normalizer_id": "1",
                "uuid": "30f01cfb-b965-41aa-b33a-c3f8354d3538",
            },
            {
                "role": "assistant",
                "content": "Hello back",
                "conversation_id": "1",
                "normalizer_id": "1",
                "uuid": "30f01cfb-b965-41aa-b33a-c3f8354d3538",
            },
        ]
    }


@pytest.fixture
def simple_conversation() -> ConversationMemoryEntryList:
    id = uuid.uuid4()
    return ConversationMemoryEntryList(
        conversations=[
            ConversationMemoryEntry(
                uuid=id,
                role="user",
                content="Hello World!",
                normalizer_id="1",
                conversation_id="1",
                timestamp_in_ns=0,
            ),
            ConversationMemoryEntry(
                role="assistant",
                content="Hello from Bot!",
                conversation_id="1",
                normalizer_id="1",
                timestamp_in_ns=0,
            ),
            ConversationMemoryEntry(
                role="user",
                content="Hi, bot! How are you?",
                conversation_id="1",
                normalizer_id="1",
                timestamp_in_ns=0,
            ),
            ConversationMemoryEntry(
                role="user",
                content="I am unrelated",
                conversation_id="2",
                timestamp_in_ns=0,
            ),
            ConversationMemoryEntry(
                role="user",
                content="I am unrelated",
                conversation_id="3",
                normalizer_id="1",
                timestamp_in_ns=0,
            ),
        ]
    )


@pytest.fixture
def memory(simple_conversation: ConversationMemoryEntryList) -> FileMemory:
    with NamedTemporaryFile(suffix=".json.memory", delete=False) as tmp:
        m = FileMemory(filepath=tmp.name, embedding_model=None)
        for entry in simple_conversation.conversations:
            m.add_chat_message_to_memory(
                conversation=ChatMessage(role=entry.role, content=entry.content),
                conversation_id=entry.conversation_id,
                normalizer_id=entry.normalizer_id,
            )
        return m


def test_json_memory_handler_save_conversation():
    with NamedTemporaryFile(suffix=MEMORY_FILE_EXTENSION, delete=False) as tmp_file:
        storage_handler = FileMemory(filepath=tmp_file.name)
        msg = ChatMessage(role="user", content="Hello 1")
        conversation_id = "1"
        storage_handler.add_chat_message_to_memory(conversation=msg, conversation_id=conversation_id)
        json_text = json.loads(tmp_file.read())
        assert len(json_text.get("conversations")) == 1
        assert json_text.get("conversations")[0].get("role") == "user"
        assert json_text.get("conversations")[0].get("content") == "Hello 1"
    os.remove(tmp_file.name)


def test_json_memory_handler_default_filepath():
    storage_handler = FileMemory()
    expected_default_memory = pathlib.Path(RESULTS_PATH, storage_handler.default_memory_file).resolve()
    assert storage_handler.filepath == expected_default_memory


def test_json_memory_handler_set_filepath():
    with NamedTemporaryFile(suffix=MEMORY_FILE_EXTENSION, delete=False) as tmp_file:
        storage_handler = FileMemory(filepath=tmp_file.name)
        assert storage_handler.filepath == Path(tmp_file.name)
    os.remove(tmp_file.name)


def test_json_memory_handler_loads_conversation(chat_memory_json: dict):
    with NamedTemporaryFile(suffix=MEMORY_FILE_EXTENSION, delete=False) as tmp_file:
        tmp_file.write(json.dumps(chat_memory_json).encode("utf-8"))
        tmp_file.seek(0)
        storage_handler = FileMemory(filepath=tmp_file.name)
        record = storage_handler.get_all_memory()
        assert len(record) == 2
        assert record[0].role == "user"
        assert record[0].content == "Hello 1"
        assert record[0].conversation_id == "1"
    os.remove(tmp_file.name)


def test_json_memory_handler_addends_suffix_to_filepath():
    with NamedTemporaryFile(delete=False) as tmp_file:
        storage_handler = FileMemory(filepath=tmp_file.name)
        assert "".join(storage_handler.filepath.suffixes) == MEMORY_FILE_EXTENSION
    os.remove(tmp_file.name)


def test_json_memory_handler_only_loads_from_correct_extension():
    with NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
        with pytest.raises(ValueError):
            FileMemory(filepath=tmp_file.name)
    os.remove(tmp_file.name)


def test_json_memory_handler_saves_and_stores_the_same_data(
    simple_conversation: ConversationMemoryEntryList,
):
    with NamedTemporaryFile(suffix=MEMORY_FILE_EXTENSION, delete=False) as tmp_file:
        storage_handler = FileMemory(filepath=tmp_file.name)
        storage_handler.save_conversation_memory_entries(simple_conversation.conversations)

        records = storage_handler.get_all_memory()

        for k, v in records[0].dict().items():
            assert v == simple_conversation.conversations[0].dict()[k]
    os.remove(tmp_file.name)


def test_file_memory_get_memory_by_exact_match_matches(
    memory: FileMemory,
):
    mem = memory.get_memory_by_exact_match(memory_entry_content="Hello World!")
    assert len(mem) == 1
    assert mem[0]
    assert mem[0].content == "Hello World!"
    assert mem[0].role == "user"


def test_test_file_memory_get_memory_by_exact_match_no_match(
    memory: FileMemory,
):
    mem = memory.get_memory_by_exact_match(memory_entry_content="<No Match>")
    assert not mem


def test_file_memory_get_memory_by_conversation_id(memory: FileMemory):
    mem = memory.get_chat_messages_with_conversation_id(conversation_id="1")
    assert len(mem) == 3


def test_file_memory_get_memory_by_normalizer_id(memory: FileMemory):
    mem = memory.get_memories_with_normalizer_id(normalizer_id="1")
    assert len(mem) == 4


def test_file_memory_labels_included(memory: FileMemory):
    memory.add_chat_message_to_memory(
        conversation=ChatMessage(role="user", content="Hello 1"),
        conversation_id="333",
        labels=["label1", "label2"],
    )
    mem = memory.get_memories_with_conversation_id(conversation_id="333")
    assert len(mem) == 1
    assert mem[0].labels == ["label1", "label2"]


def test_explicit_embedding_model_set():
    embedding = AzureTextEmbedding(api_key="testkey", endpoint="testbase", deployment="deployment")

    with NamedTemporaryFile(suffix=".json.memory") as tmp:
        memory = FileMemory(filepath=tmp.name, embedding_model=embedding)
        assert memory.memory_embedding


def test_default_embedding_model_set_none():
    with (
        NamedTemporaryFile(suffix=".json.memory") as tmp,
        patch("pyrit.memory.file_memory.default_memory_embedding_factory") as mock,
    ):
        mock.return_value = None
        memory = FileMemory(filepath=tmp.name)
        assert memory.memory_embedding is None


def test_default_embedding_model_set_correctly():
    embedding = AzureTextEmbedding(api_key="testkey", endpoint="testbase", deployment="deployment")

    with (
        NamedTemporaryFile(suffix=".json.memory") as tmp,
        patch("pyrit.memory.file_memory.default_memory_embedding_factory") as mock,
    ):
        mock.return_value = embedding
        memory = FileMemory(filepath=tmp.name)
        assert memory.memory_embedding is embedding
