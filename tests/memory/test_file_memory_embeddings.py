# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
from unittest.mock import Mock
import pytest
import uuid

from tempfile import NamedTemporaryFile
from pathlib import Path

from pyrit.memory.file_memory import FileMemory
from pyrit.memory.memory_models import (
    ConversationMemoryEntry,
    ConversationMemoryEntryList,
)
from pyrit.models import ChatMessage, EmbeddingResponse
from pyrit.common.path import PYRIT_PATH

MEMORY_FILE_EXTENSION = ".json.memory"


def load_and_extract_embedding_from_json(embedding_file: Path) -> list[float]:
    raw_embedding_data = embedding_file.read_text(encoding="utf-8")
    embedding_json_data = json.loads(raw_embedding_data)
    embedding_data: list[float] = embedding_json_data["data"][0]["embedding"]
    return embedding_data


def mock_generate_text_embedding_effect(*args, **kwargs) -> EmbeddingResponse:
    arg = kwargs.get("text", None)

    if str.lower(arg) == "hello world":
        # This is the embeddings for "hello world"
        embedding_data_path = PYRIT_PATH.joinpath("..", "tests", "data", "embedding_1.json").resolve()
        return EmbeddingResponse.load_from_file(embedding_data_path)

    # This is garbage embeddings for default
    embedding_data_path = PYRIT_PATH.joinpath("..", "tests", "data", "embedding_1.json").resolve()
    embedding_default = EmbeddingResponse.load_from_file(embedding_data_path)
    embedding_default.data[0].embedding = [0.0 for _ in range(1536)]
    return embedding_default


@pytest.fixture
def text_embedding() -> Mock:
    text_embedding = Mock()

    text_embedding.generate_text_embedding.side_effect = mock_generate_text_embedding_effect
    return text_embedding


@pytest.fixture
def simple_conversation() -> ConversationMemoryEntryList:
    id = uuid.uuid4()
    return ConversationMemoryEntryList(
        conversations=[
            ConversationMemoryEntry(
                uuid=id,
                role="user",
                content="Hello World",
                conversation_id="1",
                timestamp_in_ns=0,
            ),
            ConversationMemoryEntry(
                role="assistant",
                content="Hello from Bot!",
                conversation_id="1",
                timestamp_in_ns=0,
            ),
            ConversationMemoryEntry(
                role="user",
                content="Hi, bot! How are you?",
                conversation_id="1",
                timestamp_in_ns=0,
            ),
        ]
    )


@pytest.fixture
def memory(simple_conversation: ConversationMemoryEntryList, text_embedding: Mock) -> FileMemory:
    with NamedTemporaryFile(suffix=".json.memory", delete=False) as tmp:
        m = FileMemory(filepath=tmp.name, embedding_model=text_embedding)
        for entry in simple_conversation.conversations:
            m.add_chat_message_to_memory(
                conversation=ChatMessage(role=entry.role, content=entry.content),
                conversation_id=entry.conversation_id,
            )
        return m


@pytest.fixture
def embedding_1_data() -> list[float]:
    """This is the embeddings for "hello world" """
    embedding_data_path = PYRIT_PATH.joinpath("..", "tests", "data", "embedding_1.json").resolve()
    return load_and_extract_embedding_from_json(embedding_data_path)


@pytest.fixture
def embedding_2_data() -> list[float]:
    """This is the embeddings for "hello world!" """
    embedding_data_path = PYRIT_PATH.joinpath("..", "tests", "data", "embedding_2.json").resolve()
    return load_and_extract_embedding_from_json(embedding_data_path)


def test_embedding_similary(memory: FileMemory, embedding_2_data: list[float]):
    # hello world with  is stored in memory, embedding_2_data is Hello world!
    similar_memories = memory.get_memory_by_embedding_similarity(memory_entry_emb=embedding_2_data)
    assert len(similar_memories) == 1
    assert similar_memories[0].score < 1 and similar_memories[0].score > 0.8


def test_embedding_similary_no_matches(memory: FileMemory):
    target_emb_data = [17.0] * 1536  # This is the dimension size of the ada embedding
    similar_memories = memory.get_memory_by_embedding_similarity(memory_entry_emb=target_emb_data)
    assert len(similar_memories) == 0


def test_embedding_similare_chat_messages(memory: FileMemory):
    similar_memories = memory.get_similar_chat_messages(chat_message_content="Hello!!")
    assert len(similar_memories) == 0


def test_embedding_similare_chat_messages_none(memory: FileMemory):
    similar_memories = memory.get_similar_chat_messages(chat_message_content="I have nothing in common")
    assert len(similar_memories) == 0
