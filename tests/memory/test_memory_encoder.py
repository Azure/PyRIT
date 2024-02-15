# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Any, Coroutine

import pytest

from pyrit.interfaces import EmbeddingSupport
from pyrit.memory import MemoryEmbedding, ConversationMemoryEntry
from pyrit.models import EmbeddingData, EmbeddingResponse, EmbeddingUsageInformation


DEFAULT_EMBEDDING_DATA = EmbeddingData(embedding=[0.0], index=0, object="mock_object")


class MockEmbeddingGenerator(EmbeddingSupport):
    """Mock Memory Encoder for testing"""

    def generate_text_embedding(self, text: str, **kwargs) -> EmbeddingResponse:
        return EmbeddingResponse(
            model="mock_model",
            object="mock_object",
            usage=EmbeddingUsageInformation(prompt_tokens=0, total_tokens=0),
            data=[DEFAULT_EMBEDDING_DATA],
        )

    def generate_text_embedding_async(self, text: str, **kwargs) -> Coroutine[Any, Any, EmbeddingResponse]:
        raise NotImplementedError()


class MockChatGenerator(EmbeddingSupport):
    def __init__(self):
        pass

    def generate_text_embedding(self, text: str, **kwargs) -> EmbeddingResponse:
        return super().generate_text_embedding(text, **kwargs)

    def generate_text_embedding_async(self, text: str, **kwargs) -> Coroutine[Any, Any, EmbeddingResponse]:
        return super().generate_text_embedding_async(text, **kwargs)


def test_memory_encoder():
    memory_encoder = MemoryEmbedding(embedding_model=MockEmbeddingGenerator())
    assert memory_encoder


@pytest.fixture
def memory_encoder_w_mock_embedding_generator():
    return MemoryEmbedding(embedding_model=MockEmbeddingGenerator())


def test_memory_encoding_chat_message(
    memory_encoder_w_mock_embedding_generator: MemoryEmbedding,
):
    chat_memory = ConversationMemoryEntry(
        content="hello world!",
        role="user",
        conversation_id="my_session",
    )
    metadata = memory_encoder_w_mock_embedding_generator.generate_embedding_memory_data(chat_memory=chat_memory)
    assert metadata.uuid == chat_memory.uuid
    assert metadata.embedding == DEFAULT_EMBEDDING_DATA.embedding
    assert metadata.embedding_type_name == "MockEmbeddingGenerator"
