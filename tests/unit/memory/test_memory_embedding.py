# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Any, Coroutine, MutableSequence, Sequence

import pytest
from unit.mocks import get_sample_conversation_entries

from pyrit.memory import MemoryEmbedding, PromptMemoryEntry
from pyrit.memory.memory_embedding import default_memory_embedding_factory
from pyrit.models import (
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingSupport,
    EmbeddingUsageInformation,
)

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


@pytest.fixture
def sample_conversation_entries() -> Sequence[PromptMemoryEntry]:
    return get_sample_conversation_entries()


def test_memory_encoder():
    memory_encoder = MemoryEmbedding(embedding_model=MockEmbeddingGenerator())
    assert memory_encoder


@pytest.fixture
def memory_encoder_w_mock_embedding_generator():
    return MemoryEmbedding(embedding_model=MockEmbeddingGenerator())


def test_memory_encoding_chat_message(
    memory_encoder_w_mock_embedding_generator: MemoryEmbedding,
    sample_conversation_entries: MutableSequence[PromptMemoryEntry],
):
    chat_memory = sample_conversation_entries[0]

    metadata = memory_encoder_w_mock_embedding_generator.generate_embedding_memory_data(
        prompt_request_piece=chat_memory.get_prompt_request_piece()
    )
    assert metadata.id == chat_memory.id
    assert metadata.embedding == DEFAULT_EMBEDDING_DATA.embedding
    assert metadata.embedding_type_name == "MockEmbeddingGenerator"


def test_default_memory_embedding_factory_with_embedding_model():
    embedding_model = MockEmbeddingGenerator()
    memory_embedding = default_memory_embedding_factory(embedding_model=embedding_model)
    assert isinstance(memory_embedding, MemoryEmbedding)
    assert memory_embedding.embedding_model == embedding_model


def test_default_memory_embedding_factory_with_azure_environment_variables(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_KEY", "mock_key")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", "mock_endpoint")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "mock_deployment")

    memory_embedding = default_memory_embedding_factory()
    assert isinstance(memory_embedding, MemoryEmbedding)


def test_default_memory_embedding_factory_without_embedding_model_and_environment_variables(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_EMBEDDING_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", raising=False)

    with pytest.raises(ValueError):
        default_memory_embedding_factory()
