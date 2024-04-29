# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from unittest.mock import MagicMock

from pyrit.memory.memory_interface import MemoryInterface
from pyrit.analytics.conversation_analytics import ConversationAnalytics
from pyrit.memory.memory_models import EmbeddingData

from tests.mocks import get_sample_conversation_entries


@pytest.fixture
def mock_memory_interface():
    memory_interface = MagicMock(spec=MemoryInterface)
    return memory_interface


@pytest.fixture
def sample_conversations_entries():
    return get_sample_conversation_entries()


def test_get_similar_chat_messages_by_content(mock_memory_interface, sample_conversations_entries):

    sample_conversations_entries[0].converted_value = "Hello, how are you?"
    sample_conversations_entries[2].converted_value = "Hello, how are you?"

    mock_memory_interface.get_all_prompt_pieces.return_value = sample_conversations_entries

    analytics = ConversationAnalytics(memory_interface=mock_memory_interface)
    similar_messages = analytics.get_prompt_entries_with_same_converted_content(
        chat_message_content="Hello, how are you?"
    )

    # Expect one exact match
    assert len(similar_messages) == 2
    for message in similar_messages:
        assert message.content == "Hello, how are you?"
        assert message.score == 1.0
        assert message.metric == "exact_match"


def test_get_similar_chat_messages_by_embedding(mock_memory_interface, sample_conversations_entries):
    sample_conversations_entries[0].converted_value = "Similar message"
    sample_conversations_entries[1].converted_value = "Different message"

    # Mock EmbeddingData entries linked to the ConversationData entries
    target_embedding = [0.1, 0.2, 0.3]
    similar_embedding = [0.1, 0.2, 0.31]  # Slightly different, but should be similar
    different_embedding = [0.9, 0.8, 0.7]

    mock_embeddings = [
        EmbeddingData(id=sample_conversations_entries[0].id, embedding=similar_embedding, embedding_type_name="model1"),
        EmbeddingData(
            id=sample_conversations_entries[1].id, embedding=different_embedding, embedding_type_name="model2"
        ),
    ]

    # Mock the get_all_prompt_pieces method to return the mock EmbeddingData entries
    mock_memory_interface.get_all_embeddings.return_value = mock_embeddings
    mock_memory_interface.get_all_prompt_pieces.return_value = sample_conversations_entries

    analytics = ConversationAnalytics(memory_interface=mock_memory_interface)
    similar_messages = analytics.get_similar_chat_messages_by_embedding(
        chat_message_embedding=target_embedding, threshold=0.99
    )

    # Expect one similar message based on embedding
    assert len(similar_messages) == 1
    assert similar_messages[0].score >= 0.99
    assert similar_messages[0].metric == "cosine_similarity"
