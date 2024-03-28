# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import uuid
from unittest.mock import MagicMock

from pyrit.memory.memory_interface import MemoryInterface
from pyrit.analytics.conversation_analytics import ConversationAnalytics
from pyrit.memory.memory_models import PromptMemoryEntry, EmbeddingData


@pytest.fixture
def mock_memory_interface():
    memory_interface = MagicMock(spec=MemoryInterface)
    return memory_interface


def test_get_similar_chat_messages_by_content(mock_memory_interface):
    # Mock data returned by the memory interface
    mock_data = [
        PromptMemoryEntry(original_prompt_text="h", converted_prompt_text="Hello, how are you?", role="user"),
        PromptMemoryEntry(original_prompt_text="h", converted_prompt_text="I'm fine, thank you!", role="assistant"),
        PromptMemoryEntry(original_prompt_text="h", converted_prompt_text="Hello, how are you?", role="assistant"),  # Exact match
    ]
    mock_memory_interface.get_all_prompt_entries.return_value = mock_data

    analytics = ConversationAnalytics(memory_interface=mock_memory_interface)
    similar_messages = analytics.get_prompt_entries_with_same_converted_content(chat_message_content="Hello, how are you?")

    # Expect one exact match
    assert len(similar_messages) == 2
    for message in similar_messages:
        assert message.content == "Hello, how are you?"
        assert message.score == 1.0
        assert message.metric == "exact_match"


def test_get_similar_chat_messages_by_embedding(mock_memory_interface):
    # Mock ConversationData entries
    conversation_entries = [
        PromptMemoryEntry(original_prompt_text="h", role="user", converted_prompt_text="Similar message"),
        PromptMemoryEntry(original_prompt_text="h", role="assistant", converted_prompt_text="Different message"),
    ]

    # Mock EmbeddingData entries linked to the ConversationData entries
    target_embedding = [0.1, 0.2, 0.3]
    similar_embedding = [0.1, 0.2, 0.31]  # Slightly different, but should be similar
    different_embedding = [0.9, 0.8, 0.7]

    mock_data = [
        EmbeddingData(id=conversation_entries[0].id, embedding=similar_embedding, embedding_type_name="model1"),
        EmbeddingData(id=conversation_entries[1].id, embedding=different_embedding, embedding_type_name="model2"),
    ]

    # Mock the get_all_prompt_entries method to return the mock EmbeddingData entries
    mock_memory_interface.get_all_prompt_entries.side_effect = lambda model: (
        mock_data if model == EmbeddingData else conversation_entries
    )

    analytics = ConversationAnalytics(memory_interface=mock_memory_interface)
    similar_messages = analytics.get_similar_chat_messages_by_embedding(
        chat_message_embedding=target_embedding, threshold=0.99
    )

    # Expect one similar message based on embedding
    assert len(similar_messages) == 1
    assert similar_messages[0].score >= 0.99
    assert similar_messages[0].metric == "cosine_similarity"
