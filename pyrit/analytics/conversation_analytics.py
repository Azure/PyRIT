# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.memory.memory_models import ConversationMessageWithSimilarity, EmbeddingMessageWithSimilarity
from pyrit.memory.memory_models import ConversationData, EmbeddingData


class ConversationAnalytics:
    """
    Handles analytics operations on conversation data, such as finding similar chat messages
    based on conversation history or embedding similarity.
    """

    def __init__(self, *, memory_interface: MemoryInterface):
        """
        Initializes the ConversationAnalytics with a memory interface for data access.

        Args:
            memory_interface (MemoryInterface): An instance of MemoryInterface for accessing conversation data.
        """
        self.memory_interface = memory_interface

    def get_similar_chat_messages_by_content(
        self, *, chat_message_content: str
    ) -> list[ConversationMessageWithSimilarity]:
        """
        Retrieves chat messages that are similar to the given content based on exact matches.

        Args:
            chat_message_content (str): The content of the chat message to find similar messages for.

        Returns:
            list[ConversationMessageWithSimilarity]: A list of ConversationMessageWithSimilarity objects representing
            the similar chat messages based on content.
        """
        all_memories = self.memory_interface.get_all_memory(ConversationData)
        similar_messages = []

        for memory in all_memories:
            if memory.content == chat_message_content:
                similar_messages.append(
                    ConversationMessageWithSimilarity(
                        score=1.0,
                        role=memory.role,
                        content=memory.content,
                        metric="exact_match",  # Exact match
                    )
                )

        return similar_messages

    def get_similar_chat_messages_by_embedding(
        self, *, chat_message_embedding: list[float], threshold: float = 0.8
    ) -> list[EmbeddingMessageWithSimilarity]:
        """
        Retrieves chat messages that are similar to the given embedding based on cosine similarity.

        Args:
            chat_message_embedding (List[float]): The embedding of the chat message to find similar messages for.
            threshold (float): The similarity threshold for considering messages as similar. Defaults to 0.8.

        Returns:
            List[ConversationMessageWithSimilarity]: A list of ConversationMessageWithSimilarity objects representing
            the similar chat messages based on embedding similarity.
        """
        all_memories = self.memory_interface.get_all_memory(EmbeddingData)
        similar_messages = []

        target_embedding = np.array(chat_message_embedding).reshape(1, -1)

        for memory in all_memories:
            if not hasattr(memory, "embedding") or memory.embedding is None:
                continue

            memory_embedding = np.array(memory.embedding).reshape((1, -1))
            similarity_score = cosine_similarity(target_embedding, memory_embedding)[0][0]

            if similarity_score >= threshold:
                similar_messages.append(
                    EmbeddingMessageWithSimilarity(
                        score=similarity_score, uuid=memory.uuid, metric="cosine_similarity"  # type: ignore
                    )
                )

        return similar_messages
