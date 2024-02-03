# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from hashlib import sha256

from uuid import uuid4
from pyrit.memory.memory_embedding import MemoryEmbedding

from pyrit.memory.memory_models import (
    ConversationMemoryEntry,
    ConversationMessageWithSimilarity,
)
from pyrit.models import ChatMessage


class MemoryInterface(abc.ABC):
    """
    Represents a conversation memory that stores chat messages. This class must be overwritten with a
    specific implementation to store the memory objects (e.g. database, files, etc.)

    Args:
        embedding_model (EmbeddingSupport): If set, this includes embeddings in the memory entries
        which are extremely useful for comparing chat messages and similarities, but also includes overhead
    """

    memory_embedding: MemoryEmbedding = None

    @abc.abstractmethod
    def get_all_memory(self) -> list[ConversationMemoryEntry]:
        """
        Loads all ConversationMemoryEntry from the memory storage handler.
        """

    @abc.abstractmethod
    def save_conversation_memory_entries(self, entries: list[ConversationMemoryEntry]) -> None:
        """
        Saves the ConversationMemoryEntry to the memory storage handler.
        """

    @abc.abstractmethod
    def get_memories_with_conversation_id(self, *, conversation_id: str) -> list[ConversationMemoryEntry]:
        """
        Retrieves a list of ConversationMemoryEntry objects that have the specified conversation ID.

        Args:
            conversation_id (str): The conversation ID to match.

        Returns:
            list[ConversationMemoryEntry]: A list of chat memory entries with the specified conversation ID.
        """

    @abc.abstractmethod
    def get_memories_with_normalizer_id(self, *, normalizer_id: str) -> list[ConversationMemoryEntry]:
        """
        Retrieves a list of ConversationMemoryEntry objects that have the specified normalizer ID.

        Args:
            normalizer_id (str): The normalizer ID to match.

        Returns:
            list[ConversationMemoryEntry]: A list of chat memory entries with the specified normalizer ID.
        """

    @abc.abstractmethod
    def get_memory_by_exact_match(self, *, memory_entry_content: str) -> list[ConversationMessageWithSimilarity | None]:
        """
        Retrieves chat memory entries that exactly match the given content.

        Args:
            memory_entry_content (str): The content to match.

        Returns:
            list[ConversationMemoryEntry | None]: A list of chat memory entries that match the content, or None if no
            match is found.
        """

    @abc.abstractmethod
    def get_memory_by_embedding_similarity(
        self, *, memory_entry_emb: list[float], threshold: float = 0.8
    ) -> list[ConversationMessageWithSimilarity | None]:
        """
        Retrieves chat memory entries that have an embedding similarity with the given content.

        Args:
            memory_entry_emb (list[float]): The content to match.
            threshold (float): The threshold to use for the similarity. Defaults to 0.8.

        Returns:
            list[ConversationMessageWithSimilarity | None]: A list of chat memory entries with similarity information,
            or None if no match is found.
        """

    def get_chat_messages(self) -> list[ChatMessage]:
        """
        Retrieves all chat messages from the memory.

        Returns:
            list[ChatMessage]: The list of chat messages.
        """

        memories = self.get_all_memory()

        sorted_memory_entries = sorted(
            memories,
            key=lambda x: x.timestamp_in_ns,
            reverse=True,
        )
        return [ChatMessage(role=me.role, content=me.content) for me in sorted_memory_entries]

    def add_chat_message_to_memory(
        self,
        *,
        conversation: ChatMessage,
        conversation_id: str,
        normalizer_id: str = None,
        labels: list[str] = None,
    ):
        """
        Appends a conversation to the memory.

        Args:
            conversation (ChatMessage): The chat message to be added.
            conversation_id (str): The conversation ID.
            normalizer_id (str): The normalizer ID,
            labels (list[str]): A list of labels to be added to the memory entry.
        """
        chats = self._create_chat_message_memory_entry(
            conversation=conversation, conversation_id=conversation_id, normalizer_id=normalizer_id, labels=labels
        )

        self.save_conversation_memory_entries([chats])

    def add_chat_messages_to_memory(
        self,
        *,
        conversations: list[ChatMessage],
        conversation_id: str,
        normalizer_id: str = None,
        labels: list[str] = None,
    ):
        """
        Appends a list of conversations to the memory.

        Args:
            conversations (ChatMessage): The chat message to be added.
            conversation_id (str): The conversation ID.
            normalizer_id (str): The normalizer ID
            labels (list[str]): A list of labels to be added to the memory entry.
        """
        chat_list = []

        for conversation in conversations:
            chat_list.append(
                self._create_chat_message_memory_entry(
                    conversation=conversation,
                    conversation_id=conversation_id,
                    normalizer_id=normalizer_id,
                    labels=labels,
                )
            )
        self.save_conversation_memory_entries(chat_list)

    def get_chat_messages_with_conversation_id(self, *, conversation_id: str) -> list[ChatMessage]:
        """
        Returns the memory for a given session ID.

        Args:
            conversation_id (str): The conversation ID.

        Returns:
            list[ChatMessage]: The list of chat messages.
        """
        memory_entries = self.get_memories_with_conversation_id(conversation_id=conversation_id)
        return [ChatMessage(role=me.role, content=me.content) for me in memory_entries]

    def _create_chat_message_memory_entry(
        self,
        *,
        conversation: ChatMessage,
        conversation_id: str,
        normalizer_id: str = None,
        labels: list[str] = None,
    ) -> ConversationMemoryEntry:
        uuid = uuid4()

        new_chat_memory = ConversationMemoryEntry(
            role=conversation.role,
            content=conversation.content,
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
            uuid=uuid,
            labels=labels if labels else [],
            sha256=sha256(conversation.content.encode()).hexdigest(),
        )
        if self.memory_embedding:
            new_chat_memory.embedding_memory_data = self.memory_embedding.generate_embedding_memory_data(
                chat_memory=new_chat_memory
            )

        return new_chat_memory

    def get_similar_chat_messages(self, *, chat_message_content: str) -> list[ConversationMessageWithSimilarity]:
        """
        Retrieves a list of chat messages that are similar to the given chat message content.

        Args:
            chat_message_content (str): The content of the chat message to find similar messages for.

        Returns:
            list[ConversationMessageWithSimilarity]: A list of ConversationMessageWithSimilarity objects representing
            the similar chat messages.
        """
        matches: list[ConversationMessageWithSimilarity] = []
        exact_matches = self.get_memory_by_exact_match(memory_entry_content=chat_message_content)
        matches.extend(exact_matches)
        if self.memory_embedding:
            text_emb_response = self.memory_embedding.embedding_model.generate_text_embedding(text=chat_message_content)
            text_emd = text_emb_response.data[0].embedding
            similar_matches = self.get_memory_by_embedding_similarity(memory_entry_emb=text_emd)
            matches.extend(similar_matches)
        return matches
