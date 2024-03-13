# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from hashlib import sha256

from uuid import uuid4

from pyrit.memory.memory_models import Base, ConversationData
from pyrit.memory.memory_embedding import MemoryEmbedding
from pyrit.models import ChatMessage


class MemoryInterface(abc.ABC):
    """
    Represents a conversation memory that stores chat messages. This class must be overwritten with a
    specific implementation to store the memory objects (e.g. relational database, NoSQL database, etc.)

    Args:
        embedding_model (EmbeddingSupport): If set, this includes embeddings in the memory entries
        which are extremely useful for comparing chat messages and similarities, but also includes overhead
    """

    memory_embedding: MemoryEmbedding = None

    @abc.abstractmethod
    def get_all_memory(self) -> list[ConversationData]:
        """
        Loads all ConversationData from the memory storage handler.
        """

    @abc.abstractmethod
    def get_memories_with_conversation_id(self, *, conversation_id: str) -> list[ConversationData]:
        """
        Retrieves a list of ConversationData objects that have the specified conversation ID.

        Args:
            conversation_id (str): The conversation ID to match.

        Returns:
            list[ConversationData]: A list of chat memory entries with the specified conversation ID.
        """

    @abc.abstractmethod
    def get_memories_with_normalizer_id(self, *, normalizer_id: str) -> list[ConversationData]:
        """
        Retrieves a list of ConversationData objects that have the specified normalizer ID.

        Args:
            normalizer_id (str): The normalizer ID to match.

        Returns:
            list[ConversationData]: A list of chat memory entries with the specified normalizer ID.
        """
    
    @abc.abstractmethod
    def insert_entries(self, *, entries: list[Base]) -> None: # type: ignore
        """
        Inserts a list of entries into the memory storage.
        
        Args:
            entries (list[Base]): The list of database model instances to be inserted.
        """

    def add_chat_message_to_memory(
        self,
        *,
        conversation: ChatMessage,
        conversation_id: str,
        normalizer_id: str = None,
        labels: list[str] = None,
    ):
        """
        Adds a single chat conversation entry to the ConversationStore table.
        If embddings are set, add corresponding embedding entry to the EmbeddingStore table. 

        Args:
            conversation (ChatMessage): The chat message to be added.
            conversation_id (str): The conversation ID.
            normalizer_id (str): The normalizer ID,
            labels (list[str]): A list of labels to be added to the memory entry.
        """
        entries_to_persist = []
        chat_entry = self._create_chat_message_memory_entry(
            conversation=conversation, conversation_id=conversation_id, normalizer_id=normalizer_id, labels=labels
        )
        entries_to_persist.append(chat_entry)
        if self.memory_embedding:
            embedding_entry = self.memory_embedding.generate_embedding_memory_data(
                chat_memory=chat_entry
            )
            entries_to_persist.append(embedding_entry)
        
        self.insert_entries(entries=entries_to_persist)

    def add_chat_messages_to_memory(
        self,
        *,
        conversations: list[ChatMessage],
        conversation_id: str,
        normalizer_id: str = None,
        labels: list[str] = None,
    ):
        """
        Adds multiple chat conversation entries to the ConversationStore table.
        If embddings are set, add corresponding embedding entries to the EmbeddingStore table.

        Args:
            conversations (ChatMessage): The chat message to be added.
            conversation_id (str): The conversation ID.
            normalizer_id (str): The normalizer ID
            labels (list[str]): A list of labels to be added to the memory entry.
        """
        entries_to_persist = []

        for conversation in conversations:
            chat_entry = self._create_chat_message_memory_entry(
            conversation=conversation, conversation_id=conversation_id, normalizer_id=normalizer_id, labels=labels)
            entries_to_persist.append(chat_entry)
            if self.memory_embedding:
                embedding_entry = self.memory_embedding.generate_embedding_memory_data(
                chat_memory=chat_entry
            )
                entries_to_persist.append(embedding_entry)
            
        self.insert_entries(entries=entries_to_persist)

    def get_chat_messages_with_conversation_id(self, *, conversation_id: str) -> list[ChatMessage]:
        """
        Returns the memory for a given conversation_id.

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
    ):
        """
        Creates a new `ConversationData` instance representing a chat message entry.

        Args:
            conversation (ChatMessage): The chat message to be stored.
            conversation_id (str): Conversation ID.
            normalizer_id (str): Normalizer ID.
            labels (list[str]): Labels associated with the conversation.

        Returns:
            ConversationData: A new instance ready to be persisted in the memory storage.
        """
        uuid = uuid4()
        new_chat_memory = ConversationData(
            role=conversation.role,
            content=conversation.content,
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
            uuid=uuid,
            labels=labels if labels else [],
            sha256=sha256(conversation.content.encode()).hexdigest(),
        )

        return new_chat_memory
