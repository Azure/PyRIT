# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from hashlib import sha256
from typing import Optional
from pathlib import Path

from uuid import uuid4

from pyrit.memory.memory_models import Base, PromptMemoryEntry
from pyrit.memory.memory_embedding import MemoryEmbedding
from pyrit.memory.memory_exporter import MemoryExporter
from pyrit.models import ChatMessage
from pyrit.common.path import RESULTS_PATH


class MemoryInterface(abc.ABC):
    """
    Represents a conversation memory that stores chat messages. This class must be overwritten with a
    specific implementation to store the memory objects (e.g. relational database, NoSQL database, etc.)

    Args:
        embedding_model (EmbeddingSupport): If set, this includes embeddings in the memory entries
        which are extremely useful for comparing chat messages and similarities, but also includes overhead
    """

    memory_embedding: MemoryEmbedding = None

    def __init__(self, embedding_model=None):
        self.memory_embedding = embedding_model
        # Initialize the MemoryExporter instance
        self.exporter = MemoryExporter()

    @abc.abstractmethod
    def get_all_memory(self, model: Base) -> list[PromptMemoryEntry]:  # type: ignore
        """
        Loads all ConversationData from the memory storage handler.
        """

    @abc.abstractmethod
    def get_memories_with_conversation_id(self, *, conversation_id: str) -> list[PromptMemoryEntry]:
        """
        Retrieves a list of ConversationData objects that have the specified conversation ID.

        Args:
            conversation_id (str): The conversation ID to match.

        Returns:
            list[ConversationData]: A list of chat memory entries with the specified conversation ID.
        """

    @abc.abstractmethod
    def get_memories_with_normalizer_id(self, *, normalizer_id: str) -> list[PromptMemoryEntry]:
        """
        Retrieves a list of ConversationData objects that have the specified normalizer ID.

        Args:
            normalizer_id (str): The normalizer ID to match.

        Returns:
            list[ConversationData]: A list of chat memory entries with the specified normalizer ID.
        """

    @abc.abstractmethod
    def insert_entries(self, *, entries: list[Base]) -> None:  # type: ignore
        """
        Inserts a list of entries into the memory storage.

        Args:
            entries (list[Base]): The list of database model instances to be inserted.
        """

    @abc.abstractmethod
    def get_all_table_models(self) -> list[Base]:  # type: ignore
        """
        Returns a list of all table models from the database.

        Returns:
            list[Base]: A list of SQLAlchemy models.
        """

    @abc.abstractmethod
    def query_entries(self, model, *, conditions: Optional = None) -> list[Base]:  # type: ignore
        """
        Fetches data from the specified table model with optional conditions.

        Args:
            model: The SQLAlchemy model class corresponding to the table you want to query.
            conditions: SQLAlchemy filter conditions (optional).

        Returns:
            List of model instances representing the rows fetched from the table.
        """

    @abc.abstractmethod
    def dispose_engine(self):
        """
        Dispose the engine and clean up resources.
        """

    def add_chat_message_to_memory(
        self,
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
            embedding_entry = self.memory_embedding.generate_embedding_memory_data(chat_memory=chat_entry)
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
                conversation=conversation, conversation_id=conversation_id, normalizer_id=normalizer_id, labels=labels
            )
            entries_to_persist.append(chat_entry)
            if self.memory_embedding:
                embedding_entry = self.memory_embedding.generate_embedding_memory_data(chat_memory=chat_entry)
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
        return [ChatMessage(role=me.role, content=me.content) for me in memory_entries]  # type: ignore

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
        new_chat_memory = PromptMemoryEntry(
            role=conversation.role,
            content=conversation.content,
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
            uuid=uuid,
            labels=labels if labels else [],
            sha256=sha256(conversation.content.encode()).hexdigest(),
        )

        return new_chat_memory

    def export_all_tables(self, *, export_type: str = "json"):
        """
        Exports all table data using the specified exporter.

        Iterates over all tables, retrieves their data, and exports each to a file named after the table.

        Args:
            export_type (str): The format to export the data in (defaults to "json").
        """
        table_models = self.get_all_table_models()

        for model in table_models:
            data = self.query_entries(model)
            table_name = model.__tablename__
            file_extension = f".{export_type}"
            file_path = RESULTS_PATH / f"{table_name}{file_extension}"
            self.exporter.export_data(data, file_path=file_path, export_type=export_type)

    def export_conversation_by_id(self, *, conversation_id: str, file_path: Path = None, export_type: str = "json"):
        """
        Exports conversation data with the given conversation ID to a specified file.

        Args:
            conversation_id (str): The ID of the conversation to export.
            file_path (str): The path to the file where the data will be exported.
            If not provided, a default path using RESULTS_PATH will be constructed.
            export_type (str): The format of the export. Defaults to "json".
        """
        data = self.get_memories_with_conversation_id(conversation_id=conversation_id)

        # If file_path is not provided, construct a default using the exporter's results_path
        if not file_path:
            file_name = f"{conversation_id}.{export_type}"
            file_path = RESULTS_PATH / file_name

        self.exporter.export_data(data, file_path=file_path, export_type=export_type)
