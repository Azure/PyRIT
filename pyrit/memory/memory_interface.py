# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from pathlib import Path

from pyrit.memory.memory_embedding import default_memory_embedding_factory
from pyrit.memory.memory_models import PromptMemoryEntry, EmbeddingData
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

    def enable_embedding(self, embedding_model=None):
        self.memory_embedding = default_memory_embedding_factory(embedding_model=embedding_model)

    def disable_embedding(self):
        self.memory_embedding = None

    @abc.abstractmethod
    def get_all_prompt_entries(self) -> list[PromptMemoryEntry]:
        """
        Loads all ConversationData from the memory storage handler.
        """

    @abc.abstractmethod
    def get_all_embeddings(self) -> list[EmbeddingData]:
        """
        Loads all EmbeddingData from the memory storage handler.
        """

    @abc.abstractmethod
    def get_prompt_entries_with_conversation_id(self, *, conversation_id: str) -> list[PromptMemoryEntry]:
        """
        Retrieves a list of ConversationData objects that have the specified conversation ID.

        Args:
            conversation_id (str): The conversation ID to match.

        Returns:
            list[ConversationData]: A list of chat memory entries with the specified conversation ID.
        """

    @abc.abstractmethod
    def get_prompt_entries_with_normalizer_id(self, *, normalizer_id: str) -> list[PromptMemoryEntry]:
        """
        Retrieves a list of ConversationData objects that have the specified normalizer ID.

        Args:
            normalizer_id (str): The normalizer ID to match.

        Returns:
            list[ConversationData]: A list of chat memory entries with the specified normalizer ID.
        """

    @abc.abstractmethod
    def insert_prompt_entries(self, *, entries: list[PromptMemoryEntry]) -> None:
        """
        Inserts a list of entries into the memory storage.

        If necessary, generates embedding data for applicable entries

        Args:
            entries (list[Base]): The list of database model instances to be inserted.
        """

    @abc.abstractmethod
    def dispose_engine(self):
        """
        Dispose the engine and clean up resources.
        """

    def get_chat_messages_with_conversation_id(self, *, conversation_id: str) -> list[ChatMessage]:
        """
        Returns the memory for a given conversation_id.

        Args:
            conversation_id (str): The conversation ID.

        Returns:
            list[ChatMessage]: The list of chat messages.
        """
        memory_entries = self.get_prompt_entries_with_conversation_id(conversation_id=conversation_id)
        return [ChatMessage(role=me.role, content=me.converted_prompt_text) for me in memory_entries]  # type: ignore

    def add_chat_message_to_memory(
        self,
        conversation: ChatMessage,
        conversation_id: str,
        normalizer_id: str = None,
        labels: dict[str, str] = {},
    ):
        """
        Deprecated. Will be refactored and removed soon. It currently works incorrectly.
        but is included so functionality is maintained.

        Adds a single chat conversation entry to the ConversationStore table.
        If embddings are set, add corresponding embedding entry to the EmbeddingStore table.

        Args:
            conversation (ChatMessage): The chat message to be added.
            conversation_id (str): The conversation ID.
            normalizer_id (str): The normalizer ID,
            labels (list[str]): A list of labels to be added to the memory entry.
        """

        self.add_chat_messages_to_memory(
            conversations=[conversation], conversation_id=conversation_id, normalizer_id=normalizer_id, labels=labels
        )

    def add_chat_messages_to_memory(
        self,
        *,
        conversations: list[ChatMessage],
        conversation_id: str,
        normalizer_id: str = None,
        labels: dict[str, str] = {},
    ):
        """
        Deprecated. Will be refactored and removed soon. It currently works incorrectly.
        but is included so functionality is maintained.

        Adds multiple chat conversation entries to the ConversationStore table.
        If embddings are set, add corresponding embedding entries to the EmbeddingStore table.

        Args:
            conversations (ChatMessage): The chat message to be added.
            conversation_id (str): The conversation ID.
            normalizer_id (str): The normalizer ID
            labels (list[str]): A list of labels to be added to the memory entry.
        """
        entries_to_add = []

        for conversation in conversations:
            entry = PromptMemoryEntry(
                role=conversation.role,
                conversation_id=conversation_id,
                original_prompt_text=conversation.content,
                converted_prompt_text=conversation.content,
                labels=labels,
            )

            entry.labels["normalizer_id"] = normalizer_id

            entries_to_add.append(entry)

        self.insert_prompt_entries(entries=entries_to_add)

    def export_conversation_by_id(self, *, conversation_id: str, file_path: Path = None, export_type: str = "json"):
        """
        Exports conversation data with the given conversation ID to a specified file.

        Args:
            conversation_id (str): The ID of the conversation to export.
            file_path (str): The path to the file where the data will be exported.
            If not provided, a default path using RESULTS_PATH will be constructed.
            export_type (str): The format of the export. Defaults to "json".
        """
        data = self.get_prompt_entries_with_conversation_id(conversation_id=conversation_id)

        # If file_path is not provided, construct a default using the exporter's results_path
        if not file_path:
            file_name = f"{conversation_id}.{export_type}"
            file_path = RESULTS_PATH / file_name

        self.exporter.export_data(data, file_path=file_path, export_type=export_type)
