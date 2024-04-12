# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from pathlib import Path

from pyrit.memory.memory_models import PromptDataType, PromptMemoryEntry, EmbeddingData
from pyrit.models import PromptRequestResponse, PromptRequestPiece, PromptResponseError

from pyrit.memory.memory_embedding import default_memory_embedding_factory
from pyrit.memory.memory_embedding import MemoryEmbedding
from pyrit.memory.memory_exporter import MemoryExporter
from pyrit.models.models import ChatMessage
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
    def get_prompt_entries_by_orchestrator(self, *, orchestrator: 'Orchestrator') -> list[PromptMemoryEntry]:
        """
        Retrieves a list of PromptMemoryEntries based on a specific orchestrator object.

        Args:
            orchestrator (Orchestrator): The orchestrator object to match

        Returns:
            list[PromptMemoryEntry]: A list of PromptMemoryEntry with the specified orchestrator.
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

    def add_response_entries_to_memory(
            self,
            *,
            request: PromptRequestPiece,
            response_text_pieces: list[str],
            response_type: PromptDataType = "text",
            metadata: str = None,
            error: PromptResponseError = "none"
    ) -> PromptRequestResponse:

        request_pieces = []
        memory_entries = []

        for resp_text in response_text_pieces:
            entry = PromptRequestPiece(
                role="assistant",
                original_prompt_text=resp_text,
                converted_prompt_text=resp_text,
                conversation_id=request.conversation_id,
                sequence=request.sequence + 1,
                labels=request.labels,
                prompt_target=request.prompt_target,
                orchestrator=request.orchestrator,
                original_prompt_data_type=response_type,
                converted_prompt_data_type=response_type,
                metadata=metadata,
                response_error=error
            )
            memory_entries.append(PromptMemoryEntry(entry=entry))
            request_pieces.append(entry)

        self.insert_prompt_entries(memory_entries)
        return PromptRequestResponse(request_pieces=request_pieces)


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
