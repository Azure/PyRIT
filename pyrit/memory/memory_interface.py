# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from pathlib import Path

from typing import Optional
from uuid import uuid4

from pyrit.common.path import RESULTS_PATH
from pyrit.models import (
    ChatMessage,
    PromptRequestResponse,
    Score,
    PromptRequestPiece,
    group_conversation_request_pieces_by_sequence,
)

from pyrit.memory.memory_models import EmbeddingData
from pyrit.memory.memory_embedding import default_memory_embedding_factory, MemoryEmbedding
from pyrit.memory.memory_exporter import MemoryExporter


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
    def get_all_prompt_pieces(self) -> list[PromptRequestPiece]:
        """
        Loads all ConversationData from the memory storage handler.
        """

    @abc.abstractmethod
    def get_all_embeddings(self) -> list[EmbeddingData]:
        """
        Loads all EmbeddingData from the memory storage handler.
        """

    @abc.abstractmethod
    def _get_prompt_pieces_with_conversation_id(self, *, conversation_id: str) -> list[PromptRequestPiece]:
        """
        Retrieves a list of PromptRequestPiece objects that have the specified conversation ID.

        Args:
            conversation_id (str): The conversation ID to match.

        Returns:
            list[PromptRequestPiece]: A list of chat memory entries with the specified conversation ID.
        """

    @abc.abstractmethod
    def _get_prompt_pieces_by_orchestrator(self, *, orchestrator_id: int) -> list[PromptRequestPiece]:
        """
        Retrieves a list of PromptRequestPiece objects that have the specified orchestrator ID.

        Args:
            orchestrator_id (str): The id of the orchestrator.
                Can be retrieved by calling orchestrator.get_identifier()["id"]

        Returns:
            list[PromptRequestPiece]: A list of PromptMemoryEntry objects matching the specified orchestrator ID.
        """

    @abc.abstractmethod
    def add_request_pieces_to_memory(self, *, request_pieces: list[PromptRequestPiece]) -> None:
        """
        Inserts a list of prompt request pieces into the memory storage.
        """

    @abc.abstractmethod
    def _add_embeddings_to_memory(self, *, embedding_data: list[EmbeddingData]) -> None:
        """
        Inserts embedding data into memory storage
        """

    @abc.abstractmethod
    def add_scores_to_memory(self, *, scores: list[Score]) -> None:
        """
        Inserts a list of scores into the memory storage.
        """

    @abc.abstractmethod
    def get_scores_by_prompt_ids(self, *, prompt_request_response_ids: list[str]) -> list[Score]:
        """
        Gets a list of scores based on prompt_request_response_ids.
        """

    def get_conversation(self, *, conversation_id: str) -> list[PromptRequestResponse]:
        """
        Retrieves a list of PromptRequestResponse objects that have the specified conversation ID.

        Args:
            conversation_id (str): The conversation ID to match.

        Returns:
            list[PromptRequestResponse]: A list of chat memory entries with the specified conversation ID.
        """
        request_pieces = self._get_prompt_pieces_with_conversation_id(conversation_id=conversation_id)
        return group_conversation_request_pieces_by_sequence(request_pieces=request_pieces)

    @abc.abstractmethod
    def get_prompt_request_pieces_by_id(self, *, prompt_ids: list[str]) -> list[PromptRequestPiece]:
        """
        Retrieves a list of PromptRequestPiece objects that have the specified prompt ids.

        Args:
            prompt_ids (list[int]): The prompt IDs to match.

        Returns:
            list[PromptRequestPiece]: A list of PromptRequestPiece with the specified conversation ID.
        """

    def get_prompt_request_piece_by_orchestrator_id(self, *, orchestrator_id: int) -> list[PromptRequestPiece]:
        """
        Retrieves a list of PromptRequestPiece objects that have the specified orchestrator ID.

        Args:
            orchestrator_id (int): The orchestrator ID to match.

        Returns:
            list[PromptRequestPiece]: A list of PromptRequestPiece with the specified conversation ID.
        """

        prompt_pieces = self._get_prompt_pieces_by_orchestrator(orchestrator_id=orchestrator_id)
        return sorted(prompt_pieces, key=lambda x: (x.conversation_id, x.timestamp))

    def duplicate_conversation_for_new_orchestrator(
        self,
        *,
        new_orchestrator_id: str,
        conversation_id: str,
        new_conversation_id: Optional[str] = None,
    ) -> None:
        """
        Duplicates a conversation from one orchestrator to another.

        This can be useful when an attack strategy requires branching out from a particular point in the conversation.
        One cannot continue both branches with the same orchestrator and conversation IDs since that would corrupt
        the memory. Instead, one needs to duplicate the conversation and continue with the new orchestrator ID.

        Args:
            new_orchestrator_id (str): The new orchestrator ID to assign to the duplicated conversations.
            conversation_id (str): The conversation ID with existing conversations.
            new_conversation_id (str): The new conversation ID to assign to the duplicated conversations.
                If no new_conversation_id is provided, a new one will be generated.
        """
        new_conversation_id = new_conversation_id or str(uuid4())
        if conversation_id == new_conversation_id:
            raise ValueError("The new conversation ID must be different from the existing conversation ID.")
        prompt_pieces = self._get_prompt_pieces_with_conversation_id(conversation_id=conversation_id)
        for piece in prompt_pieces:
            piece.id = uuid4()
            if piece.orchestrator_identifier["id"] == new_orchestrator_id:
                raise ValueError("The new orchestrator ID must be different from the existing orchestrator ID.")
            piece.orchestrator_identifier["id"] = new_orchestrator_id
            piece.conversation_id = new_conversation_id

        self.add_request_pieces_to_memory(request_pieces=prompt_pieces)

    def export_conversation_by_orchestrator_id(
        self, *, orchestrator_id: int, file_path: Path = None, export_type: str = "json"
    ):
        """
        Exports conversation data with the given orchestrator ID to a specified file.
        This will contain all conversations that were sent by the same orchestrator.

        Args:
            orchestrator_id (str): The ID of the orchestrator from which to export conversations.
            file_path (str): The path to the file where the data will be exported.
            If not provided, a default path using RESULTS_PATH will be constructed.
            export_type (str): The format of the export. Defaults to "json".
        """
        data = self._get_prompt_pieces_by_orchestrator(orchestrator_id=orchestrator_id)

        # If file_path is not provided, construct a default using the exporter's results_path
        if not file_path:
            file_name = f"{str(orchestrator_id)}.{export_type}"
            file_path = RESULTS_PATH / file_name

        self.exporter.export_data(data, file_path=file_path, export_type=export_type)

    def add_request_response_to_memory(self, *, request: PromptRequestResponse) -> None:
        """
        Inserts a list of prompt request pieces into the memory storage.

        Automatically updates the sequence to be the next number in the conversation.
        If necessary, generates embedding data for applicable entries

        Args:
            request (PromptRequestPiece): The request piece to add to the memory.

        Returns:
            None
        """
        request.validate()

        embedding_entries = []
        request_pieces = request.request_pieces

        self._update_sequence(request_pieces=request_pieces)

        self.add_request_pieces_to_memory(request_pieces=request_pieces)

        if self.memory_embedding:
            for piece in request_pieces:
                embedding_entry = self.memory_embedding.generate_embedding_memory_data(prompt_request_piece=piece)
                embedding_entries.append(embedding_entry)

            self._add_embeddings_to_memory(embedding_data=embedding_entries)

    def _update_sequence(self, *, request_pieces: list[PromptRequestPiece]):
        """
        Updates the sequence number of the request pieces in the conversation.

        Args:
            request_pieces (list[PromptRequestPiece]): The list of request pieces to update.
        """

        prev_conversations = self._get_prompt_pieces_with_conversation_id(
            conversation_id=request_pieces[0].conversation_id
        )

        sequence = 0

        if len(prev_conversations) > 0:
            sequence = max(prev_conversations, key=lambda item: item.sequence).sequence + 1

        for piece in request_pieces:
            piece.sequence = sequence

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
        memory_entries = self._get_prompt_pieces_with_conversation_id(conversation_id=conversation_id)
        return [ChatMessage(role=me.role, content=me.converted_value) for me in memory_entries]  # type: ignore

    def export_conversation_by_id(self, *, conversation_id: str, file_path: Path = None, export_type: str = "json"):
        """
        Exports conversation data with the given conversation ID to a specified file.

        Args:
            conversation_id (str): The ID of the conversation to export.
            file_path (str): The path to the file where the data will be exported.
            If not provided, a default path using RESULTS_PATH will be constructed.
            export_type (str): The format of the export. Defaults to "json".
        """
        data = self._get_prompt_pieces_with_conversation_id(conversation_id=conversation_id)

        # If file_path is not provided, construct a default using the exporter's results_path
        if not file_path:
            file_name = f"{conversation_id}.{export_type}"
            file_path = RESULTS_PATH / file_name

        self.exporter.export_data(data, file_path=file_path, export_type=export_type)
