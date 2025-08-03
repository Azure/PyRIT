# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Prompts and conversations mixin for MemoryInterface containing prompt and conversation-related operations."""

import copy
import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, MutableSequence, Optional, Sequence, Union

from sqlalchemy import and_

from pyrit.memory.memory_interface.protocol import MemoryInterfaceProtocol
from pyrit.memory.memory_models import PromptMemoryEntry
from pyrit.models import (
    ChatMessage,
    PromptRequestPiece,
    PromptRequestResponse,
    group_conversation_request_pieces_by_sequence,
    sort_request_pieces,
)

logger = logging.getLogger(__name__)

# Use protocol inheritance only during type checking to avoid metaclass conflicts.
# The protocol uses typing._ProtocolMeta which conflicts with the Singleton metaclass
# used by concrete memory classes. This conditional inheritance provides full type
# checking and IDE support while avoiding runtime metaclass conflicts.
if TYPE_CHECKING:
    _MixinBase = MemoryInterfaceProtocol
else:
    _MixinBase = object


class MemoryPromptsMixin(_MixinBase):
    """Mixin providing prompt and conversation-related methods for memory management."""

    def get_conversation(self, *, conversation_id: str) -> MutableSequence[PromptRequestResponse]:
        """
        Retrieves a list of PromptRequestResponse objects that have the specified conversation ID.

        Args:
            conversation_id (str): The conversation ID to match.

        Returns:
            MutableSequence[PromptRequestResponse]: A list of chat memory entries with the specified conversation ID.
        """
        request_pieces = self.get_prompt_request_pieces(conversation_id=conversation_id)
        return group_conversation_request_pieces_by_sequence(request_pieces=request_pieces)

    def get_prompt_request_pieces(
        self,
        *,
        orchestrator_id: Optional[str | uuid.UUID] = None,
        role: Optional[str] = None,
        conversation_id: Optional[str | uuid.UUID] = None,
        prompt_ids: Optional[Sequence[str | uuid.UUID]] = None,
        labels: Optional[dict[str, str]] = None,
        prompt_metadata: Optional[dict[str, Union[str, int]]] = None,
        sent_after: Optional[datetime] = None,
        sent_before: Optional[datetime] = None,
        original_values: Optional[Sequence[str]] = None,
        converted_values: Optional[Sequence[str]] = None,
        data_type: Optional[str] = None,
        not_data_type: Optional[str] = None,
        converted_value_sha256: Optional[Sequence[str]] = None,
    ) -> Sequence[PromptRequestPiece]:
        """
        Retrieves a list of PromptRequestPiece objects based on the specified filters.

        Args:
            orchestrator_id (Optional[str | uuid.UUID], optional): The ID of the orchestrator. Defaults to None.
            role (Optional[str], optional): The role of the prompt. Defaults to None.
            conversation_id (Optional[str | uuid.UUID], optional): The ID of the conversation. Defaults to None.
            prompt_ids (Optional[Sequence[str] | Sequence[uuid.UUID]], optional): A list of prompt IDs.
                Defaults to None.
            labels (Optional[dict[str, str]], optional): A dictionary of labels. Defaults to None.
            sent_after (Optional[datetime], optional): Filter for prompts sent after this datetime. Defaults to None.
            sent_before (Optional[datetime], optional): Filter for prompts sent before this datetime. Defaults to None.
            original_values (Optional[Sequence[str]], optional): A list of original values. Defaults to None.
            converted_values (Optional[Sequence[str]], optional): A list of converted values. Defaults to None.
            data_type (Optional[str], optional): The data type to filter by. Defaults to None.
            not_data_type (Optional[str], optional): The data type to exclude. Defaults to None.
            converted_value_sha256 (Optional[Sequence[str]], optional): A list of SHA256 hashes of converted values.
                Defaults to None.
        Returns:
            Sequence[PromptRequestPiece]: A list of PromptRequestPiece objects that match the specified filters.
        Raises:
            Exception: If there is an error retrieving the prompts,
                an exception is logged and an empty list is returned.
        """

        conditions = []
        if orchestrator_id:
            conditions.append(self._get_prompt_pieces_orchestrator_conditions(orchestrator_id=str(orchestrator_id)))
        if role:
            conditions.append(PromptMemoryEntry.role == role)
        if conversation_id:
            conditions.append(PromptMemoryEntry.conversation_id == str(conversation_id))
        if prompt_ids:
            prompt_ids = [str(pi) for pi in prompt_ids]
            conditions.append(PromptMemoryEntry.id.in_(prompt_ids))
        if labels:
            conditions.append(self._get_prompt_pieces_memory_label_conditions(memory_labels=labels))
        if prompt_metadata:
            conditions.append(self._get_prompt_pieces_prompt_metadata_conditions(prompt_metadata=prompt_metadata))
        if sent_after:
            conditions.append(PromptMemoryEntry.timestamp >= sent_after)
        if sent_before:
            conditions.append(PromptMemoryEntry.timestamp <= sent_before)
        if original_values:
            conditions.append(PromptMemoryEntry.original_value.in_(original_values))
        if converted_values:
            conditions.append(PromptMemoryEntry.converted_value.in_(converted_values))
        if data_type:
            conditions.append(PromptMemoryEntry.converted_value_data_type == data_type)
        if not_data_type:
            conditions.append(PromptMemoryEntry.converted_value_data_type != not_data_type)
        if converted_value_sha256:
            conditions.append(PromptMemoryEntry.converted_value_sha256.in_(converted_value_sha256))

        try:
            memory_entries: Sequence[PromptMemoryEntry] = self._query_entries(
                PromptMemoryEntry, conditions=and_(*conditions) if conditions else None, join_scores=True
            )  # type: ignore
            prompt_pieces = [memory_entry.get_prompt_request_piece() for memory_entry in memory_entries]
            return sort_request_pieces(prompt_pieces=prompt_pieces)
        except Exception as e:
            logger.exception(f"Failed to retrieve prompts with error {e}")
            return []

    def duplicate_conversation(self, *, conversation_id: str, new_orchestrator_id: Optional[str] = None) -> str:
        """
        Duplicates a conversation for reuse

        This can be useful when an attack strategy requires branching out from a particular point in the conversation.
        One cannot continue both branches with the same orchestrator and conversation IDs since that would corrupt
        the memory. Instead, one needs to duplicate the conversation and continue with the new orchestrator ID.

        Args:
            conversation_id (str): The conversation ID with existing conversations.
            new_orchestrator_id (str, Optional): The new orchestrator ID to assign to the duplicated conversations.
                If no new orchestrator ID is provided, the orchestrator ID will remain the same. Defaults to None.
        Returns:
            The uuid for the new conversation.
        """
        new_conversation_id = str(uuid.uuid4())
        # Deep copy objects to prevent any mutability-related issues that could arise due to in-memory databases.
        prompt_pieces = copy.deepcopy(self.get_prompt_request_pieces(conversation_id=conversation_id))
        for piece in prompt_pieces:
            # Assign duplicated piece a new ID, but note that the `original_prompt_id` remains the same.
            piece.id = uuid.uuid4()
            if piece.orchestrator_identifier["id"] == new_orchestrator_id:
                raise ValueError("The new orchestrator ID must be different from the existing orchestrator ID.")

            if new_orchestrator_id:
                piece.orchestrator_identifier["id"] = new_orchestrator_id

            piece.conversation_id = new_conversation_id

        self.add_request_pieces_to_memory(request_pieces=prompt_pieces)
        return new_conversation_id

    def duplicate_conversation_excluding_last_turn(
        self, *, conversation_id: str, new_orchestrator_id: Optional[str] = None
    ) -> str:
        """
        Duplicate a conversation, excluding the last turn. In this case, last turn is defined as before the last
        user request (e.g. if there is half a turn, it just removes that half).

        This can be useful when an attack strategy requires back tracking the last prompt/response pair.

        Args:
            conversation_id (str): The conversation ID with existing conversations.
            new_orchestrator_id (str, Optional): The new orchestrator ID to assign to the duplicated conversations.
                If no new orchestrator ID is provided, the orchestrator ID will remain the same. Defaults to None.
        Returns:
            The uuid for the new conversation.
        """
        new_conversation_id = str(uuid.uuid4())
        # Deep copy objects to prevent any mutability-related issues that could arise due to in-memory databases.
        prompt_pieces = copy.deepcopy(self.get_prompt_request_pieces(conversation_id=conversation_id))

        # remove the final turn from the conversation
        if len(prompt_pieces) == 0:
            return new_conversation_id

        last_prompt = max(prompt_pieces, key=lambda x: x.sequence)

        length_of_sequence_to_remove = 0

        if last_prompt.role == "system" or last_prompt.role == "user":
            length_of_sequence_to_remove = 1
        else:
            length_of_sequence_to_remove = 2

        prompt_pieces = [
            prompt_piece
            for prompt_piece in prompt_pieces
            if prompt_piece.sequence <= last_prompt.sequence - length_of_sequence_to_remove
        ]

        for piece in prompt_pieces:
            # Assign duplicated piece a new ID, but note that the `original_prompt_id` remains the same.
            piece.id = uuid.uuid4()
            if new_orchestrator_id:
                piece.orchestrator_identifier["id"] = new_orchestrator_id
            piece.conversation_id = new_conversation_id

        self.add_request_pieces_to_memory(request_pieces=prompt_pieces)

        return new_conversation_id

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

    def _update_sequence(self, *, request_pieces: Sequence[PromptRequestPiece]):
        """
        Updates the sequence number of the request pieces in the conversation.

        Args:
            request_pieces (Sequence[PromptRequestPiece]): The list of request pieces to update.
        """

        prev_conversations = self.get_prompt_request_pieces(conversation_id=request_pieces[0].conversation_id)

        sequence = 0

        if len(prev_conversations) > 0:
            sequence = max(prev_conversations, key=lambda item: item.sequence).sequence + 1

        for piece in request_pieces:
            piece.sequence = sequence

    def update_prompt_entries_by_conversation_id(self, *, conversation_id: str, update_fields: dict) -> bool:
        """
        Updates prompt entries for a given conversation ID with the specified field values.

        Args:
            conversation_id (str): The conversation ID of the entries to be updated.
            update_fields (dict): A dictionary of field names and their new values (ex. {"labels": {"test": "value"}})

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        if not update_fields:
            raise ValueError("update_fields must be provided to update prompt entries.")
        # Fetch the relevant entries using query_entries
        entries_to_update = self._query_entries(
            PromptMemoryEntry, conditions=PromptMemoryEntry.conversation_id == conversation_id
        )
        # Check if there are entries to update
        if not entries_to_update:
            logger.info(f"No entries found with conversation_id {conversation_id} to update.")
            return False

        # Use the utility function to update the entries
        success = self._update_entries(entries=entries_to_update, update_fields=update_fields)

        if success:
            logger.info(f"Updated {len(entries_to_update)} entries with conversation_id {conversation_id}.")
        else:
            logger.error(f"Failed to update entries with conversation_id {conversation_id}.")
        return success

    def update_labels_by_conversation_id(self, *, conversation_id: str, labels: dict) -> bool:
        """
        Updates the labels of prompt entries in memory for a given conversation ID.

        Args:
            conversation_id (str): The conversation ID of the entries to be updated.
            labels (dict): New dictionary of labels.

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        return self.update_prompt_entries_by_conversation_id(
            conversation_id=conversation_id, update_fields={"labels": labels}
        )

    def update_prompt_metadata_by_conversation_id(
        self, *, conversation_id: str, prompt_metadata: dict[str, Union[str, int]]
    ) -> bool:
        """
        Updates the metadata of prompt entries in memory for a given conversation ID.

        Args:
            conversation_id (str): The conversation ID of the entries to be updated.
            metadata (dict[str, str | int]): New metadata.

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        return self.update_prompt_entries_by_conversation_id(
            conversation_id=conversation_id, update_fields={"prompt_metadata": prompt_metadata}
        )

    def get_chat_messages_with_conversation_id(self, *, conversation_id: str) -> Sequence[ChatMessage]:
        """
        Returns the memory for a given conversation_id.

        Args:
            conversation_id (str): The conversation ID.

        Returns:
            Sequence[ChatMessage]: The list of chat messages.
        """
        memory_entries = self.get_prompt_request_pieces(conversation_id=conversation_id)
        return [ChatMessage(role=me.role, content=me.converted_value) for me in memory_entries]  # type: ignore
