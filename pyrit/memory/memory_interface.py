# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import atexit
import copy
import logging
import uuid
import weakref
from datetime import datetime
from pathlib import Path
from typing import MutableSequence, Optional, Sequence, Tuple, TypeVar, Union

from sqlalchemy import and_
from sqlalchemy.orm.attributes import InstrumentedAttribute

from pyrit.common.path import DB_DATA_PATH
from pyrit.memory.memory_embedding import (
    MemoryEmbedding,
    default_memory_embedding_factory,
)
from pyrit.memory.memory_exporter import MemoryExporter
from pyrit.memory.memory_models import (
    Base,
    EmbeddingDataEntry,
    PromptMemoryEntry,
    ScoreEntry,
    SeedPromptEntry,
)
from pyrit.models import (
    ChatMessage,
    DataTypeSerializer,
    PromptRequestPiece,
    PromptRequestResponse,
    Score,
    SeedPrompt,
    SeedPromptDataset,
    SeedPromptGroup,
    StorageIO,
    data_serializer_factory,
    group_conversation_request_pieces_by_sequence,
    sort_request_pieces,
)

logger = logging.getLogger(__name__)


Model = TypeVar("Model")


class MemoryInterface(abc.ABC):
    """
    Represents a conversation memory that stores chat messages. This class must be overwritten with a
    specific implementation to store the memory objects (e.g. relational database, NoSQL database, etc.)

    Args:
        embedding_model (EmbeddingSupport): If set, this includes embeddings in the memory entries
        which are extremely useful for comparing chat messages and similarities, but also includes overhead
    """

    memory_embedding: MemoryEmbedding = None
    results_storage_io: StorageIO = None
    results_path: str = None

    def __init__(self, embedding_model=None):
        self.memory_embedding = embedding_model
        # Initialize the MemoryExporter instance
        self.exporter = MemoryExporter()
        self._init_storage_io()

        # Ensure cleanup at process exit
        self.cleanup()

    def enable_embedding(self, embedding_model=None):
        self.memory_embedding = default_memory_embedding_factory(embedding_model=embedding_model)

    def disable_embedding(self):
        self.memory_embedding = None

    @abc.abstractmethod
    def get_all_embeddings(self) -> Sequence[EmbeddingDataEntry]:
        """
        Loads all EmbeddingData from the memory storage handler.
        """

    @abc.abstractmethod
    def _init_storage_io(self):
        """
        Initialize the storage IO handler results_storage_io.
        """

    @abc.abstractmethod
    def _get_prompt_pieces_memory_label_conditions(self, *, memory_labels: dict[str, str]) -> list:
        """
        Returns a list of conditions for filtering memory entries based on memory labels.

        Args:
            memory_labels (dict[str, str]): A free-form dictionary for tagging prompts with custom labels.
                These labels can be used to track all prompts sent as part of an operation, score prompts based on
                the operation ID (op_id), and tag each prompt with the relevant Responsible AI (RAI) harm category.
                Users can define any key-value pairs according to their needs.

        Returns:
            list: A list of conditions for filtering memory entries based on memory labels.
        """

    @abc.abstractmethod
    def _get_prompt_pieces_prompt_metadata_conditions(self, *, prompt_metadata: dict[str, Union[str, int]]) -> list:
        """
        Returns a list of conditions for filtering memory entries based on prompt metadata.

        Args:
            prompt_metadata (dict[str, str | int]): A free-form dictionary for tagging prompts with custom metadata.
                This includes information that is useful for the specific target you're probing, such as encoding data.

        Returns:
            list: A list of conditions for filtering memory entries based on prompt metadata.
        """

    @abc.abstractmethod
    def _get_prompt_pieces_orchestrator_conditions(self, *, orchestrator_id: str):
        """
        Returns a condition to retrieve based on orchestrator ID.
        """

    @abc.abstractmethod
    def _get_seed_prompts_metadata_conditions(self, *, metadata: dict[str, Union[str, int]]):
        """
        Returns a list of conditions for filtering seed prompt entries based on prompt metadata.

        Args:
            prompt_metadata (dict[str, str | int]): A free-form dictionary for tagging prompts with custom metadata.
                This includes information that is useful for the specific target you're probing, such as encoding data.

        Returns:
            list: A list of conditions for filtering memory entries based on prompt metadata.
        """

    @abc.abstractmethod
    def add_request_pieces_to_memory(self, *, request_pieces: Sequence[PromptRequestPiece]) -> None:
        """
        Inserts a list of prompt request pieces into the memory storage.
        """

    @abc.abstractmethod
    def _add_embeddings_to_memory(self, *, embedding_data: Sequence[EmbeddingDataEntry]) -> None:
        """
        Inserts embedding data into memory storage
        """

    @abc.abstractmethod
    def _query_entries(
        self, Model, *, conditions: Optional = None, distinct: bool = False, join_scores: bool = False  # type: ignore
    ) -> MutableSequence[Model]:  # type: ignore
        """
        Fetches data from the specified table model with optional conditions.

        Args:
            model: The SQLAlchemy model class corresponding to the table you want to query.
            conditions: SQLAlchemy filter conditions (Optional).
            distinct: Whether to return distinct rows only. Defaults to False.
            join_scores: Whether to join the scores table. Defaults to False.

        Returns:
            List of model instances representing the rows fetched from the table.
        """

    @abc.abstractmethod
    def _insert_entry(self, entry: Base) -> None:  # type: ignore
        """
        Inserts an entry into the Table.

        Args:
            entry: An instance of a SQLAlchemy model to be added to the Table.
        """

    @abc.abstractmethod
    def _insert_entries(self, *, entries: Sequence[Base]) -> None:  # type: ignore
        """Inserts multiple entries into the database."""

    @abc.abstractmethod
    def _update_entries(self, *, entries: MutableSequence[Base], update_fields: dict) -> bool:  # type: ignore
        """
        Updates the given entries with the specified field values.

        Args:
            entries (Sequence[Base]): A list of SQLAlchemy model instances to be updated.
            update_fields (dict): A dictionary of field names and their new values.
        """

    def add_scores_to_memory(self, *, scores: Sequence[Score]) -> None:
        """
        Inserts a list of scores into the memory storage.
        """
        for score in scores:
            if score.prompt_request_response_id:
                prompt_request_response_id = score.prompt_request_response_id
                prompt_piece = self.get_prompt_request_pieces(prompt_ids=[str(prompt_request_response_id)])
                if not prompt_piece:
                    logging.error(f"Prompt with ID {prompt_request_response_id} not found in memory.")
                    continue
                # auto-link score to the original prompt id if the prompt is a duplicate
                if prompt_piece[0].original_prompt_id != prompt_piece[0].id:
                    score.prompt_request_response_id = prompt_piece[0].original_prompt_id
        self._insert_entries(entries=[ScoreEntry(entry=score) for score in scores])

    def get_scores_by_prompt_ids(self, *, prompt_request_response_ids: Sequence[str]) -> Sequence[Score]:
        """
        Gets a list of scores based on prompt_request_response_ids.
        """
        prompt_pieces = self.get_prompt_request_pieces(prompt_ids=prompt_request_response_ids)
        # Get the original prompt IDs from the prompt pieces so correct scores can be obtained
        prompt_request_response_ids = [str(piece.original_prompt_id) for piece in prompt_pieces]
        entries: Sequence[ScoreEntry] = self._query_entries(
            ScoreEntry, conditions=ScoreEntry.prompt_request_response_id.in_(prompt_request_response_ids)
        )

        return [entry.get_score() for entry in entries]

    def get_scores_by_orchestrator_id(self, *, orchestrator_id: str) -> Sequence[Score]:
        """
        Retrieves a list of Score objects associated with the PromptRequestPiece objects
        which have the specified orchestrator ID.

        Args:
            orchestrator_id (str): The id of the orchestrator.
                Can be retrieved by calling orchestrator.get_identifier()["id"]

        Returns:
            Sequence[Score]: A list of Score objects associated with the PromptRequestPiece objects
                which match the specified orchestrator ID.
        """
        prompt_pieces = self.get_prompt_request_pieces(orchestrator_id=orchestrator_id)
        # Since duplicate pieces do not have their own score entries, get the original prompt IDs from the pieces.
        prompt_ids = [str(piece.original_prompt_id) for piece in prompt_pieces]
        return self.get_scores_by_prompt_ids(prompt_request_response_ids=prompt_ids)

    def get_scores_by_memory_labels(self, *, memory_labels: dict[str, str]) -> Sequence[Score]:
        """
        Retrieves a list of Score objects associated with the PromptRequestPiece objects
        which have the specified memory labels.

        Args:
            memory_labels (dict[str, str]): A free-form dictionary for tagging prompts with custom labels.
                These labels can be used to track all prompts sent as part of an operation, score prompts based on
                the operation ID (op_id), and tag each prompt with the relevant Responsible AI (RAI) harm category.
                Users can define any key-value pairs according to their needs.

        Returns:
            Sequence[Score]: A list of Score objects associated with the PromptRequestPiece objects
                which match the specified memory labels.
        """
        prompt_pieces = self.get_prompt_request_pieces(labels=memory_labels)
        # Since duplicate pieces do not have their own score entries, get the original prompt IDs from the pieces.
        prompt_ids = [str(piece.original_prompt_id) for piece in prompt_pieces]
        return self.get_scores_by_prompt_ids(prompt_request_response_ids=prompt_ids)

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
        entries_to_update: MutableSequence[Base] = self._query_entries(
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

    @abc.abstractmethod
    def dispose_engine(self):
        """
        Dispose the engine and clean up resources.
        """

    def cleanup(self):
        """
        Ensure cleanup on process exit
        """
        # Ensure cleanup at process exit
        atexit.register(self.dispose_engine)

        # Ensure cleanup happens even if the object is garbage collected before process exits
        weakref.finalize(self, self.dispose_engine)

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

    def get_seed_prompts(
        self,
        *,
        value: Optional[str] = None,
        value_sha256: Optional[Sequence[str]] = None,
        dataset_name: Optional[str] = None,
        data_types: Optional[Sequence[str]] = None,
        harm_categories: Optional[Sequence[str]] = None,
        added_by: Optional[str] = None,
        authors: Optional[Sequence[str]] = None,
        groups: Optional[Sequence[str]] = None,
        source: Optional[str] = None,
        parameters: Optional[Sequence[str]] = None,
        metadata: Optional[dict[str, Union[str, int]]] = None,
    ) -> Sequence[SeedPrompt]:
        """
        Retrieves a list of seed prompts based on the specified filters.

        Args:
            value (str): The value to match by substring. If None, all values are returned.
            value_sha256 (str): The SHA256 hash of the value to match. If None, all values are returned.
            dataset_name (str): The dataset name to match. If None, all dataset names are considered.
            data_types (Optional[Sequence[str], Optional): List of data types to filter seed prompts by
                (e.g., text, image_path).
            harm_categories (Sequence[str]): A list of harm categories to filter by. If None,
            all harm categories are considered.
                Specifying multiple harm categories returns only prompts that are marked with all harm categories.
            added_by (str): The user who added the prompts.
            authors (Sequence[str]): A list of authors to filter by.
                Note that this filters by substring, so a query for "Adam Jones" may not return results if the record
                is "A. Jones", "Jones, Adam", etc. If None, all authors are considered.
            groups (Sequence[str]): A list of groups to filter by. If None, all groups are considered.
            source (str): The source to filter by. If None, all sources are considered.
            parameters (Sequence[str]): A list of parameters to filter by. Specifying parameters effectively returns
                prompt templates instead of prompts.

        Returns:
            Sequence[SeedPrompt]: A list of prompts matching the criteria.
        """
        conditions = []

        # Apply filters for non-list fields
        if value:
            conditions.append(SeedPromptEntry.value.contains(value))
        if value_sha256:
            conditions.append(SeedPromptEntry.value_sha256.in_(value_sha256))
        if dataset_name:
            conditions.append(SeedPromptEntry.dataset_name == dataset_name)
        if data_types:
            data_type_conditions = SeedPromptEntry.data_type.in_(data_types)
            conditions.append(data_type_conditions)
        if added_by:
            conditions.append(SeedPromptEntry.added_by == added_by)
        if source:
            conditions.append(SeedPromptEntry.source == source)

        self._add_list_conditions(SeedPromptEntry.harm_categories, harm_categories, conditions)
        self._add_list_conditions(SeedPromptEntry.authors, authors, conditions)
        self._add_list_conditions(SeedPromptEntry.groups, groups, conditions)

        if parameters:
            self._add_list_conditions(SeedPromptEntry.parameters, parameters, conditions)

        if metadata:
            conditions.append(self._get_seed_prompts_metadata_conditions(metadata=metadata))

        try:
            memory_entries: Sequence[SeedPromptEntry] = self._query_entries(
                SeedPromptEntry,
                conditions=and_(*conditions) if conditions else None,
            )  # type: ignore
            return [memory_entry.get_seed_prompt() for memory_entry in memory_entries]
        except Exception as e:
            logger.exception(f"Failed to retrieve prompts with dataset name {dataset_name} with error {e}")
            return []

    def _add_list_conditions(
        self, field: InstrumentedAttribute, values: Optional[Sequence[str]], conditions: list
    ) -> None:
        if values:
            for value in values:
                conditions.append(field.contains(value))

    async def _serialize_seed_prompt_value(self, prompt: SeedPrompt) -> str:
        """
        Serializes the value of a seed prompt based on its data type.

        Args:
            prompt (SeedPrompt): The seed prompt to serialize. Must have a valid `data_type`.

        Returns:
            str: The serialized value for the prompt.

        Raises:
            ValueError: If the `data_type` of the prompt is unsupported.
        """
        extension = DataTypeSerializer.get_extension(prompt.value)
        if extension:
            extension = extension.lstrip(".")
        serializer = data_serializer_factory(
            category="seed-prompt-entries", data_type=prompt.data_type, value=prompt.value, extension=extension
        )
        serialized_prompt_value = None
        if prompt.data_type == "image_path":
            # Read the image
            original_img_bytes = await serializer.read_data_base64()
            # Save the image
            await serializer.save_b64_image(original_img_bytes)
            serialized_prompt_value = str(serializer.value)
        elif prompt.data_type in ["audio_path", "video_path"]:
            audio_bytes = await serializer.read_data()
            await serializer.save_data(data=audio_bytes)
            serialized_prompt_value = str(serializer.value)
        return serialized_prompt_value

    async def add_seed_prompts_to_memory_async(
        self, *, prompts: Sequence[SeedPrompt], added_by: Optional[str] = None
    ) -> None:
        """
        Inserts a list of prompts into the memory storage.

        Args:
            prompts (Sequence[SeedPrompt]): A list of prompts to insert.
            added_by (str): The user who added the prompts.
        """
        entries: MutableSequence[SeedPromptEntry] = []
        current_time = datetime.now()
        for prompt in prompts:
            if added_by:
                prompt.added_by = added_by
            if not prompt.added_by:
                raise ValueError(
                    """The 'added_by' attribute must be set for each prompt.
                    Set it explicitly or pass a value to the 'added_by' parameter."""
                )
            if prompt.date_added is None:
                prompt.date_added = current_time

            prompt.set_encoding_metadata()

            serialized_prompt_value = await self._serialize_seed_prompt_value(prompt)
            if serialized_prompt_value:
                prompt.value = serialized_prompt_value

            await prompt.set_sha256_value_async()

            if not self.get_seed_prompts(value_sha256=[prompt.value_sha256], dataset_name=prompt.dataset_name):
                entries.append(SeedPromptEntry(entry=prompt))

        self._insert_entries(entries=entries)

    def get_seed_prompt_dataset_names(self) -> Sequence[str]:
        """
        Returns a list of all seed prompt dataset names in the memory storage.
        """
        try:
            entries: Sequence[Tuple[str]] = self._query_entries(
                SeedPromptEntry.dataset_name,
                conditions=and_(
                    SeedPromptEntry.dataset_name is not None, SeedPromptEntry.dataset_name != ""  # type: ignore
                ),
                distinct=True,
            )  # type: ignore
            # return value is list of tuples with a single entry (the dataset name)
            return [entry[0] for entry in entries]
        except Exception as e:
            logger.exception(f"Failed to retrieve dataset names with error {e}")
            return []

    async def add_seed_prompt_groups_to_memory(
        self, *, prompt_groups: Sequence[SeedPromptGroup], added_by: Optional[str] = None
    ) -> None:
        """
        Inserts a list of seed prompt groups into the memory storage.

        Args:
            prompt_groups (Sequence[SeedPromptGroup]): A list of prompt groups to insert.
            added_by (str): The user who added the prompt groups.

        Raises:
            ValueError: If a prompt group does not have at least one prompt.
            ValueError: If prompt group IDs are inconsistent within the same prompt group.
        """
        if not prompt_groups:
            raise ValueError("At least one prompt group must be provided.")
        # Validates the prompt group IDs and sets them if possible before leveraging
        # the add_seed_prompts_to_memory method.
        all_prompts: MutableSequence[SeedPrompt] = []
        for prompt_group in prompt_groups:
            if not prompt_group.prompts:
                raise ValueError("Prompt group must have at least one prompt.")
            # Determine the prompt group ID.
            # It should either be set uniformly or generated if not set.
            # Inconsistent prompt group IDs will raise an error.
            group_id_set = set(prompt.prompt_group_id for prompt in prompt_group.prompts)
            if len(group_id_set) > 1:
                raise ValueError(
                    f"""Inconsistent 'prompt_group_id' attribute between members of the
                    same prompt group. Found {group_id_set}"""
                )
            prompt_group_id = group_id_set.pop() or uuid.uuid4()
            for prompt in prompt_group.prompts:
                prompt.prompt_group_id = prompt_group_id
            all_prompts.extend(prompt_group.prompts)
        await self.add_seed_prompts_to_memory_async(prompts=all_prompts, added_by=added_by)

    def get_seed_prompt_groups(
        self,
        *,
        value_sha256: Optional[Sequence[str]] = None,
        dataset_name: Optional[str] = None,
        data_types: Optional[Sequence[str]] = None,
        harm_categories: Optional[Sequence[str]] = None,
        added_by: Optional[str] = None,
        authors: Optional[Sequence[str]] = None,
        groups: Optional[Sequence[str]] = None,
        source: Optional[str] = None,
    ) -> Sequence[SeedPromptGroup]:
        """Retrieves groups of seed prompts based on the provided filtering criteria.

        Args:
            value_sha256 (Optional[Sequence[str]], Optional): SHA256 hash of value to filter seed prompt groups by.
            dataset_name (Optional[str], Optional): Name of the dataset to filter seed prompts.
            data_types (Optional[Sequence[str]], Optional): List of data types to filter seed prompts by
            (e.g., text, image_path).
            harm_categories (Optional[Sequence[str]], Optional): List of harm categories to filter seed prompts by.
            added_by (Optional[str], Optional): The user who added the seed prompt groups to filter by.
            authors (Optional[Sequence[str]], Optional): List of authors to filter seed prompt groups by.
            groups (Optional[Sequence[str]], Optional): List of groups to filter seed prompt groups by.
            source (Optional[str], Optional): The source from which the seed prompts originated.

        Returns:
            Sequence[SeedPromptGroup]: A list of `SeedPromptGroup` objects that match the filtering criteria.
        """
        seed_prompts = self.get_seed_prompts(
            value_sha256=value_sha256,
            dataset_name=dataset_name,
            data_types=data_types,
            harm_categories=harm_categories,
            added_by=added_by,
            authors=authors,
            groups=groups,
            source=source,
        )
        seed_prompt_groups = SeedPromptDataset.group_seed_prompts_by_prompt_group_id(seed_prompts)
        return seed_prompt_groups

    def export_conversations(
        self,
        *,
        orchestrator_id: Optional[str | uuid.UUID] = None,
        conversation_id: Optional[str | uuid.UUID] = None,
        prompt_ids: Optional[Sequence[str] | Sequence[uuid.UUID]] = None,
        labels: Optional[dict[str, str]] = None,
        sent_after: Optional[datetime] = None,
        sent_before: Optional[datetime] = None,
        original_values: Optional[Sequence[str]] = None,
        converted_values: Optional[Sequence[str]] = None,
        data_type: Optional[str] = None,
        not_data_type: Optional[str] = None,
        converted_value_sha256: Optional[Sequence[str]] = None,
        file_path: Optional[Path] = None,
        export_type: str = "json",
    ) -> Path:
        """
        Exports conversation data with the given inputs to a specified file.
            Defaults to all conversations if no filters are provided.

        Args:
            orchestrator_id (Optional[str | uuid.UUID], optional): The ID of the orchestrator. Defaults to None.
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
            file_path (Optional[Path], optional): The path to the file where the data will be exported.
                Defaults to None.
            export_type (str, optional): The format of the export. Defaults to "json".
        """
        data = self.get_prompt_request_pieces(
            orchestrator_id=orchestrator_id,
            conversation_id=conversation_id,
            prompt_ids=prompt_ids,
            labels=labels,
            sent_after=sent_after,
            sent_before=sent_before,
            original_values=original_values,
            converted_values=converted_values,
            data_type=data_type,
            not_data_type=not_data_type,
            converted_value_sha256=converted_value_sha256,
        )

        # If file_path is not provided, construct a default using the exporter's results_path
        if not file_path:
            file_name = f"exported_conversations_on_{datetime.now().strftime('%Y_%m_%d')}.{export_type}"
            file_path = DB_DATA_PATH / file_name

        self.exporter.export_data(data, file_path=file_path, export_type=export_type)

        return file_path
