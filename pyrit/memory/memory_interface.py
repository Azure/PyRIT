# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from collections import defaultdict
import copy
from datetime import datetime
import logging
from pathlib import Path
from sqlalchemy import and_
from sqlalchemy.orm.attributes import InstrumentedAttribute
from typing import MutableSequence, Optional, Sequence, Union
import uuid

from pyrit.common.path import RESULTS_PATH
from pyrit.models import (
    ChatMessage,
    group_conversation_request_pieces_by_sequence,
    PromptRequestResponse,
    PromptRequestPiece,
    Score,
    SeedPrompt,
    SeedPromptDataset,
    SeedPromptGroup,
    StorageIO,
)

from pyrit.memory.memory_models import Base, EmbeddingDataEntry, PromptMemoryEntry, ScoreEntry, SeedPromptEntry
from pyrit.memory.memory_embedding import default_memory_embedding_factory, MemoryEmbedding
from pyrit.memory.memory_exporter import MemoryExporter

logger = logging.getLogger(__name__)


class MemoryInterface(abc.ABC):
    """
    Represents a conversation memory that stores chat messages. This class must be overwritten with a
    specific implementation to store the memory objects (e.g. relational database, NoSQL database, etc.)

    Args:
        embedding_model (EmbeddingSupport): If set, this includes embeddings in the memory entries
        which are extremely useful for comparing chat messages and similarities, but also includes overhead
    """

    memory_embedding: MemoryEmbedding = None
    storage_io: StorageIO = None
    results_path: str = None

    def __init__(self, embedding_model=None):
        self.memory_embedding = embedding_model
        # Initialize the MemoryExporter instance
        self.exporter = MemoryExporter()
        self._init_storage_io()

    def enable_embedding(self, embedding_model=None):
        self.memory_embedding = default_memory_embedding_factory(embedding_model=embedding_model)

    def disable_embedding(self):
        self.memory_embedding = None

    @abc.abstractmethod
    def get_all_prompt_pieces(self) -> Sequence[PromptRequestPiece]:
        """
        Loads all ConversationData from the memory storage handler.
        """

    @abc.abstractmethod
    def get_all_embeddings(self) -> Sequence[EmbeddingDataEntry]:
        """
        Loads all EmbeddingData from the memory storage handler.
        """

    @abc.abstractmethod
    def _init_storage_io(self):
        """
        Initialize the storage IO handler storage_io.
        """

    @abc.abstractmethod
    def _get_prompt_pieces_with_conversation_id(self, *, conversation_id: str) -> MutableSequence[PromptRequestPiece]:
        """
        Retrieves a list of PromptRequestPiece objects that have the specified conversation ID.

        Args:
            conversation_id (str): The conversation ID to match.

        Returns:
            MutableSequence[PromptRequestPiece]: A list of chat memory entries with the specified conversation ID.
        """

    @abc.abstractmethod
    def _get_prompt_pieces_by_orchestrator(self, *, orchestrator_id: str) -> Sequence[PromptRequestPiece]:
        """
        Retrieves a list of PromptRequestPiece objects that have the specified orchestrator ID.

        Args:
            orchestrator_id (str): The id of the orchestrator.
                Can be retrieved by calling orchestrator.get_identifier()["id"]

        Returns:
            Sequence[PromptRequestPiece]: A list of PromptMemoryEntry objects matching the specified orchestrator ID.
        """

    @abc.abstractmethod
    def add_request_pieces_to_memory(self, *, request_pieces: Sequence[PromptRequestPiece]) -> None:
        """
        Inserts a list of prompt request pieces into the memory storage.
        """

    @abc.abstractmethod
    def _add_embeddings_to_memory(self, *, embedding_data: list[EmbeddingDataEntry]) -> None:
        """
        Inserts embedding data into memory storage
        """

    @abc.abstractmethod
    def query_entries(
        self, model, *, conditions: Optional = None, distinct: bool = False  # type: ignore
    ) -> list[Base]:  # type: ignore
        """
        Fetches data from the specified table model with optional conditions.

        Args:
            model: The SQLAlchemy model class corresponding to the table you want to query.
            conditions: SQLAlchemy filter conditions (Optional).
            distinct: Whether to return distinct rows only. Defaults to False.

        Returns:
            List of model instances representing the rows fetched from the table.
        """

    @abc.abstractmethod
    def insert_entry(self, entry: Base) -> None:  # type: ignore
        """
        Inserts an entry into the Table.

        Args:
            entry: An instance of a SQLAlchemy model to be added to the Table.
        """

    @abc.abstractmethod
    def insert_entries(self, *, entries: list[Base]) -> None:  # type: ignore
        """Inserts multiple entries into the database."""

    @abc.abstractmethod
    def update_entries(self, *, entries: MutableSequence[Base], update_fields: dict) -> bool:  # type: ignore
        """
        Updates the given entries with the specified field values.

        Args:
            entries (list[Base]): A list of SQLAlchemy model instances to be updated.
            update_fields (dict): A dictionary of field names and their new values.
        """

    def add_scores_to_memory(self, *, scores: list[Score]) -> None:
        """
        Inserts a list of scores into the memory storage.
        """
        for score in scores:
            if score.prompt_request_response_id:
                prompt_request_response_id = score.prompt_request_response_id
                prompt_piece = self.get_prompt_request_pieces_by_id(prompt_ids=[str(prompt_request_response_id)])
                if not prompt_piece:
                    logging.error(f"Prompt with ID {prompt_request_response_id} not found in memory.")
                    continue
                # auto-link score to the original prompt id if the prompt is a duplicate
                if prompt_piece[0].original_prompt_id != prompt_piece[0].id:
                    score.prompt_request_response_id = prompt_piece[0].original_prompt_id
        self.insert_entries(entries=[ScoreEntry(entry=score) for score in scores])

    def get_scores_by_prompt_ids(self, *, prompt_request_response_ids: list[str]) -> list[Score]:
        """
        Gets a list of scores based on prompt_request_response_ids.
        """
        prompt_pieces = self.get_prompt_request_pieces_by_id(prompt_ids=prompt_request_response_ids)
        # Get the original prompt IDs from the prompt pieces so correct scores can be obtained
        prompt_request_response_ids = [str(piece.original_prompt_id) for piece in prompt_pieces]
        entries = self.query_entries(
            ScoreEntry, conditions=ScoreEntry.prompt_request_response_id.in_(prompt_request_response_ids)
        )

        return [entry.get_score() for entry in entries]

    def get_scores_by_orchestrator_id(self, *, orchestrator_id: str) -> list[Score]:
        """
        Retrieves a list of Score objects associated with the PromptRequestPiece objects
        which have the specified orchestrator ID.

        Args:
            orchestrator_id (str): The id of the orchestrator.
                Can be retrieved by calling orchestrator.get_identifier()["id"]

        Returns:
            list[Score]: A list of Score objects associated with the PromptRequestPiece objects
                which match the specified orchestrator ID.
        """
        prompt_pieces = self.get_prompt_request_piece_by_orchestrator_id(orchestrator_id=orchestrator_id)
        # Since duplicate pieces do not have their own score entries, get the original prompt IDs from the pieces.
        prompt_ids = [str(piece.original_prompt_id) for piece in prompt_pieces]
        return self.get_scores_by_prompt_ids(prompt_request_response_ids=prompt_ids)

    def get_scores_by_memory_labels(self, *, memory_labels: dict[str, str]) -> list[Score]:
        """
        Retrieves a list of Score objects associated with the PromptRequestPiece objects
        which have the specified memory labels.

        Args:
            memory_labels (dict[str, str]): A free-form dictionary for tagging prompts with custom labels.
                These labels can be used to track all prompts sent as part of an operation, score prompts based on
                the operation ID (op_id), and tag each prompt with the relevant Responsible AI (RAI) harm category.
                Users can define any key-value pairs according to their needs.

        Returns:
            list[Score]: A list of Score objects associated with the PromptRequestPiece objects
                which match the specified memory labels.
        """
        prompt_pieces = self.get_prompt_request_piece_by_memory_labels(memory_labels=memory_labels)
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
        request_pieces = self._get_prompt_pieces_with_conversation_id(conversation_id=conversation_id)
        return group_conversation_request_pieces_by_sequence(request_pieces=request_pieces)

    @abc.abstractmethod
    def get_prompt_request_pieces_by_id(self, *, prompt_ids: list[str]) -> Sequence[PromptRequestPiece]:
        """
        Retrieves a list of PromptRequestPiece objects that have the specified prompt ids.

        Args:
            prompt_ids (list[int]): The prompt IDs to match.

        Returns:
            Sequence[PromptRequestPiece]: A list of PromptRequestPiece with the specified conversation ID.
        """

    def get_prompt_request_piece_by_orchestrator_id(self, *, orchestrator_id: str) -> list[PromptRequestPiece]:
        """
        Retrieves a list of PromptRequestPiece objects that have the specified orchestrator ID.

        Args:
            orchestrator_id (str): The orchestrator ID to match.

        Returns:
            list[PromptRequestPiece]: A list of PromptRequestPiece with the specified conversation ID.
        """

        prompt_pieces = self._get_prompt_pieces_by_orchestrator(orchestrator_id=orchestrator_id)
        return sorted(prompt_pieces, key=lambda x: (x.conversation_id, x.timestamp))

    def get_prompt_request_piece_by_memory_labels(
        self, *, memory_labels: dict[str, str] = {}
    ) -> list[PromptRequestPiece]:
        """
        Retrieves a list of PromptRequestPiece objects that have the specified memory labels.

        Args:
            memory_labels (dict[str, str]): A free-form dictionary for tagging prompts with custom labels.
            These labels can be used to track all prompts sent as part of an operation, score prompts based on
            the operation ID (op_id), and tag each prompt with the relevant Responsible AI (RAI) harm category.
            Users can define any key-value pairs according to their needs. Defaults to an empty dictionary.

        Returns:
            list[PromptRequestPiece]: A list of PromptRequestPiece with the specified memory labels.
        """

    def get_prompt_ids_by_orchestrator(self, *, orchestrator_id: str) -> list[str]:
        prompt_pieces = self._get_prompt_pieces_by_orchestrator(orchestrator_id=orchestrator_id)

        prompt_ids = []
        for piece in prompt_pieces:
            prompt_ids.append(str(piece.id))

        return prompt_ids

    def duplicate_conversation_for_new_orchestrator(self, *, new_orchestrator_id: str, conversation_id: str) -> str:
        """
        Duplicates a conversation from one orchestrator to another.

        This can be useful when an attack strategy requires branching out from a particular point in the conversation.
        One cannot continue both branches with the same orchestrator and conversation IDs since that would corrupt
        the memory. Instead, one needs to duplicate the conversation and continue with the new orchestrator ID.

        Args:
            new_orchestrator_id (str): The new orchestrator ID to assign to the duplicated conversations.
            conversation_id (str): The conversation ID with existing conversations.
        Returns:
            The uuid for the new conversation.
        """
        new_conversation_id = str(uuid.uuid4())
        # Deep copy objects to prevent any mutability-related issues that could arise due to in-memory databases.
        prompt_pieces = copy.deepcopy(self._get_prompt_pieces_with_conversation_id(conversation_id=conversation_id))
        for piece in prompt_pieces:
            # Assign duplicated piece a new ID, but note that the `original_prompt_id` remains the same.
            piece.id = uuid.uuid4()
            if piece.orchestrator_identifier["id"] == new_orchestrator_id:
                raise ValueError("The new orchestrator ID must be different from the existing orchestrator ID.")
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
        prompt_pieces = copy.deepcopy(self._get_prompt_pieces_with_conversation_id(conversation_id=conversation_id))

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

    def export_conversation_by_orchestrator_id(
        self, *, orchestrator_id: str, file_path: Path = None, export_type: str = "json"
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
        entries_to_update = self.query_entries(
            PromptMemoryEntry, conditions=PromptMemoryEntry.conversation_id == conversation_id
        )
        # Check if there are entries to update
        if not entries_to_update:
            logger.info(f"No entries found with conversation_id {conversation_id} to update.")
            return False

        # Use the utility function to update the entries
        success = self.update_entries(entries=entries_to_update, update_fields=update_fields)

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

    def update_prompt_metadata_by_conversation_id(self, *, conversation_id: str, prompt_metadata: str) -> bool:
        """
        Updates the metadata of prompt entries in memory for a given conversation ID.

        Args:
            conversation_id (str): The conversation ID of the entries to be updated.
            metadata (str): New metadata.

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

    def get_seed_prompts(
        self,
        *,
        value: Optional[str] = None,
        dataset_name: Optional[str] = None,
        harm_categories: Optional[list[str]] = None,
        added_by: Optional[str] = None,
        authors: Optional[list[str]] = None,
        groups: Optional[list[str]] = None,
        source: Optional[str] = None,
        parameters: Optional[list[str]] = None,
    ) -> list[SeedPrompt]:
        """
        Retrieves a list of seed prompts based on the specified filters.

        Args:
            value (str): The value to match by substring. If None, all values are returned.
            dataset_name (str): The dataset name to match. If None, all dataset names are considered.
            harm_categories (list[str]): A list of harm categories to filter by. If None,
            all harm categories are considered.
                Specifying multiple harm categories returns only prompts that are marked with all harm categories.
            added_by (str): The user who added the prompts.
            authors (list[str]): A list of authors to filter by.
                Note that this filters by substring, so a query for "Adam Jones" may not return results if the record
                is "A. Jones", "Jones, Adam", etc. If None, all authors are considered.
            groups (list[str]): A list of groups to filter by. If None, all groups are considered.
            source (str): The source to filter by. If None, all sources are considered.
            parameters (list[str]): A list of parameters to filter by. Specifying parameters effectively returns
                prompt templates instead of prompts.
                If None, only prompts without parameters are returned.

        Returns:
            list[SeedPrompt]: A list of prompts matching the criteria.
        """
        conditions = []

        # Apply filters for non-list fields
        if value:
            conditions.append(SeedPromptEntry.value.contains(value))
        if dataset_name:
            conditions.append(SeedPromptEntry.dataset_name == dataset_name)
        if added_by:
            conditions.append(SeedPromptEntry.added_by == added_by)
        if source:
            conditions.append(SeedPromptEntry.source == source)

        self._add_list_conditions(SeedPromptEntry.harm_categories, harm_categories, conditions)
        self._add_list_conditions(SeedPromptEntry.authors, authors, conditions)
        self._add_list_conditions(SeedPromptEntry.groups, groups, conditions)
        self._add_list_conditions(SeedPromptEntry.parameters, parameters, conditions)

        try:
            memory_entries = self.query_entries(
                SeedPromptEntry,
                conditions=and_(*conditions) if conditions else None,
            )  # type: ignore
            return [memory_entry.get_seed_prompt() for memory_entry in memory_entries]
        except Exception as e:
            logger.exception(f"Failed to retrieve prompts with dataset name {dataset_name} with error {e}")
            return []

    def _add_list_conditions(self, field: InstrumentedAttribute, values: Optional[list[str]], conditions: list) -> None:
        if values:
            for value in values:
                conditions.append(field.contains(value))

    def add_seed_prompts_to_memory(self, *, prompts: list[SeedPrompt], added_by: Optional[str] = None) -> None:
        """
        Inserts a list of prompts into the memory storage.

        Args:
            prompts (list[SeedPrompt]): A list of prompts to insert.
            added_by (str): The user who added the prompts.
        """
        entries: list[SeedPromptEntry] = []
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
            entries.append(SeedPromptEntry(entry=prompt))

        self.insert_entries(entries=entries)

    def get_seed_prompt_dataset_names(self) -> list[str]:
        """
        Returns a list of all seed prompt dataset names in the memory storage.
        """
        try:
            entries = self.query_entries(
                SeedPromptEntry.dataset_name,
                conditions=and_(SeedPromptEntry.dataset_name is not None, SeedPromptEntry.dataset_name != ""),
                distinct=True,
            )  # type: ignore
            # return value is list of tuples with a single entry (the dataset name)
            return [entry[0] for entry in entries]
        except Exception as e:
            logger.exception(f"Failed to retrieve dataset names with error {e}")
            return []

    def add_seed_prompt_groups_to_memory(
        self, *, prompt_groups: list[SeedPromptGroup], added_by: Optional[str] = None
    ) -> None:
        """
        Inserts a list of seed prompt groups into the memory storage.

        Args:
            prompt_groups (list[SeedPromptGroup]): A list of prompt groups to insert.
            added_by (str): The user who added the prompt groups.

        Raises:
            ValueError: If a prompt group does not have at least one prompt.
            ValueError: If prompt group IDs are inconsistent within the same prompt group.
        """
        if not prompt_groups:
            raise ValueError("At least one prompt group must be provided.")
        # Validates the prompt group IDs and sets them if possible before leveraging
        # the add_seed_prompts_to_memory method.
        all_prompts = []
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
        self.add_seed_prompts_to_memory(prompts=all_prompts, added_by=added_by)

    def get_seed_prompt_groups(
        self,
        *,
        dataset_name: Optional[str] = None,
        data_types: Optional[list[str]] = None,
        harm_categories: Optional[list[str]] = None,
        added_by: Optional[str] = None,
        authors: Optional[list[str]] = None,
        groups: Optional[list[str]] = None,
        source: Optional[str] = None,
    ) -> list[SeedPromptGroup]:
        """Retrieves groups of seed prompts based on the provided filtering criteria._summary_

        Args:
            dataset_name (Optional[str], Optional): Name of the dataset to filter seed prompts.
            data_types (Optional[Sequence[str]], Optional): List of data types to filter seed prompts by
            (e.g., text, image_path).
            harm_categories (Optional[Sequence[str]], Optional): List of harm categories to filter seed prompts by.
            added_by (Optional[str], Optional): The user who added the seed prompt groups to filter by.
            authors (Optional[Sequence[str]], Optional): List of authors to filter seed prompt groups by.
            groups (Optional[Sequence[str]], Optional): List of groups to filter seed prompt groups by.
            source (Optional[str], Optional): The source from which the seed prompts originated.

        Returns:
            list[SeedPromptGroup]: A list of `SeedPromptGroup` objects that match the filtering criteria.
        """
        conditions = []

        # Apply basic filters if provided
        if dataset_name:
            conditions.append(SeedPromptEntry.dataset_name == dataset_name)
        if added_by:
            conditions.append(SeedPromptEntry.added_by == added_by)
        if source:
            conditions.append(SeedPromptEntry.source == source)
        if data_types:
            data_type_conditions = SeedPromptEntry.data_type.in_(data_types)
            conditions.append(data_type_conditions)

        # Add conditions for lists: harm categories, authors, and groups
        self._add_list_conditions(SeedPromptEntry.harm_categories, harm_categories, conditions)
        self._add_list_conditions(SeedPromptEntry.authors, authors, conditions)
        self._add_list_conditions(SeedPromptEntry.groups, groups, conditions)

        # Query DB for matching entries
        memory_entries = self.query_entries(
            SeedPromptEntry,
            conditions=and_(*conditions) if conditions else None,
        )  # type: ignore

        # Extract seed prompts and group them by prompt group ID
        seed_prompts = [memory_entry.get_seed_prompt() for memory_entry in memory_entries]
        seed_prompt_groups = SeedPromptDataset.group_seed_prompts_by_prompt_group_id(seed_prompts)
        return seed_prompt_groups

    def export_all_conversations_with_scores(self, *, file_path: Optional[Path] = None, export_type: str = "json"):
        """
        Exports all conversations with scores to a specified file.
        Args:
            file_path (str): The path to the file where the data will be exported.
            If not provided, a default path using RESULTS_PATH will be constructed.
            export_type (str): The format of the export. Defaults to "json".
        """
        all_prompt_pieces = self.get_all_prompt_pieces()

        # Group pieces by original prompt ID
        grouped_pieces: defaultdict[Union[uuid.UUID, str], list[str]] = defaultdict(list)
        for piece in all_prompt_pieces:
            grouped_pieces[piece.original_prompt_id].append(piece.converted_value)

        all_scores = self.get_scores_by_prompt_ids(
            prompt_request_response_ids=[str(key) for key in list(grouped_pieces.keys())]
        )

        # Combine data for export
        combined_data = [
            {
                "prompt_request_response_id": str(score.prompt_request_response_id),
                "conversation": grouped_pieces[score.prompt_request_response_id],
                "score_value": score.score_value,
            }
            for score in all_scores
        ]

        # If file_path is not provided, construct a default using the exporter's results_path
        if not file_path:
            file_name = f"conversations_and_scores.{export_type}"
            file_path = RESULTS_PATH / file_name

        self.exporter.export_data(combined_data, file_path=file_path, export_type=export_type)
