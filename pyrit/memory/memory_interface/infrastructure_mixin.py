# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Infrastructure mixin for MemoryInterface containing core database operations and lifecycle management."""

import abc
import atexit
import logging
import weakref
from typing import MutableSequence, Optional, Sequence, TypeVar, Union

from sqlalchemy import Engine
from sqlalchemy.orm.attributes import InstrumentedAttribute

from pyrit.memory.memory_embedding import (
    MemoryEmbedding,
    default_memory_embedding_factory,
)
from pyrit.memory.memory_exporter import MemoryExporter
from pyrit.memory.memory_models import Base, EmbeddingDataEntry
from pyrit.models import StorageIO

logger = logging.getLogger(__name__)
Model = TypeVar("Model")


class MemoryInfrastructureMixin(abc.ABC):
    """Mixin providing core infrastructure methods for memory management."""

    memory_embedding: Optional[MemoryEmbedding] = None
    results_storage_io: Optional[StorageIO] = None
    results_path: Optional[str] = None
    engine: Optional[Engine] = None
    exporter: Optional[MemoryExporter] = None

    def __init__(self, embedding_model=None):
        """Initialize the MemoryInterface.

        Args:
            embedding_model: If set, this includes embeddings in the memory entries
                which are extremely useful for comparing chat messages and similarities,
                but also includes overhead.
        """
        self.memory_embedding = embedding_model
        # Initialize the MemoryExporter instance
        self.exporter = MemoryExporter()
        self._init_storage_io()

        # Ensure cleanup at process exit
        self.cleanup()

    # ========================================
    # Abstract Methods (must be implemented by concrete classes)
    # ========================================

    @abc.abstractmethod
    def _init_storage_io(self):
        """Initialize the storage IO handler results_storage_io."""

    @abc.abstractmethod
    def _get_prompt_pieces_memory_label_conditions(self, *, memory_labels: dict[str, str]) -> list:
        """Returns a list of conditions for filtering memory entries based on memory labels."""

    @abc.abstractmethod
    def _get_prompt_pieces_prompt_metadata_conditions(self, *, prompt_metadata: dict[str, Union[str, int]]) -> list:
        """Returns a list of conditions for filtering memory entries based on prompt metadata."""

    @abc.abstractmethod
    def _get_prompt_pieces_orchestrator_conditions(self, *, orchestrator_id: str):
        """Returns a condition to retrieve based on orchestrator ID."""

    @abc.abstractmethod
    def _get_seed_prompts_metadata_conditions(self, *, metadata: dict[str, Union[str, int]]):
        """Returns a list of conditions for filtering seed prompt entries based on prompt metadata."""

    @abc.abstractmethod
    def _query_entries(
        self, Model, *, conditions: Optional = None, distinct: bool = False, join_scores: bool = False  # type: ignore
    ) -> MutableSequence[Model]:  # type: ignore
        """Fetches data from the specified table model with optional conditions."""

    @abc.abstractmethod
    def _insert_entry(self, entry: Base) -> None:  # type: ignore
        """Inserts an entry into the Table."""

    @abc.abstractmethod
    def _insert_entries(self, *, entries: Sequence[Base]) -> None:  # type: ignore
        """Inserts multiple entries into the database."""

    @abc.abstractmethod
    def _update_entries(self, *, entries: MutableSequence[Base], update_fields: dict) -> bool:  # type: ignore
        """Updates the given entries with the specified field values."""

    @abc.abstractmethod
    def _add_embeddings_to_memory(self, *, embedding_data: Sequence[EmbeddingDataEntry]) -> None:
        """Inserts embedding data into memory storage."""

    @abc.abstractmethod
    def get_all_embeddings(self) -> Sequence[EmbeddingDataEntry]:
        """Loads all EmbeddingData from the memory storage handler."""

    @abc.abstractmethod
    def dispose_engine(self):
        """Dispose the engine and clean up resources."""

    # ========================================
    # Embedding Management
    # ========================================

    def enable_embedding(self, embedding_model=None):
        """Enable embedding functionality with optional model specification."""
        self.memory_embedding = default_memory_embedding_factory(embedding_model=embedding_model)

    def disable_embedding(self):
        """Disable embedding functionality."""
        self.memory_embedding = None

    # ========================================
    # Lifecycle Management
    # ========================================

    def cleanup(self):
        """Ensure cleanup on process exit."""
        # Ensure cleanup at process exit
        atexit.register(self.dispose_engine)

        # Ensure cleanup happens even if the object is garbage collected before process exits
        weakref.finalize(self, self.dispose_engine)

    # ========================================
    # Utility Methods
    # ========================================

    def _add_list_conditions(
        self, field: InstrumentedAttribute, conditions: list, values: Optional[Sequence[str]] = None
    ) -> None:
        """Helper method to add list-based conditions to query filters."""
        if values:
            for value in values:
                conditions.append(field.contains(value))

    def print_schema(self):
        """Prints the schema of all tables in the database."""
        from sqlalchemy import MetaData

        if not self.engine:
            print("No engine available to print schema")
            return

        metadata = MetaData()
        metadata.reflect(bind=self.engine)

        for table_name in metadata.tables:
            table = metadata.tables[table_name]
            print(f"Schema for {table_name}:")
            for column in table.columns:
                print(f"  Column {column.name} ({column.type})")
