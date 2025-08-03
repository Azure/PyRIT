# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Infrastructure mixin for MemoryInterface containing core database operations and lifecycle management."""

import atexit
import logging
import weakref
from typing import TYPE_CHECKING, Optional, Sequence, TypeVar

from sqlalchemy import Engine
from sqlalchemy.orm.attributes import InstrumentedAttribute

from pyrit.memory.memory_embedding import (
    MemoryEmbedding,
    default_memory_embedding_factory,
)
from pyrit.memory.memory_exporter import MemoryExporter
from pyrit.memory.memory_interface.protocol import MemoryInterfaceProtocol
from pyrit.models import StorageIO

logger = logging.getLogger(__name__)
Model = TypeVar("Model")

# Use protocol inheritance only during type checking to avoid metaclass conflicts.
# The protocol uses typing._ProtocolMeta which conflicts with the Singleton metaclass
# used by concrete memory classes. This conditional inheritance provides full type
# checking and IDE support while avoiding runtime metaclass conflicts.
if TYPE_CHECKING:
    _MixinBase = MemoryInterfaceProtocol
else:
    _MixinBase = object


class MemoryInfrastructureMixin(_MixinBase):
    """Mixin providing core infrastructure methods for memory management."""

    memory_embedding: Optional[MemoryEmbedding]
    results_storage_io: Optional[StorageIO]
    results_path: Optional[str]
    engine: Optional[Engine]
    exporter: Optional[MemoryExporter]

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
