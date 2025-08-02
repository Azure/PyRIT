# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Protocol defining the interface that mixins can depend on."""

import uuid
from datetime import datetime
from typing import Any, MutableSequence, Optional, Protocol, Sequence, Union

from sqlalchemy.orm.attributes import InstrumentedAttribute

from pyrit.memory.memory_embedding import MemoryEmbedding
from pyrit.memory.memory_exporter import MemoryExporter
from pyrit.memory.memory_models import Base, EmbeddingDataEntry
from pyrit.models import PromptRequestPiece


class MemoryInterfaceProtocol(Protocol):
    """Protocol defining the interface that memory mixins can depend on."""

    # Attributes that mixins may access
    memory_embedding: Optional[MemoryEmbedding]
    exporter: MemoryExporter

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
        """Retrieves a list of PromptRequestPiece objects based on the specified filters."""
        ...

    def _insert_entries(self, *, entries: Sequence[Base]) -> None:
        """Inserts multiple entries into the database."""
        ...

    def _query_entries(
        self, Model: type, *, conditions: Optional[Any] = None, distinct: bool = False, join_scores: bool = False
    ) -> MutableSequence[Any]:
        """Fetches data from the specified table model with optional conditions."""
        ...

    def _update_entries(self, *, entries: MutableSequence[Base], update_fields: dict[str, Any]) -> bool:
        """Updates the given entries with the specified field values."""
        ...

    def _add_embeddings_to_memory(self, *, embedding_data: Sequence[EmbeddingDataEntry]) -> None:
        """Inserts embedding data into memory storage."""
        ...

    def _get_prompt_pieces_orchestrator_conditions(self, *, orchestrator_id: str) -> Any:
        """Returns a condition to retrieve based on orchestrator ID."""
        ...

    def _get_prompt_pieces_memory_label_conditions(self, *, memory_labels: dict[str, str]) -> list[Any]:
        """Returns a list of conditions for filtering memory entries based on memory labels."""
        ...

    def _get_prompt_pieces_prompt_metadata_conditions(self, *, prompt_metadata: dict[str, Union[str, int]]) -> list[Any]:
        """Returns a list of conditions for filtering memory entries based on prompt metadata."""
        ...

    def _get_seed_prompts_metadata_conditions(self, *, metadata: dict[str, Union[str, int]]) -> Any:
        """Returns a list of conditions for filtering seed prompt entries based on prompt metadata."""
        ...

    def _add_list_conditions(
        self, field: InstrumentedAttribute, conditions: list[Any], values: Optional[Sequence[str]] = None
    ) -> None:
        """Helper method to add list-based conditions to query filters."""
        ...

    def _init_storage_io(self) -> None:
        """Initialize the storage IO handler results_storage_io."""
        ...

    def _insert_entry(self, entry: Base) -> None:
        """Inserts an entry into the Table."""
        ...

    def get_all_embeddings(self) -> Sequence[EmbeddingDataEntry]:
        """Loads all EmbeddingData from the memory storage handler."""
        ...

    def dispose_engine(self) -> None:
        """Dispose the engine and clean up resources."""
        ...

    def add_request_pieces_to_memory(self, *, request_pieces: Sequence[PromptRequestPiece]) -> None:
        """Inserts a list of prompt request pieces into the memory storage."""
        ...
