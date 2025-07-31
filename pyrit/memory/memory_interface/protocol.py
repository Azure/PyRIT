# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Protocol defining the interface that mixins can depend on."""

from typing import Protocol, Sequence, MutableSequence, Optional, Union
import uuid
from datetime import datetime

from pyrit.memory.memory_models import Base, ScoreEntry
from pyrit.models import PromptRequestPiece, Score


class MemoryInterfaceProtocol(Protocol):
    """Protocol defining the interface that memory mixins can depend on."""
    
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
        self, Model, *, conditions: Optional = None, distinct: bool = False, join_scores: bool = False
    ) -> MutableSequence:
        """Fetches data from the specified table model with optional conditions."""
        ...
