# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Attack results mixin for MemoryInterface containing attack result-related operations."""

import logging
from typing import TYPE_CHECKING, Optional, Sequence

from sqlalchemy import and_
from sqlalchemy.sql.elements import ColumnElement

from pyrit.memory.memory_interface.protocol import MemoryInterfaceProtocol
from pyrit.memory.memory_models import AttackResultEntry
from pyrit.models.attack_result import AttackResult

logger = logging.getLogger(__name__)

# Use protocol inheritance only during type checking to avoid metaclass conflicts.
# The protocol uses typing._ProtocolMeta which conflicts with the Singleton metaclass
# used by concrete memory classes. This conditional inheritance provides full type
# checking and IDE support while avoiding runtime metaclass conflicts.
if TYPE_CHECKING:
    _MixinBase = MemoryInterfaceProtocol
else:
    _MixinBase = object


class MemoryAttackResultsMixin(_MixinBase):
    """Mixin providing attack result-related methods for memory management."""

    def add_attack_results_to_memory(self, *, attack_results: Sequence[AttackResult]) -> None:
        """
        Inserts a list of attack results into the memory storage.
        The database model automatically calculates objective_sha256 for consistency.
        """
        self._insert_entries(entries=[AttackResultEntry(entry=attack_result) for attack_result in attack_results])

    def get_attack_results(
        self,
        *,
        attack_result_ids: Optional[Sequence[str]] = None,
        conversation_id: Optional[str] = None,
        objective: Optional[str] = None,
        objective_sha256: Optional[Sequence[str]] = None,
        outcome: Optional[str] = None,
    ) -> Sequence[AttackResult]:
        """
        Retrieves a list of AttackResult objects based on the specified filters.

        Args:
            attack_result_ids (Optional[Sequence[str]], optional): A list of attack result IDs. Defaults to None.
            conversation_id (Optional[str], optional): The conversation ID to filter by. Defaults to None.
            objective (Optional[str], optional): The objective to filter by (substring match). Defaults to None.
            objective_sha256 (Optional[Sequence[str]], optional): A list of objective SHA256 hashes to filter by.
                Defaults to None.
            outcome (Optional[str], optional): The outcome to filter by (success, failure, undetermined).
                Defaults to None.

        Returns:
            Sequence[AttackResult]: A list of AttackResult objects that match the specified filters.
        """
        conditions: list[ColumnElement[bool]] = []

        if attack_result_ids is not None:
            if len(attack_result_ids) == 0:
                # Empty list means no results
                return []
            conditions.append(AttackResultEntry.id.in_(attack_result_ids))
        if conversation_id:
            conditions.append(AttackResultEntry.conversation_id == conversation_id)
        if objective:
            conditions.append(AttackResultEntry.objective.contains(objective))
        if objective_sha256:
            conditions.append(AttackResultEntry.objective_sha256.in_(objective_sha256))
        if outcome:
            conditions.append(AttackResultEntry.outcome == outcome)

        try:
            entries: Sequence[AttackResultEntry] = self._query_entries(
                AttackResultEntry, conditions=and_(*conditions) if conditions else None
            )
            return [entry.get_attack_result() for entry in entries]
        except Exception as e:
            logger.exception(f"Failed to retrieve attack results with error {e}")
            return []
