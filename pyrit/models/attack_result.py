# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, TypeVar

from pyrit.executor.core.strategy import StrategyContext
from pyrit.models.conversation_reference import ConversationReference, ConversationType
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.score import Score
from pyrit.models.strategy_result import StrategyResult

AttackResultT = TypeVar("AttackResultT", bound="AttackResult")


class AttackOutcome(Enum):
    """
    Enum representing the possible outcomes of an attack.
    """

    # The attack was successful in achieving its objective
    SUCCESS = "success"

    # The attack failed to achieve its objective
    FAILURE = "failure"

    # The outcome of the attack is unknown or could not be determined
    UNDETERMINED = "undetermined"


@dataclass
class AttackResult(StrategyResult):
    """Base class for all attack results"""

    # Identity
    # Unique identifier of the conversation that produced this result
    conversation_id: str

    # Natural-language description of the attacker's objective
    objective: str

    # Identifier of the attack (e.g., name, module)
    attack_identifier: dict[str, str]

    # Evidence
    # Model response generated in the final turn of the attack
    last_response: Optional[PromptRequestPiece] = None

    # Score assigned to the final response by a scorer component
    last_score: Optional[Score] = None

    # Metrics
    # Total number of turns that were executed
    executed_turns: int = 0

    # Total execution time of the attack in milliseconds
    execution_time_ms: int = 0

    # Outcome
    # The outcome of the attack, indicating success, failure, or undetermined
    outcome: AttackOutcome = AttackOutcome.UNDETERMINED

    # Optional reason for the outcome, providing additional context
    outcome_reason: Optional[str] = None

    # Flexible conversation refs (nothing unused)
    related_conversations: set[ConversationReference] = field(default_factory=set)

    # Arbitrary metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_conversations_by_type(self, conversation_type: ConversationType):
        """
        Return all related conversations of the requested type.

        Args:
            conversation_type (ConversationType): The type of conversation to filter by.

        Returns:
            list: A list of related conversations matching the specified type.
        """
        return [ref for ref in self.related_conversations if ref.conversation_type == conversation_type]

    def __str__(self):
        return f"AttackResult: {self.conversation_id}: {self.outcome.value}: " f"{self.objective[:50]}..."

@dataclass
class IntermediateAttackResult(AttackResult):
    """
    Subclass for AttackResult to indicate that the result is intermediate
    and that the attack should be invoked repeatedly with the same context.
    
    Iterative attacks return this result to indicate that the attack is not yet complete;
    it is expected that attacks using this class will manage their own lifecycle guarantees
    using the AttackStrategy methods.
    
    The inner attribute contains the actual AttackResult produced so far. It should not be
    another instance of IntermediateAttackResult, because we want to avoid recursion.
    """
    context: Optional[StrategyContext] = None
    
    # Note that we subclass from AttackResult to inherit all its fields;
    # this instance will likely be another IntermediateAttackResult,
    # but we use AttackResult to keep the implementation ergonomic.
        
    @property
    def final(self) -> bool:
        """
        Whether this result is final (i.e., not intermediate).
        """
        return self.context is None
    