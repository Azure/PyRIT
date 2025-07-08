# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar

from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.score import Score

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
class AttackConversationIds:
    # List of conversation IDs to the objective target that were pruned from the attack
    pruned_conversation_ids: List[str] = field(default_factory=list)

    # List of conversation IDs to the adversarial chat that were used for the attack
    adversarial_chat_conversation_ids: List[str] = field(default_factory=list)

    # List of conversation IDs used to score the attack
    scored_conversation_ids: List[str] = field(default_factory=list)
    
    # List of conversation IDs used to convert the attack
    converter_conversation_ids: List[str] = field(default_factory=list)



@dataclass
class AttackResult:
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

    # Conversation IDs used to generate the attack
    attack_generation_conversation_ids: AttackConversationIds = field(default_factory=AttackConversationIds)

    # Additional information
    # Metadata can be included as key-value pairs to provide extra context
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"AttackResult: {self.conversation_id}: {self.outcome.value}: {self.objective[:50]}..."
    
