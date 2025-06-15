# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TypeVar, Union

from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.models.score import Score
from pyrit.models.seed_prompt import SeedPromptGroup

ContextT = TypeVar("ContextT", bound="AttackContext")


@dataclass
class AttackContext:
    """Base class for all attack contexts"""

    # Natural-language description of what the attack tries to achieve
    objective: str

    # Conversation that is automatically prepended to the target model
    prepended_conversation: List[PromptRequestResponse] = field(default_factory=list)

    # Key–value pairs stored in the model's memory for this single request
    memory_labels: Dict[str, str] = field(default_factory=dict)

    def duplicate(self: ContextT) -> ContextT:
        """
        Create a deep copy of the context to avoid concurrency issues.

        Returns:
            AttackContext: A deep copy of the context.
        """
        return deepcopy(self)


@dataclass
class ConversationSession:
    """Session for conversations"""

    # Unique identifier of the main conversation between the attacker and model
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Separate identifier used when the attack leverages an adversarial chat
    adversarial_chat_conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class MultiTurnAttackContext(AttackContext):
    """Context for multi-turn attacks"""

    # Object holding all conversation-level identifiers for this attack
    session: ConversationSession = field(default_factory=lambda: ConversationSession())

    # Counter of turns that have actually been executed so far
    executed_turns: int = 0

    # Maximum number of turns the attack will run before stopping
    max_turns: int = 10

    # Model response produced in the latest turn
    last_response: Optional[PromptRequestResponse] = None

    # Score assigned to the latest response by a scorer component
    last_score: Optional[Score] = None

    # Optional custom prompt that overrides the default one for the next turn
    custom_prompt: Optional[str] = None

    # Text that was refused by the target in the previous attempt (used for backtracking)
    refused_text: Optional[str] = None

    # Counter for number of backtracks performed during the attack
    backtrack_count: int = 0


@dataclass
class SingleTurnAttackContext(AttackContext):
    """Context for single-turn attacks"""

    # Unique identifier of the main conversation between the attacker and model
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Maximum number of attempts to retry the attack in case of failure
    # (e.g., if the target model refuses to respond)
    max_attempts_on_failure: int = 0

    # Group of seed prompts from which single-turn prompts will be drawn
    seed_prompt_group: Optional[SeedPromptGroup] = None

    # System prompt for chat-based targets
    system_prompt: Optional[str] = None

    # Arbitrary metadata that downstream orchestrators or scorers may attach
    metadata: Optional[dict[str, Union[str, int]]] = None
