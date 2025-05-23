# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Self, TypeVar, Union

from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.models.score import Score
from pyrit.models.seed_prompt import SeedPromptGroup
from pyrit.prompt_converter.prompt_converter import PromptConverter

ContextT = TypeVar("ContextT", bound="AttackContext")


@dataclass
class AttackContext:
    """Base class for all attack contexts"""

    def duplicate(self) -> Self:
        """
        Create a deep copy of the context to avoid concurrency issues

        Returns:
            AttackContext: A deep copy of the context
        """
        return deepcopy(self)


@dataclass
class ConversationSession:
    """Session for conversations"""

    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    adversarial_chat_conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class MultiTurnAttackContext(AttackContext):
    """Context for multi-turn attacks"""

    session: ConversationSession = field(default_factory=lambda: ConversationSession())
    objective: Optional[str] = None
    max_turns: int = 5
    achieved_objective: bool = False
    executed_turns: int = 0
    last_response: Optional[PromptRequestPiece] = None
    last_score: Optional[Score] = None
    custom_prompt: Optional[str] = None
    prompt_converters: List[PromptConverter] = field(default_factory=list)
    prepended_conversation: List[PromptRequestResponse] = field(default_factory=list)
    memory_labels: Optional[Dict[str, str]] = None


@dataclass
class SingleTurnAttackContext(AttackContext):
    """Context for single-turn attacks"""

    batch_size: int = 1
    num_retries_on_failure: int = 0
    objective: Optional[str] = None
    seed_prompt_group: Optional[SeedPromptGroup] = None
    prepended_conversation: List[PromptRequestResponse] = field(default_factory=list)
    memory_labels: Optional[Dict[str, str]] = None
    metadata: Optional[dict[str, Union[str, int]]] = None
