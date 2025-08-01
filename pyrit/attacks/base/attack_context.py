# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypeVar, Union

from pyrit.models.conversation_reference import ConversationReference
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.models.score import Score
from pyrit.models.seed_prompt import SeedPromptGroup

ContextT = TypeVar("ContextT", bound="AttackContext")


@dataclass
class AttackContext(ABC):
    """Base class for all attack contexts"""

    # Natural-language description of what the attack tries to achieve
    objective: str

    # Additional labels that can be applied to the prompts throughout the attack
    memory_labels: Dict[str, str] = field(default_factory=dict)

    # Conversations relevant while the attack is running
    related_conversations: set[ConversationReference] = field(default_factory=set)

    @classmethod
    @abstractmethod
    def create_from_params(
        cls: type[ContextT],
        *,
        attack_input: Any,
        prepended_conversation: List[PromptRequestResponse],
        memory_labels: Dict[str, str],
        **kwargs,
    ) -> ContextT:
        """
        Factory method to create context from standard parameters.

        All subclasses must implement this method to handle their specific fields.

        Args:
            objective: The objective of the attack.
            prepended_conversation: Conversation to prepend to the target model.
            memory_labels: Additional labels that can be applied to the prompts.
            **kwargs: Additional parameters specific to the context type.

        Returns:
            An instance of the context type.
        """
        pass

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

    # Conversation that is automatically prepended to the target model
    prepended_conversation: List[PromptRequestResponse] = field(default_factory=list)

    # Counter of turns that have actually been executed so far
    executed_turns: int = 0

    # Model response produced in the latest turn
    last_response: Optional[PromptRequestResponse] = None

    # Score assigned to the latest response by a scorer component
    last_score: Optional[Score] = None

    # Optional custom prompt that overrides the default one for the next turn
    custom_prompt: Optional[str] = None

    @classmethod
    def create_from_params(
        cls,
        *,
        attack_input: Any,
        prepended_conversation: List[PromptRequestResponse],
        memory_labels: Dict[str, str],
        **kwargs,
    ) -> "MultiTurnAttackContext":
        """Create MultiTurnAttackContext from parameters."""

        # For multi-turn attacks, attack_input is expected to be the objective string
        if not isinstance(attack_input, str):
            raise ValueError(
                "MultiTurnAttackContext expects attack_input to be a string (objective), "
                f"got {type(attack_input).__name__}"
            )

        objective = attack_input

        custom_prompt = kwargs.get("custom_prompt")

        # Validate custom_prompt if provided
        if custom_prompt is not None and not isinstance(custom_prompt, str):
            raise ValueError(f"custom_prompt must be a string, got {type(custom_prompt).__name__}")

        return cls(
            objective=objective,
            prepended_conversation=prepended_conversation,
            memory_labels=memory_labels,
            custom_prompt=custom_prompt,
        )


@dataclass
class SingleTurnAttackContext(AttackContext):
    """Context for single-turn attacks"""

    # Unique identifier of the main conversation between the attacker and model
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Conversation that is automatically prepended to the target model
    prepended_conversation: List[PromptRequestResponse] = field(default_factory=list)

    # Group of seed prompts from which single-turn prompts will be drawn
    seed_prompt_group: Optional[SeedPromptGroup] = None

    # System prompt for chat-based targets
    system_prompt: Optional[str] = None

    # Arbitrary metadata that downstream orchestrators or scorers may attach
    metadata: Optional[dict[str, Union[str, int]]] = None

    @classmethod
    def create_from_params(
        cls,
        *,
        attack_input: Any,
        prepended_conversation: List[PromptRequestResponse],
        memory_labels: Dict[str, str],
        **kwargs,
    ) -> "SingleTurnAttackContext":
        """Create SingleTurnAttackContext from parameters."""

        # For single-turn attacks, attack_input is expected to be the objective string
        if not isinstance(attack_input, str):
            raise ValueError(
                "SingleTurnAttackContext expects attack_input to be a string (objective), "
                f"got {type(attack_input).__name__}"
            )

        objective = attack_input

        # Extract and validate optional parameters
        seed_prompt_group = kwargs.get("seed_prompt_group")
        if seed_prompt_group is not None and not isinstance(seed_prompt_group, SeedPromptGroup):
            raise ValueError(f"seed_prompt_group must be a SeedPromptGroup, got {type(seed_prompt_group).__name__}")

        system_prompt = kwargs.get("system_prompt")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise ValueError(f"system_prompt must be a string, got {type(system_prompt).__name__}")

        return cls(
            objective=objective,
            prepended_conversation=prepended_conversation,
            memory_labels=memory_labels,
            seed_prompt_group=seed_prompt_group,
            system_prompt=system_prompt,
        )
