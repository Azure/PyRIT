# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import uuid
from abc import ABC
from dataclasses import dataclass, field
from typing import List, Optional, TypeVar, overload

from pyrit.common.logger import logger
from pyrit.common.utils import get_kwarg_param
from pyrit.executor.attack.core import (
    AttackContext,
    AttackStrategy,
    AttackStrategyResultT,
)
from pyrit.models import (
    PromptRequestResponse,
    Score,
)

MultiTurnAttackStrategyContextT = TypeVar("MultiTurnAttackStrategyContextT", bound="MultiTurnAttackContext")


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

    # Model response produced in the latest turn
    last_response: Optional[PromptRequestResponse] = None

    # Score assigned to the latest response by a scorer component
    last_score: Optional[Score] = None

    # Optional custom prompt that overrides the default one for the next turn
    custom_prompt: Optional[str] = None


class MultiTurnAttackStrategy(AttackStrategy[MultiTurnAttackStrategyContextT, AttackStrategyResultT], ABC):
    """
    Strategy for executing single-turn attacks.
    This strategy is designed to handle attacks that consist of a single turn
    of interaction with the target model.
    """

    def __init__(self, *, context_type: type[MultiTurnAttackStrategyContextT], logger: logging.Logger = logger):
        """
        The base class for multi-turn attack strategies.

        Args:
            context_type (type[MultiTurnAttackContext]): The type of context this strategy will use
            logger (logging.Logger): Logger instance for logging events and messages
        """
        super().__init__(context_type=context_type, logger=logger)

    @overload
    async def execute_async(
        self,
        *,
        objective: str,
        prepended_conversation: Optional[List[PromptRequestResponse]] = None,
        custom_prompt: Optional[str] = None,
        memory_labels: Optional[dict[str, str]] = None,
        **kwargs,
    ) -> AttackStrategyResultT:
        """
        Execute the multi-turn attack strategy asynchronously with the provided parameters.

        Args:
            objective (str): The objective of the attack.
            prepended_conversation (Optional[List[PromptRequestResponse]]): Conversation to prepend.
            custom_prompt (Optional[str]): Custom prompt for the attack.
            memory_labels (Optional[Dict[str, str]]): Memory labels for the attack context.
            **kwargs: Additional parameters for the attack.

        Returns:
            AttackStrategyResultT: The result of the attack execution.
        """
        ...

    @overload
    async def execute_async(
        self,
        **kwargs,
    ) -> AttackStrategyResultT: ...

    async def execute_async(
        self,
        **kwargs,
    ) -> AttackStrategyResultT:
        """
        Execute the attack strategy asynchronously with the provided parameters.
        """

        # Validate parameters before creating context
        custom_prompt = get_kwarg_param(kwargs=kwargs, param_name="custom_prompt", expected_type=str, required=False)

        return await super().execute_async(**kwargs, custom_prompt=custom_prompt)
