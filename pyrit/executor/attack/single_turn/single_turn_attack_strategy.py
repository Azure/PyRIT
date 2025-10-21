# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import uuid
from abc import ABC
from dataclasses import dataclass, field
from typing import List, Optional, Union, overload

from pyrit.common.logger import logger
from pyrit.common.utils import get_kwarg_param
from pyrit.executor.attack.core import AttackContext, AttackStrategy
from pyrit.models import AttackResult, Message, SeedPromptGroup


@dataclass
class SingleTurnAttackContext(AttackContext):
    """Context for single-turn attacks"""

    # Unique identifier of the main conversation between the attacker and model
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Group of seed prompts from which single-turn prompts will be drawn
    seed_prompt_group: Optional[SeedPromptGroup] = None

    # System prompt for chat-based targets
    system_prompt: Optional[str] = None

    # Arbitrary metadata that downstream attacks or scorers may attach
    metadata: Optional[dict[str, Union[str, int]]] = None


class SingleTurnAttackStrategy(AttackStrategy[SingleTurnAttackContext, AttackResult], ABC):
    """
    Strategy for executing single-turn attacks.
    This strategy is designed to handle attacks that consist of a single turn
    of interaction with the target model.
    """

    def __init__(self, *, context_type: type[SingleTurnAttackContext], logger: logging.Logger = logger):
        """
        The base class for single-turn attack strategies.

        Args:
            context_type (type[SingleTurnAttackContext]): The type of context this strategy will use
            logger (logging.Logger): Logger instance for logging events and messages
        """
        super().__init__(context_type=context_type, logger=logger)

    @overload
    async def execute_async(
        self,
        *,
        objective: str,
        prepended_conversation: Optional[List[Message]] = None,
        seed_prompt_group: Optional[SeedPromptGroup] = None,
        memory_labels: Optional[dict[str, str]] = None,
        **kwargs,
    ) -> AttackResult:
        """
        Execute the single-turn attack strategy asynchronously with the provided parameters.

        Args:
            objective (str): The objective of the attack.
            prepended_conversation (Optional[List[Message]]): Conversation to prepend.
            seed_prompt_group (Optional[SeedPromptGroup]): Group of seed prompts for the attack.
            memory_labels (Optional[Dict[str, str]]): Memory labels for the attack context.
            **kwargs: Additional parameters for the attack.

        Returns:
            AttackResult: The result of the attack execution.
        """
        ...

    @overload
    async def execute_async(
        self,
        **kwargs,
    ) -> AttackResult: ...

    async def execute_async(
        self,
        **kwargs,
    ) -> AttackResult:
        """
        Execute the attack strategy asynchronously with the provided parameters.
        """

        # Validate parameters before creating context
        seed_prompt_group = get_kwarg_param(
            kwargs=kwargs, param_name="seed_prompt_group", expected_type=SeedPromptGroup, required=False
        )
        system_prompt = get_kwarg_param(kwargs=kwargs, param_name="system_prompt", expected_type=str, required=False)
        return await super().execute_async(**kwargs, seed_prompt_group=seed_prompt_group, system_prompt=system_prompt)
