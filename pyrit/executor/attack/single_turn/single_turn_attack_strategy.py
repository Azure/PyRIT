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
from pyrit.models import AttackResult, Message, SeedGroup
from pyrit.prompt_target import PromptTarget


@dataclass
class SingleTurnAttackContext(AttackContext):
    """Context for single-turn attacks."""

    # Unique identifier of the main conversation between the attacker and model
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Optional message to send to the objective target (overrides objective if provided)
    message: Optional[Message] = None

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

    def __init__(
        self,
        *,
        objective_target: PromptTarget,
        context_type: type[SingleTurnAttackContext],
        logger: logging.Logger = logger,
    ):
        """
        Define a base class for single-turn attack strategies.

        Args:
            objective_target (PromptTarget): The target system to attack.
            context_type (type[SingleTurnAttackContext]): The type of context this strategy will use
            logger (logging.Logger): Logger instance for logging events and messages
        """
        super().__init__(objective_target=objective_target, context_type=context_type, logger=logger)

    @overload
    async def execute_async(
        self,
        *,
        objective: str,
        prepended_conversation: Optional[List[Message]] = None,
        message: Optional[Message] = None,
        memory_labels: Optional[dict[str, str]] = None,
        **kwargs,
    ) -> AttackResult: ...

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
        Execute the single-turn attack strategy asynchronously with the provided parameters.

        Args:
            objective (str): The objective of the attack.
            prepended_conversation (Optional[List[Message]]): Conversation to prepend.
            message (Optional[Message]): Optional message to send to the objective target.
            memory_labels (Optional[Dict[str, str]]): Memory labels for the attack context.
            **kwargs: Additional parameters for the attack.

        Returns:
            AttackResult: The result of the attack execution.
        """
        # Validate parameters before creating context
        message = get_kwarg_param(kwargs=kwargs, param_name="message", expected_type=Message, required=False)
        objective = get_kwarg_param(kwargs=kwargs, param_name="objective", expected_type=str, required=False)
        system_prompt = get_kwarg_param(kwargs=kwargs, param_name="system_prompt", expected_type=str, required=False)
        
        return await super().execute_async(
            **kwargs, message=message, system_prompt=system_prompt, objective=objective
        )
