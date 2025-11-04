# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, TypeVar, overload

from pyrit.common.logger import logger
from pyrit.common.utils import get_kwarg_param
from pyrit.executor.attack.core.attack_config import AttackScoringConfig
from pyrit.executor.core import (
    Strategy,
    StrategyContext,
    StrategyEvent,
    StrategyEventData,
    StrategyEventHandler,
)
from pyrit.memory.central_memory import CentralMemory
from pyrit.models import (
    AttackOutcome,
    AttackResult,
    ConversationReference,
    Message,
)
from pyrit.prompt_target import PromptTarget

AttackStrategyContextT = TypeVar("AttackStrategyContextT", bound="AttackContext")
AttackStrategyResultT = TypeVar("AttackStrategyResultT", bound="AttackResult")


@dataclass
class AttackContext(StrategyContext, ABC):
    """Base class for all attack contexts"""

    # Natural-language description of what the attack tries to achieve
    objective: str

    # Start time of the attack execution
    start_time: float = 0.0

    # Additional labels that can be applied to the prompts throughout the attack
    memory_labels: Dict[str, str] = field(default_factory=dict)

    # Conversations relevant while the attack is running
    related_conversations: set[ConversationReference] = field(default_factory=set)

    # Conversation that is automatically prepended to the target model
    prepended_conversation: list[Message] = field(default_factory=list)


class _DefaultAttackStrategyEventHandler(StrategyEventHandler[AttackStrategyContextT, AttackStrategyResultT]):
    """
    Default event handler for attack strategies.
    Handles events during the execution of an attack strategy.
    """

    def __init__(self, logger: logging.Logger = logger):
        """
        Initialize the default event handler with a logger.

        Args:
            logger (logging.Logger): Logger instance for logging events.
        """
        self._logger = logger
        self._events = {
            StrategyEvent.ON_PRE_EXECUTE: self._on_pre_execute,
            StrategyEvent.ON_POST_EXECUTE: self._on_post_execute,
        }
        self._memory = CentralMemory.get_memory_instance()

    async def on_event(self, event_data: StrategyEventData[AttackStrategyContextT, AttackStrategyResultT]) -> None:
        """
        Handle an event during the attack strategy execution.

        Args:
            event_data (StrategyEventData[AttackStrategyContextT, AttackStrategyResultT]): The event data containing
                context and result.
        """
        if event_data.event in self._events:
            handler = self._events[event_data.event]
            await handler(event_data)
        else:
            await self._on(event_data)

    async def _on(self, event_data: StrategyEventData[AttackStrategyContextT, AttackStrategyResultT]) -> None:
        """
        Handle specific events during the attack strategy execution.

        Args:
            event_data (StrategyEventData[AttackStrategyContextT, AttackStrategyResultT]): The event data containing
                context and result.
        """
        self._logger.debug(f"Attack is in '{event_data.event.value}' stage for {self.__class__.__name__}")

    async def _on_pre_execute(
        self, event_data: StrategyEventData[AttackStrategyContextT, AttackStrategyResultT]
    ) -> None:
        """
        Handle pre-execution logic before the attack strategy runs.

        Args:
            event_data (StrategyEventData[AttackStrategyContextT, AttackStrategyResultT]): The event data containing
                context and result.
        """
        if not event_data.context:
            raise ValueError("Attack context is None. Cannot proceed with execution.")

        # Initialize start time for execution
        event_data.context.start_time = time.perf_counter()

        # Log the start of the attack
        self._logger.info(f"Starting attack: {event_data.context.objective}")

    async def _on_post_execute(
        self, event_data: StrategyEventData[AttackStrategyContextT, AttackStrategyResultT]
    ) -> None:
        """
        Handle post-execution logic after the attack strategy has run.

        Args:
            result (AttackResult): The result of the attack strategy execution.
        """

        if not event_data.result:
            raise ValueError("Attack result is None. Cannot log or record the outcome.")

        end_time = time.perf_counter()
        execution_time_ms = int((end_time - event_data.context.start_time) * 1000)
        event_data.result.execution_time_ms = execution_time_ms

        self._logger.debug(f"Attack execution completed in {execution_time_ms}ms")

        self._log_attack_outcome(event_data.result)
        self._memory.add_attack_results_to_memory(attack_results=[event_data.result])

    def _log_attack_outcome(self, result: AttackResult) -> None:
        """
        Log the outcome of the attack.

        Args:
            result (AttackResult): The result of the attack containing outcome and reason.
        """
        attack_name = self.__class__.__name__
        reason = f"Reason: {result.outcome_reason or 'Not specified'}"

        if result.outcome == AttackOutcome.SUCCESS:
            message = f"{attack_name} achieved the objective. {reason}"
        elif result.outcome == AttackOutcome.UNDETERMINED:
            message = f"{attack_name} outcome is undetermined. {reason}"
        else:
            message = f"{attack_name} did not achieve the objective. {reason}"

        self._logger.info(message)


class AttackStrategy(Strategy[AttackStrategyContextT, AttackStrategyResultT], ABC):
    """
    Abstract base class for attack strategies.
    Defines the interface for executing attacks and handling results.
    """

    def __init__(
        self,
        *,
        objective_target: PromptTarget,
        context_type: type[AttackStrategyContextT],
        logger: logging.Logger = logger
    ):
        """
        Initialize the attack strategy with a specific context type and logger.

        Args:
            objective_target (PromptTarget): The target system to attack.
            context_type (type[AttackStrategyContextT]): The type of context this strategy operates on.
            logger (logging.Logger): Logger instance for logging events.
        """
        super().__init__(
            context_type=context_type,
            event_handler=_DefaultAttackStrategyEventHandler[AttackStrategyContextT, AttackStrategyResultT](
                logger=logger
            ),
            logger=logger,
        )
        self._objective_target = objective_target

    def get_objective_target(self) -> PromptTarget:
        """
        Get the objective target for this attack strategy.

        Returns:
            PromptTarget: The target system being attacked.
        """
        return self._objective_target

    def get_attack_scoring_config(self) -> Optional[AttackScoringConfig]:
        """
        Get the attack scoring configuration used by this strategy.

        Returns:
            Optional[AttackScoringConfig]: The scoring configuration, or None if not applicable.

        Note:
            Subclasses that use scoring should override this method to return their
            scoring configuration. The default implementation returns None.
        """
        return None

    @overload
    async def execute_async(
        self,
        *,
        objective: str,
        prepended_conversation: Optional[list[Message]] = None,
        memory_labels: Optional[dict[str, str]] = None,
        **kwargs,
    ) -> AttackStrategyResultT:
        """
        Execute the attack strategy asynchronously with the provided parameters.
        Args:
            objective (str): The objective of the attack.
            prepended_conversation (Optional[List[Message]]): Conversation to prepend.
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
        objective = get_kwarg_param(kwargs=kwargs, param_name="objective", expected_type=str)

        memory_labels = get_kwarg_param(kwargs=kwargs, param_name="memory_labels", expected_type=dict, required=False)

        if "prepended_conversation" in kwargs:
            # Attacks such as TAP do not use prepended conversations
            prepended_conversation = get_kwarg_param(
                kwargs=kwargs, param_name="prepended_conversation", expected_type=list, required=False
            )
            kwargs["prepended_conversation"] = prepended_conversation

        return await super().execute_async(**kwargs, objective=objective, memory_labels=memory_labels)
