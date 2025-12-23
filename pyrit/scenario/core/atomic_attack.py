# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
AtomicAttack class for executing single attack configurations against datasets.

This module provides the AtomicAttack class that represents an atomic test combining
an attack, a dataset, and execution parameters. Multiple AtomicAttacks can be grouped
together into larger test scenarios for comprehensive security testing.

Eventually it's a good goal to unify attacks as much as we can. But there are
times when that may not be possible or make sense. So this class exists to
have a common interface for scenarios.
"""

import logging
from typing import Any, Dict, List, Literal, Optional

from pyrit.executor.attack import (
    AttackExecutor,
    AttackStrategy,
    MultiTurnAttackContext,
    SingleTurnAttackContext,
)
from pyrit.executor.attack.core.attack_executor import AttackExecutorResult
from pyrit.models import AttackResult, Message

logger = logging.getLogger(__name__)


class AtomicAttack:
    """
    Represents a single atomic attack test combining an attack strategy and dataset.

    An AtomicAttack is an executable unit that executes a configured attack against
    all objectives in a dataset. Multiple AtomicAttacks can be grouped together into
    larger test scenarios for comprehensive security testing and evaluation.

    The AtomicAttack automatically detects whether the attack is single-turn or multi-turn
    and calls the appropriate executor method. For single-turn attacks, you can provide
    seed_groups. For multi-turn attacks, you can provide custom_prompts.

    Example:
        >>> from pyrit.scenario import AtomicAttack
        >>> from pyrit.attacks import PromptAttack
        >>> from pyrit.prompt_target import OpenAIChatTarget
        >>>
        >>> target = OpenAIChatTarget()
        >>> attack = PromptAttack(objective_target=target)
        >>> objectives = ["how to make a bomb", "how to hack a system"]
        >>>
        >>> atomic_attack = AtomicAttack(
        ...     attack=attack,
        ...     objectives=objectives,
        ...     memory_labels={"test": "run1"}
        ... )
        >>> results = await atomic_attack.run_async(max_concurrency=5)
        >>>
        >>> # With prepended conversations
        >>> from pyrit.models import Message
        >>> conversation = [Message(...)]
        >>> atomic_attack = AtomicAttack(
        ...     attack=attack,
        ...     objectives=objectives,
        ...     prepended_conversations=[conversation]
        ... )
        >>> results = await atomic_attack.run_async(max_concurrency=5)
        >>>
        >>> # Single-turn attack with seeds
        >>> from pyrit.models import SeedGroup
        >>> seeds = [SeedGroup(...), SeedGroup(...)]
        >>> atomic_attack = AtomicAttack(
        ...     attack=single_turn_attack,
        ...     objectives=objectives,
        ...     seed_groups=seeds
        ... )
        >>> results = await atomic_attack.run_async(max_concurrency=3)
        >>>
        >>> # Multi-turn attack with custom prompts
        >>> custom_prompts = ["Tell me about chemistry", "Explain system administration"]
        >>> atomic_attack = AtomicAttack(
        ...     attack=multi_turn_attack,
        ...     objectives=objectives,
        ...     custom_prompts=custom_prompts
        ... )
        >>> results = await atomic_attack.run_async(max_concurrency=3)
    """

    def __init__(
        self,
        *,
        atomic_attack_name: str,
        attack: AttackStrategy,
        objectives: List[str],
        prepended_conversations: Optional[List[List[Message]]] = None,
        messages: Optional[List[Message]] = None,
        custom_prompts: Optional[List[str]] = None,
        memory_labels: Optional[Dict[str, str]] = None,
        **attack_execute_params: Any,
    ) -> None:
        """
        Initialize an atomic attack with an attack strategy and dataset parameters.

        Args:
            atomic_attack_name (str): Used to group an AtomicAttack with related attacks for a
                strategy.
            attack (AttackStrategy): The configured attack strategy to execute.
            objectives (List[str]): List of attack objectives to test against.
            prepended_conversations (Optional[List[List[Message]]]): Optional
                list of conversation histories to prepend to each attack execution. This will be
                used for all objectives.
            messages (Optional[List[Message]]): List of messages
                for single-turn attacks. Only valid for single-turn attacks.
            custom_prompts (Optional[List[str]]): List of custom prompts for multi-turn attacks.
                Only valid for multi-turn attacks.
            memory_labels (Optional[Dict[str, str]]): Additional labels to apply to prompts.
                These labels help track and categorize the atomic attack in memory.
            **attack_execute_params (Any): Additional parameters to pass to the attack
                execution method (e.g., batch_size).

        Raises:
            ValueError: If objectives list is empty.
        """
        self.atomic_attack_name = atomic_attack_name

        if not objectives:
            raise ValueError("objectives list cannot be empty")

        # Store attack first so we can use it in helper methods
        self._attack = attack

        # Determine context type once during initialization
        self._context_type: Literal["single_turn", "multi_turn", "unknown"] = self._determine_context_type(attack)

        self._objectives = objectives
        self._prepended_conversations = prepended_conversations
        self._messages = messages
        self._memory_labels = memory_labels or {}
        self._attack_execute_params = attack_execute_params

        logger.info(
            f"Initialized atomic attack with {len(self._objectives)} objectives, "
            f"attack type: {type(attack).__name__}, context type: {self._context_type}"
        )

    @property
    def objectives(self) -> List[str]:
        """
        Get a copy of the objectives list for this atomic attack.

        This property is read-only. To use different objectives, create a new AtomicAttack instance.

        Returns:
            List[str]: A copy of the objectives list.
        """
        return list(self._objectives)

    def _determine_context_type(self, attack: AttackStrategy) -> Literal["single_turn", "multi_turn", "unknown"]:
        """
        Determine the context type of the attack strategy.

        Args:
            attack (AttackStrategy): The attack strategy to check.

        Returns:
            Literal["single_turn", "multi_turn", "unknown"]: The context type of the attack.
        """
        if hasattr(attack, "_context_type"):
            if issubclass(attack._context_type, SingleTurnAttackContext):
                return "single_turn"
            elif issubclass(attack._context_type, MultiTurnAttackContext):
                return "multi_turn"
        return "unknown"

    async def run_async(
        self,
        *,
        max_concurrency: int = 1,
        return_partial_on_failure: bool = True,
        **attack_params,
    ) -> AttackExecutorResult[AttackResult]:
        """
        Execute the atomic attack against all objectives in the dataset.

        This method uses AttackExecutor to run the configured attack against
        all objectives from the dataset. It automatically detects whether to use
        single-turn or multi-turn execution based on the attack's context type.

        When return_partial_on_failure=True (default), this method will return
        an AttackExecutorResult containing both completed results and incomplete
        objectives (those that didn't finish execution due to exceptions). This allows
        scenarios to save progress and retry only the incomplete objectives.

        Note: "completed" means the execution finished, not that the attack objective
        was achieved. "incomplete" means execution didn't finish (threw an exception).

        Args:
            max_concurrency (int): Maximum number of concurrent attack executions.
                Defaults to 1 for sequential execution.
            return_partial_on_failure (bool): If True, returns partial results even when
                some objectives don't complete execution. If False, raises an exception on
                any execution failure. Defaults to True.
            **attack_params: Additional parameters to pass to the attack strategy.

        Returns:
            AttackExecutorResult[AttackResult]: Result containing completed attack results and
                incomplete objectives (those that didn't finish execution).

        Raises:
            ValueError: If the attack execution fails completely and return_partial_on_failure=False.
        """
        # Create the executor with the specified concurrency
        executor = AttackExecutor(max_concurrency=max_concurrency)

        # Merge memory labels from initialization and execution parameters
        merged_memory_labels = {**self._memory_labels}

        # Determine prepended_conversations to use
        prepended_conversations = self._prepended_conversations

        logger.info(
            f"Starting atomic attack execution with {len(self._objectives)} objectives "
            f"and max_concurrency={max_concurrency}"
        )

        try:
            # Execute based on context type with common parameters
            if self._context_type == "single_turn":
                results = await executor.execute_single_turn_attacks_async(
                    attack=self._attack,
                    objectives=self._objectives,
                    messages=self._messages,
                    prepended_conversations=prepended_conversations,
                    memory_labels=merged_memory_labels,
                    return_partial_on_failure=return_partial_on_failure,
                    **self._attack_execute_params,
                )
            elif self._context_type == "multi_turn":
                results = await executor.execute_multi_turn_attacks_async(
                    attack=self._attack,
                    objectives=self._objectives,
                    messages=self._messages,
                    prepended_conversations=prepended_conversations,
                    memory_labels=merged_memory_labels,
                    return_partial_on_failure=return_partial_on_failure,
                    **self._attack_execute_params,
                )
            else:
                # Fall back to generic execute_multi_objective_attack_async
                # Note: This method uses prepended_conversation (singular) instead of prepended_conversations
                results = await executor.execute_multi_objective_attack_async(
                    attack=self._attack,
                    objectives=self._objectives,
                    prepended_conversation=prepended_conversations[0] if prepended_conversations else None,
                    memory_labels=merged_memory_labels,
                    return_partial_on_failure=return_partial_on_failure,
                    **self._attack_execute_params,
                )

            # Log completion status
            if results.has_incomplete:
                logger.warning(
                    f"Atomic attack execution completed with {len(results.completed_results)} completed "
                    f"and {len(results.incomplete_objectives)} incomplete objectives"
                )
            else:
                logger.info(
                    f"Atomic attack execution completed successfully with {len(results.completed_results)} results"
                )

            return results

        except Exception as e:
            logger.error(f"Atomic attack execution failed: {str(e)}")
            raise ValueError(f"Failed to execute atomic attack: {str(e)}") from e
