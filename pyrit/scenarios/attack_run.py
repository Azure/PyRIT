# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
AttackRun class for executing single attack configurations against datasets.

This module provides the AttackRun class that represents an atomic test combining
an attack, a dataset, and execution parameters. Multiple AttackRuns can be grouped
together into larger test scenarios for comprehensive security testing.
"""

import logging
from typing import Any, Dict, List, Literal, Optional

from pyrit.executor.attack import AttackExecutor, AttackStrategy
from pyrit.executor.attack.multi_turn.multi_turn_attack_strategy import (
    MultiTurnAttackContext,
)
from pyrit.executor.attack.single_turn.single_turn_attack_strategy import (
    SingleTurnAttackContext,
)
from pyrit.models import AttackResult, Message, SeedPromptGroup

logger = logging.getLogger(__name__)


class AttackRun:
    """
    Represents a single atomic attack test combining an attack strategy and dataset.

    An AttackRun is an executable unit that executes a configured attack against
    all objectives in a dataset. Multiple AttackRuns can be grouped together into
    larger test scenarios for comprehensive security testing and evaluation.

    The AttackRun automatically detects whether the attack is single-turn or multi-turn
    and calls the appropriate executor method. For single-turn attacks, you can provide
    seed_prompt_groups. For multi-turn attacks, you can provide custom_prompts.

    Example:
        >>> from pyrit.scenarios import AttackRun
        >>> from pyrit.attacks import PromptAttack
        >>> from pyrit.prompt_target import OpenAIChatTarget
        >>>
        >>> target = OpenAIChatTarget()
        >>> attack = PromptAttack(objective_target=target)
        >>> objectives = ["how to make a bomb", "how to hack a system"]
        >>>
        >>> attack_run = AttackRun(
        ...     attack=attack,
        ...     objectives=objectives,
        ...     memory_labels={"test": "run1"}
        ... )
        >>> results = await attack_run.run_async(max_concurrency=5)
        >>>
        >>> # With prepended conversation
        >>> from pyrit.models import Message
        >>> conversation = [Message(...)]
        >>> attack_run = AttackRun(
        ...     attack=attack,
        ...     objectives=objectives,
        ...     prepended_conversation=conversation
        ... )
        >>> results = await attack_run.run_async(max_concurrency=5)
        >>>
        >>> # Single-turn attack with seed prompts
        >>> from pyrit.models import SeedPromptGroup
        >>> seed_prompts = [SeedPromptGroup(...), SeedPromptGroup(...)]
        >>> attack_run = AttackRun(
        ...     attack=single_turn_attack,
        ...     objectives=objectives,
        ...     seed_prompt_groups=seed_prompts
        ... )
        >>> results = await attack_run.run_async(max_concurrency=3)
        >>>
        >>> # Multi-turn attack with custom prompts
        >>> custom_prompts = ["Tell me about chemistry", "Explain system administration"]
        >>> attack_run = AttackRun(
        ...     attack=multi_turn_attack,
        ...     objectives=objectives,
        ...     custom_prompts=custom_prompts
        ... )
        >>> results = await attack_run.run_async(max_concurrency=3)
    """

    def __init__(
        self,
        *,
        attack: AttackStrategy,
        objectives: List[str],
        prepended_conversations: Optional[List[List[Message]]] = None,
        seed_prompt_groups: Optional[List[SeedPromptGroup]] = None,
        custom_prompts: Optional[List[str]] = None,
        memory_labels: Optional[Dict[str, str]] = None,
        **attack_execute_params: Any,
    ) -> None:
        """
        Initialize an attack run with an attack strategy and dataset parameters.

        Args:
            attack (AttackStrategy): The configured attack strategy to execute.
            objectives (List[str]): List of attack objectives to test against.
            prepended_conversations (Optional[List[List[Message]]]): Optional
                list of conversation histories to prepend to each attack execution. This will be
                used for all objectives.
            seed_prompt_groups (Optional[List[SeedPromptGroup]]): List of seed prompt groups
                for single-turn attacks. Must match the length of objectives if provided.
                Only valid for single-turn attacks.
            custom_prompts (Optional[List[str]]): List of custom prompts for multi-turn attacks.
                Must match the length of objectives if provided. Only valid for multi-turn attacks.
            memory_labels (Optional[Dict[str, str]]): Additional labels to apply to prompts.
                These labels help track and categorize the attack run in memory.
            **attack_execute_params (Any): Additional parameters to pass to the attack
                execution method (e.g., batch_size).

        Raises:
            ValueError: If objectives list is empty, or if parameters don't match requirements.
            TypeError: If seed_prompt_groups is provided for multi-turn attacks or
            custom_prompts (Optional[List[str]]): List of custom prompts for multi-turn attacks.
                Must match the length of objectives if provided. Only valid for multi-turn attacks.
            memory_labels (Optional[Dict[str, str]]): Additional labels to apply to prompts.
                These labels help track and categorize the attack run in memory.
            **attack_execute_params (Any): Additional parameters to pass to the attack
                execution method (e.g., batch_size).

        Raises:
            ValueError: If objectives list is empty, or if parameters don't match requirements.
            TypeError: If seed_prompt_groups is provided for multi-turn attacks or
                custom_prompts is provided for single-turn attacks.
        """
        if not objectives:
            raise ValueError("objectives list cannot be empty")

        # Store attack first so we can use it in helper methods
        self._attack = attack

        # Determine context type once during initialization
        self._context_type: Literal["single_turn", "multi_turn", "unknown"] = self._determine_context_type(attack)

        # Validate attack context type and parameters
        self._validate_parameters(
            seed_prompt_groups=seed_prompt_groups,
            custom_prompts=custom_prompts,
        )

        self._objectives = objectives
        self._prepended_conversation = prepended_conversation
        self._prepended_conversations = prepended_conversations
        self._seed_prompt_groups = seed_prompt_groups
        self._custom_prompts = custom_prompts
        self._memory_labels = memory_labels or {}
        self._attack_execute_params = attack_execute_params

        logger.info(
            f"Initialized attack run with {len(self._objectives)} objectives, "
            f"attack type: {type(attack).__name__}, context type: {self._context_type}"
        )

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

    def _validate_parameters(
        self,
        *,
        seed_prompt_groups: Optional[List[SeedPromptGroup]],
        custom_prompts: Optional[List[str]],
    ) -> None:
        """
        Validate that parameters match the attack context type.

        Args:
            seed_prompt_groups (Optional[List[SeedPromptGroup]]): Seed prompt groups parameter.
            custom_prompts (Optional[List[str]]): Custom prompts parameter.

        Raises:
            TypeError: If parameters don't match the attack context type.
        """
        # Validate seed_prompt_groups is only used with single-turn attacks
        if seed_prompt_groups is not None and self._context_type != "single_turn":
            raise TypeError(
                f"seed_prompt_groups can only be used with single-turn attacks. "
                f"Attack {self._attack.__class__.__name__} uses {self._context_type} context"
            )

        # Validate custom_prompts is only used with multi-turn attacks
        if custom_prompts is not None and self._context_type != "multi_turn":
            raise TypeError(
                f"custom_prompts can only be used with multi-turn attacks. "
                f"Attack {self._attack.__class__.__name__} uses {self._context_type} context"
            )

    async def run_async(self, *, max_concurrency: int = 1) -> List[AttackResult]:
        """
        Execute the attack run against all objectives in the dataset.

        This method uses AttackExecutor to run the configured attack against
        all objectives from the dataset. It automatically detects whether to use
        single-turn or multi-turn execution based on the attack's context type.

        Args:
            max_concurrency (int): Maximum number of concurrent attack executions.
                Defaults to 1 for sequential execution.

        Returns:
            List[AttackResult]: List of attack results, one for each objective.

        Raises:
            ValueError: If the attack execution fails.
        """
        # Create the executor with the specified concurrency
        executor = AttackExecutor(max_concurrency=max_concurrency)

        # Merge memory labels from initialization and execution parameters
        merged_memory_labels = {**self._memory_labels}

        # Determine prepended_conversations to use
        prepended_conversations = self._prepended_conversations
        if prepended_conversations is None and self._prepended_conversation is not None:
            # If single prepended_conversation provided, replicate it for all objectives
            prepended_conversations = [self._prepended_conversation] * len(self._objectives)

        logger.info(
            f"Starting attack run execution with {len(self._objectives)} objectives "
            f"and max_concurrency={max_concurrency}"
        )

        try:
            # Execute based on context type
            if self._context_type == "single_turn":
                results = await executor.execute_single_turn_attacks_async(
                    attack=self._attack,
                    objectives=self._objectives,
                    seed_prompt_groups=self._seed_prompt_groups,
                    prepended_conversations=prepended_conversations,
                    memory_labels=merged_memory_labels,
                )
            elif self._context_type == "multi_turn":
                results = await executor.execute_multi_turn_attacks_async(
                    attack=self._attack,
                    objectives=self._objectives,
                    custom_prompts=self._custom_prompts,
                    prepended_conversations=prepended_conversations,
                    memory_labels=merged_memory_labels,
                )
            else:
                # Fall back to generic execute_multi_objective_attack_async
                execute_params = {
                    "objectives": self._objectives,
                    "prepended_conversation": self._prepended_conversation,
                    "memory_labels": merged_memory_labels,
                    **self._attack_execute_params,
                }
                results = await executor.execute_multi_objective_attack_async(
                    attack=self._attack,
                    **execute_params,
                )

            logger.info(f"Attack run execution completed successfully with {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Attack run execution failed: {str(e)}")
            raise ValueError(f"Failed to execute attack run: {str(e)}") from e
