# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
AttackRun class for executing single attack configurations against datasets.

This module provides the AttackRun class that represents an atomic test combining
an attack, a dataset, and execution parameters. Multiple AttackRuns can be grouped
together into larger test scenarios for comprehensive security testing.
"""

import logging
from typing import Any, Dict, List, Optional

from pyrit.executor.attack import AttackExecutor, AttackStrategy
from pyrit.models import AttackResult, Message

logger = logging.getLogger(__name__)


class AttackRun:
    """
    Represents a single atomic attack test combining an attack strategy and dataset.

    An AttackRun is an executable unit that executes a configured attack against
    all objectives in a dataset. Multiple AttackRuns can be grouped together into
    larger test scenarios for comprehensive security testing and evaluation.

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
    """

    def __init__(
        self,
        *,
        attack: AttackStrategy,
        objectives: List[str],
        prepended_conversation: Optional[List[Message]] = None,
        memory_labels: Optional[Dict[str, str]] = None,
        **attack_execute_params: Any,
    ) -> None:
        """
        Initialize an attack run with an attack strategy and dataset parameters.

        Args:
            attack (AttackStrategy): The configured attack strategy to execute.
            objectives (List[str]): List of attack objectives to test against.
            prepended_conversation (Optional[List[Message]]): Optional
                conversation history to prepend to each attack execution.
            memory_labels (Optional[Dict[str, str]]): Additional labels to apply to prompts.
                These labels help track and categorize the attack run in memory.
            **attack_execute_params (Any): Additional parameters to pass to the attack
                execution method (e.g., custom_prompts, batch_size).

        Raises:
            ValueError: If objectives list is empty.
        """
        if not objectives:
            raise ValueError("objectives list cannot be empty")

        self._attack = attack
        self._objectives = objectives
        self._prepended_conversation = prepended_conversation
        self._memory_labels = memory_labels or {}
        self._attack_execute_params = attack_execute_params

        logger.info(
            f"Initialized attack run with {len(self._objectives)} objectives "
            f"and attack type: {type(attack).__name__}"
        )

    async def run_async(self, *, max_concurrency: int = 1) -> List[AttackResult]:
        """
        Execute the attack run against all objectives in the dataset.

        This method uses AttackExecutor to run the configured attack against
        all objectives from the dataset.

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

        # Build the parameters for execute_multi_objective_attack_async
        execute_params = {
            "objectives": self._objectives,
            "prepended_conversation": self._prepended_conversation,
            "memory_labels": merged_memory_labels,
            **self._attack_execute_params,
        }

        logger.info(
            f"Starting attack run execution with {len(self._objectives)} objectives "
            f"and max_concurrency={max_concurrency}"
        )

        try:
            # Execute the attack using the executor
            results = await executor.execute_multi_objective_attack_async(
                attack=self._attack,
                **execute_params,
            )

            logger.info(f"Attack run execution completed successfully with {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Attack run execution failed: {str(e)}")
            raise ValueError(f"Failed to execute attack run: {str(e)}") from e
