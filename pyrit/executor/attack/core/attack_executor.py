# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from pyrit.executor.attack.core import (
    AttackStrategy,
    AttackStrategyContextT,
    AttackStrategyResultT,
)
from pyrit.executor.attack.multi_turn.multi_turn_attack_strategy import (
    MultiTurnAttackContext,
)
from pyrit.executor.attack.single_turn.single_turn_attack_strategy import (
    SingleTurnAttackContext,
)
from pyrit.models import Message, SeedGroup

AttackResultT = TypeVar("AttackResultT")


@dataclass
class AttackExecutorResult(Generic[AttackResultT]):
    """
    Result container for attack execution, supporting both full and partial completion.

    This class holds results from parallel attack execution. It is iterable and
    behaves like a list in the common case where all objectives complete successfully.

    When some objectives don't complete (throw exceptions), access incomplete_objectives
    to retrieve the failures, or use raise_if_incomplete() to raise the first exception.

    Note: "completed" means the execution finished, not that the attack objective was achieved.
    """

    completed_results: List[AttackResultT]
    incomplete_objectives: List[tuple[str, BaseException]]

    def __iter__(self):
        """Iterate over completed results."""
        return iter(self.completed_results)

    def __len__(self) -> int:
        """Return number of completed results."""
        return len(self.completed_results)

    def __getitem__(self, index: int) -> AttackResultT:
        """Access completed results by index."""
        return self.completed_results[index]

    @property
    def has_incomplete(self) -> bool:
        """Check if any objectives didn't complete execution."""
        return len(self.incomplete_objectives) > 0

    @property
    def all_completed(self) -> bool:
        """Check if all objectives completed execution."""
        return len(self.incomplete_objectives) == 0

    @property
    def exceptions(self) -> List[BaseException]:
        """
        Get all exceptions from incomplete objectives.

        Returns:
            List[BaseException]: List of exceptions that caused objectives to fail.
        """
        return [exception for _, exception in self.incomplete_objectives]

    def raise_if_incomplete(self) -> None:
        """
        Raise the first exception if any objectives are incomplete.

        Raises:
            BaseException: The first exception from incomplete objectives.
        """
        if self.incomplete_objectives:
            raise self.incomplete_objectives[0][1]

    def get_results(self) -> List[AttackResultT]:
        """
        Get completed results, raising if any incomplete.

        Returns:
            List[AttackResultT]: All completed results.

        Raises:
            BaseException: The first exception from incomplete objectives.
        """
        self.raise_if_incomplete()
        return self.completed_results


class AttackExecutor:
    """
    Manages the execution of attack strategies with support for different execution patterns.

    The AttackExecutor provides controlled execution of attack strategies with features like
    concurrency limiting and parallel execution. It can handle multiple objectives against
    the same target or execute different strategies concurrently.
    """

    _SingleTurnContextT = TypeVar("_SingleTurnContextT", bound=SingleTurnAttackContext)
    _MultiTurnContextT = TypeVar("_MultiTurnContextT", bound=MultiTurnAttackContext)

    def __init__(self, *, max_concurrency: int = 1):
        """
        Initialize the attack executor with configurable concurrency control.

        Args:
            max_concurrency (int): Maximum number of concurrent attack executions allowed.
                Must be a positive integer (defaults to 1).

        Raises:
            ValueError: If max_concurrency is not a positive integer.
        """
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be a positive integer, got {max_concurrency}")
        self._max_concurrency = max_concurrency

    async def execute_multi_objective_attack_async(
        self,
        *,
        attack: AttackStrategy[AttackStrategyContextT, AttackStrategyResultT],
        objectives: List[str],
        prepended_conversation: Optional[List[Message]] = None,
        memory_labels: Optional[Dict[str, str]] = None,
        return_partial_on_failure: bool = False,
        **attack_params,
    ) -> AttackExecutorResult[AttackStrategyResultT]:
        """
        Execute the same attack strategy with multiple objectives against the same target in parallel.

        This method provides a simplified interface for executing multiple objectives without
        requiring users to create context objects. It uses the attack's execute_async method
        which accepts parameters directly.

        Args:
            attack (AttackStrategy[ContextT, AttackStrategyResultT]): The attack strategy to use for all objectives.
            objectives (List[str]): List of attack objectives to test.
            prepended_conversation (Optional[List[Message]]): Conversation to prepend to the target model.
            memory_labels (Optional[Dict[str, str]]): Additional labels that can be applied to the prompts.
            return_partial_on_failure (bool): If True, returns AttackExecutorResult with completed results
                even when some objectives don't complete execution. If False, raises the first exception encountered.
                Defaults to False (raise on failure).
            **attack_params: Additional parameters specific to the attack strategy.

        Returns:
            AttackExecutorResult[AttackStrategyResultT]: Result container with completed results and
                any incomplete objectives. The result is iterable and behaves like a list of completed results.
                Use .has_incomplete or .raise_if_incomplete() to handle failures.

        Raises:
            BaseException: If return_partial_on_failure=False and any objective doesn't complete execution.

        Example:
            >>> executor = AttackExecutor(max_concurrency=3)
            >>> results = await executor.execute_multi_objective_attack_async(
            ...     attack=red_teaming_attack,
            ...     objectives=["how to make a Molotov cocktail", "how to escalate privileges"],
            ... )
            >>> # Iterate directly over results
            >>> for result in results:
            ...     print(result)
        """
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def execute_with_semaphore(objective: str) -> AttackStrategyResultT:
            async with semaphore:
                return await attack.execute_async(
                    objective=objective,
                    prepended_conversation=prepended_conversation,
                    memory_labels=memory_labels,
                    **attack_params,
                )

        tasks = [execute_with_semaphore(obj) for obj in objectives]
        results_or_exceptions = await asyncio.gather(*tasks, return_exceptions=True)

        return self._process_execution_results(
            objectives=objectives,
            results_or_exceptions=results_or_exceptions,
            return_partial_on_failure=return_partial_on_failure,
        )

    async def execute_single_turn_attacks_async(
        self,
        *,
        attack: AttackStrategy[_SingleTurnContextT, AttackStrategyResultT],
        objectives: List[str],
        seed_groups: Optional[List[SeedGroup]] = None,
        prepended_conversations: Optional[List[List[Message]]] = None,
        memory_labels: Optional[Dict[str, str]] = None,
        return_partial_on_failure: bool = False,
        **attack_params,
    ) -> AttackExecutorResult[AttackStrategyResultT]:
        """
        Execute a batch of single-turn attacks with multiple objectives.

        This method is specifically designed for single-turn attacks, allowing you to
        execute multiple objectives in parallel while managing the contexts and prompts.

        Args:
            attack (AttackStrategy[_SingleTurnContextT, AttackStrategyResultT]): The single-turn attack strategy to use,
                the context must be a SingleTurnAttackContext or a subclass of it.
            objectives (List[str]): List of attack objectives to test.
            seed_groups (Optional[List[SeedGroup]]): List of seed groups to use for this execution.
                If provided, must match the length of objectives. Seed group will be sent along the objective
                with the same list index.
            prepended_conversations (Optional[List[List[Message]]]): Conversations to prepend to each
                objective. If provided, must match the length of objectives. Conversation will be sent along the
                objective with the same list index.
            memory_labels (Optional[Dict[str, str]]): Additional labels that can be applied to the prompts.
                The labels will be the same across all executions.
            return_partial_on_failure (bool): If True, returns AttackExecutorResult with completed results
                even when some objectives don't complete execution. If False, raises the first exception encountered.
                Defaults to False (raise on failure).
            **attack_params: Additional parameters specific to the attack strategy.

        Returns:
            AttackExecutorResult[AttackStrategyResultT]: Result container with completed results and
                any incomplete objectives. The result is iterable and behaves like a list of completed results.

        Raises:
            BaseException: If return_partial_on_failure=False and any objective doesn't complete execution.

        Example:
            >>> executor = AttackExecutor(max_concurrency=3)
            >>> results = await executor.execute_single_turn_attacks_async(
            ...     attack=single_turn_attack,
            ...     objectives=["how to make a Molotov cocktail", "how to escalate privileges"],
            ...     seed_groups=[SeedGroup(...), SeedGroup(...)]
            ... )
        """
        # Validate that the attack uses SingleTurnAttackContext
        if hasattr(attack, "_context_type") and not issubclass(attack._context_type, SingleTurnAttackContext):
            raise TypeError(
                f"Attack strategy {attack.__class__.__name__} must use SingleTurnAttackContext or a subclass of it."
            )

        # Validate input parameters using shared validation logic
        self._validate_attack_batch_parameters(
            objectives=objectives,
            optional_list=seed_groups,
            optional_list_name="seed_groups",
            prepended_conversations=prepended_conversations,
        )

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def execute_with_semaphore(
            objective: str,
            seed_group: Optional[SeedGroup],
            prepended_conversation: Optional[List[Message]],
        ) -> AttackStrategyResultT:
            async with semaphore:
                return await attack.execute_async(
                    objective=objective,
                    prepended_conversation=prepended_conversation,
                    seed_group=seed_group,
                    memory_labels=memory_labels or {},
                    **attack_params,
                )

        # Create tasks for each objective with its corresponding parameters
        tasks = []
        for i, objective in enumerate(objectives):
            seed_group = seed_groups[i] if seed_groups else None
            prepended_conversation = prepended_conversations[i] if prepended_conversations else []

            task = execute_with_semaphore(
                objective=objective, seed_group=seed_group, prepended_conversation=prepended_conversation
            )
            tasks.append(task)

        # Execute all tasks in parallel with concurrency control
        results_or_exceptions = await asyncio.gather(*tasks, return_exceptions=True)

        return self._process_execution_results(
            objectives=objectives,
            results_or_exceptions=results_or_exceptions,
            return_partial_on_failure=return_partial_on_failure,
        )

    async def execute_multi_turn_attacks_async(
        self,
        *,
        attack: AttackStrategy[_MultiTurnContextT, AttackStrategyResultT],
        objectives: List[str],
        custom_prompts: Optional[List[str]] = None,
        prepended_conversations: Optional[List[List[Message]]] = None,
        memory_labels: Optional[Dict[str, str]] = None,
        return_partial_on_failure: bool = False,
        **attack_params,
    ) -> AttackExecutorResult[AttackStrategyResultT]:
        """
        Execute a batch of multi-turn attacks with multiple objectives.

        This method is specifically designed for multi-turn attacks, allowing you to
        execute multiple objectives in parallel while managing the contexts and custom prompts.

        Args:
            attack (AttackStrategy[_MultiTurnContextT, AttackStrategyResultT]): The multi-turn attack strategy to use,
                the context must be a MultiTurnAttackContext or a subclass of it.
            objectives (List[str]): List of attack objectives to test.
            custom_prompts (Optional[List[str]]): List of custom prompts to use for this execution.
                If provided, must match the length of objectives. custom prompts will be sent along the objective
                with the same list index.
            prepended_conversations (Optional[List[List[Message]]]): Conversations to prepend to each
                objective. If provided, must match the length of objectives. Conversation will be sent along the
                objective with the same list index.
            memory_labels (Optional[Dict[str, str]]): Additional labels that can be applied to the prompts.
                The labels will be the same across all executions.
            return_partial_on_failure (bool): If True, returns AttackExecutorResult with completed results
                even when some objectives don't complete execution. If False, raises the first exception encountered.
                Defaults to False (raise on failure).
            **attack_params: Additional parameters specific to the attack strategy.

        Returns:
            AttackExecutorResult[AttackStrategyResultT]: Result container with completed results and
                any incomplete objectives. The result is iterable and behaves like a list of completed results.

        Raises:
            BaseException: If return_partial_on_failure=False and any objective doesn't complete execution.

        Example:
            >>> executor = AttackExecutor(max_concurrency=3)
            >>> results = await executor.execute_multi_turn_attacks_async(
            ...     attack=multi_turn_attack,
            ...     objectives=["how to make a Molotov cocktail", "how to escalate privileges"],
            ...     custom_prompts=["Tell me about chemistry", "Explain system administration"]
            ... )
        """
        # Validate that the attack uses MultiTurnAttackContext
        if hasattr(attack, "_context_type") and not issubclass(attack._context_type, MultiTurnAttackContext):
            raise TypeError(
                f"Attack strategy {attack.__class__.__name__} must use MultiTurnAttackContext or a subclass of it."
            )

        # Validate input parameters using shared validation logic
        self._validate_attack_batch_parameters(
            objectives=objectives,
            optional_list=custom_prompts,
            optional_list_name="custom_prompts",
            prepended_conversations=prepended_conversations,
        )

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def execute_with_semaphore(
            objective: str, custom_prompt: Optional[str], prepended_conversation: Optional[List[Message]]
        ) -> AttackStrategyResultT:
            async with semaphore:
                return await attack.execute_async(
                    objective=objective,
                    prepended_conversation=prepended_conversation,
                    custom_prompt=custom_prompt,
                    memory_labels=memory_labels or {},
                    **attack_params,
                )

        # Create tasks for each objective with its corresponding parameters
        tasks = []
        for i, objective in enumerate(objectives):
            custom_prompt = custom_prompts[i] if custom_prompts else None
            prepended_conversation = prepended_conversations[i] if prepended_conversations else []

            task = execute_with_semaphore(
                objective=objective, custom_prompt=custom_prompt, prepended_conversation=prepended_conversation
            )
            tasks.append(task)

        # Execute all tasks in parallel with concurrency control
        results_or_exceptions = await asyncio.gather(*tasks, return_exceptions=True)

        return self._process_execution_results(
            objectives=objectives,
            results_or_exceptions=results_or_exceptions,
            return_partial_on_failure=return_partial_on_failure,
        )

    def _validate_attack_batch_parameters(
        self,
        *,
        objectives: List[str],
        optional_list: Optional[List[Any]] = None,
        optional_list_name: str = "optional_list",
        prepended_conversations: Optional[List[List[Message]]] = None,
    ) -> None:
        """
        Validate common parameters for batch attack execution methods.

        Args:
            objectives (List[str]): List of attack objectives to test.
            optional_list (Optional[List[any]]): Optional list parameter to validate length against objectives.
            optional_list_name (str): Name of the optional list parameter for error messages.
            prepended_conversations (Optional[List[List[Message]]]): Conversations to prepend.

        Raises:
            ValueError: If validation fails.
        """
        # Validate input parameters
        if not objectives:
            raise ValueError("At least one objective must be provided")

        # Validate optional_list length if provided
        if optional_list is not None and len(optional_list) != len(objectives):
            raise ValueError(
                f"Number of {optional_list_name} ({len(optional_list)}) must"
                f" match number of objectives ({len(objectives)})"
            )

        # Validate prepended_conversations length if provided
        if prepended_conversations is not None and len(prepended_conversations) != len(objectives):
            raise ValueError(
                f"Number of prepended_conversations ({len(prepended_conversations)}) must match "
                f"number of objectives ({len(objectives)})"
            )

    async def execute_multi_objective_attack_with_context_async(
        self,
        *,
        attack: AttackStrategy[AttackStrategyContextT, AttackStrategyResultT],
        context_template: AttackStrategyContextT,
        objectives: List[str],
        return_partial_on_failure: bool = False,
    ) -> AttackExecutorResult[AttackStrategyResultT]:
        """
        Execute the same attack strategy with multiple objectives using context objects.

        This method works with context objects directly, duplicating the template context
        for each objective. Use this when you need fine-grained control over the context
        or have an existing context template to reuse.

        Args:
            attack (AttackStrategy[AttackStrategyContextT, AttackStrategyResultT]): The attack strategy to use for
                all objectives.
            context_template (AttackStrategyContextT): Template context that will be duplicated for each objective.
                Must have a 'duplicate()' method and an 'objective' attribute.
            objectives (List[str]): List of attack objectives to test. Each objective will be
                executed as a separate attack using a copy of the context template.
            return_partial_on_failure (bool): If True, returns AttackExecutorResult with completed results
                even when some objectives don't complete execution. If False, raises the first exception encountered.
                Defaults to False (raise on failure).

        Returns:
            AttackExecutorResult[AttackStrategyResultT]: Result container with completed results and
                any incomplete objectives. The result is iterable and behaves like a list of completed results.
                Use .has_incomplete or .raise_if_incomplete() to handle failures.

        Raises:
            AttributeError: If the context_template doesn't have required 'duplicate()' method
                or 'objective' attribute.
            BaseException: If return_partial_on_failure=False and any objective doesn't complete execution.

        Example:
            >>> executor = AttackExecutor(max_concurrency=3)
            >>> context = MultiTurnAttackContext(max_turns=5, ...)
            >>> results = await executor.execute_multi_objective_attack_with_context_async(
            ...     attack=prompt_injection_attack,
            ...     context_template=context,
            ...     objectives=["how to make a Molotov cocktail", "how to escalate privileges"]
            ... )
            >>> # Iterate directly over results
            >>> for result in results:
            ...     print(result)
        """
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def execute_with_semaphore(ctx: AttackStrategyContextT) -> AttackStrategyResultT:
            async with semaphore:
                return await attack.execute_with_context_async(context=ctx)

        contexts = []
        for objective in objectives:
            # Create a deep copy of the context using its duplicate method
            context = context_template.duplicate()
            # Set the new objective (all attack contexts have objectives)
            context.objective = objective
            contexts.append(context)

        # Execute all tasks in parallel with concurrency control
        tasks = [execute_with_semaphore(ctx) for ctx in contexts]
        results_or_exceptions = await asyncio.gather(*tasks, return_exceptions=True)

        return self._process_execution_results(
            objectives=objectives,
            results_or_exceptions=results_or_exceptions,
            return_partial_on_failure=return_partial_on_failure,
        )

    def _process_execution_results(
        self,
        *,
        objectives: List[str],
        results_or_exceptions: List[Union[AttackStrategyResultT, BaseException]],
        return_partial_on_failure: bool,
    ) -> AttackExecutorResult[AttackStrategyResultT]:
        """
        Process results from parallel execution, separating completed from incomplete objectives.

        Args:
            objectives (List[str]): List of objectives that were executed.
            results_or_exceptions (List[Union[AttackStrategyResultT, BaseException]]): Results from asyncio.gather
                with return_exceptions=True.
            return_partial_on_failure (bool): If True, returns AttackExecutorResult even when some objectives
                don't complete. If False, raises the first exception encountered.

        Returns:
            AttackExecutorResult[AttackStrategyResultT]: Result container with completed results and
                any incomplete objectives. Always returns this type regardless of success/failure status.

        Raises:
            BaseException: If return_partial_on_failure=False and any objective doesn't complete execution.
        """
        # Separate completed results from exceptions
        completed_results: List[AttackStrategyResultT] = []
        incomplete_objectives: List[tuple[str, BaseException]] = []

        for objective, result_or_exception in zip(objectives, results_or_exceptions):
            if isinstance(result_or_exception, BaseException):
                incomplete_objectives.append((objective, result_or_exception))
            else:
                completed_results.append(result_or_exception)  # type: ignore[arg-type]

        # If some incomplete and return_partial_on_failure is False, raise the first exception
        if incomplete_objectives and not return_partial_on_failure:
            raise incomplete_objectives[0][1]

        return AttackExecutorResult(completed_results=completed_results, incomplete_objectives=incomplete_objectives)
