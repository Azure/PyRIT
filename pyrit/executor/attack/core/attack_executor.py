# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from typing import Any, Dict, List, Optional, TypeVar

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
from pyrit.models import PromptRequestResponse, SeedPromptGroup


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
        prepended_conversation: Optional[List[PromptRequestResponse]] = None,
        memory_labels: Optional[Dict[str, str]] = None,
        **attack_params,
    ) -> List[AttackStrategyResultT]:
        """
        Execute the same attack strategy with multiple objectives against the same target in parallel.

        This method provides a simplified interface for executing multiple objectives without
        requiring users to create context objects. It uses the attack's execute_async method
        which accepts parameters directly.

        Args:
            attack (AttackStrategy[ContextT, AttackStrategyResultT]): The attack strategy to use for all objectives.
            objectives (List[str]): List of attack objectives to test.
            prepended_conversation (Optional[List[PromptRequestResponse]]): Conversation to prepend to the target model.
            memory_labels (Optional[Dict[str, str]]): Additional labels that can be applied to the prompts.
            **attack_params: Additional parameters specific to the attack strategy.

        Returns:
            List[AttackStrategyResultT]: List of attack results in the same order as the objectives list.

        Example:
            >>> executor = AttackExecutor(max_concurrency=3)
            >>> results = await executor.execute_multi_objective_attack_async(
            ...     attack=red_teaming_attack,
            ...     objectives=["how to make a Molotov cocktail", "how to escalate privileges"],
            ... )
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
        return await asyncio.gather(*tasks)

    async def execute_single_turn_attacks_async(
        self,
        *,
        attack: AttackStrategy[_SingleTurnContextT, AttackStrategyResultT],
        objectives: List[str],
        seed_prompt_groups: Optional[List[SeedPromptGroup]] = None,
        prepended_conversations: Optional[List[List[PromptRequestResponse]]] = None,
        memory_labels: Optional[Dict[str, str]] = None,
    ) -> List[AttackStrategyResultT]:
        """
        Execute a batch of single-turn attacks with multiple objectives.

        This method is specifically designed for single-turn attacks, allowing you to
        execute multiple objectives in parallel while managing the contexts and prompts.

        Args:
            attack (AttackStrategy[_SingleTurnContextT, AttackStrategyResultT]): The single-turn attack strategy to use,
                the context must be a SingleTurnAttackContext or a subclass of it.
            objectives (List[str]): List of attack objectives to test.
            seed_prompt_groups (Optional[List[SeedPromptGroup]]): List of seed prompt groups to use for this execution.
                If provided, must match the length of objectives. Seed prompt group will be sent along the objective
                with the same list index.
            prepended_conversations (Optional[List[List[PromptRequestResponse]]]): Conversations to prepend to each
                objective. If provided, must match the length of objectives. Conversation will be sent along the
                objective with the same list index.
            memory_labels (Optional[Dict[str, str]]): Additional labels that can be applied to the prompts.
                The labels will be the same across all executions.
        Returns:
            List[AttackStrategyResultT]: List of attack results in the same order as the objectives list.

        Example:
            >>> executor = AttackExecutor(max_concurrency=3)
            >>> results = await executor.execute_single_turn_batch_async(
            ...     attack=single_turn_attack,
            ...     objectives=["how to make a Molotov cocktail", "how to escalate privileges"],
            ...     seed_prompts=[SeedPromptGroup(...), SeedPromptGroup(...)]
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
            optional_list=seed_prompt_groups,
            optional_list_name="seed_prompt_groups",
            prepended_conversations=prepended_conversations,
        )

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def execute_with_semaphore(
            objective: str,
            seed_prompt_group: Optional[SeedPromptGroup],
            prepended_conversation: Optional[List[PromptRequestResponse]],
        ) -> AttackStrategyResultT:
            async with semaphore:
                return await attack.execute_async(
                    objective=objective,
                    prepended_conversation=prepended_conversation,
                    seed_prompt_group=seed_prompt_group,
                    memory_labels=memory_labels or {},
                )

        # Create tasks for each objective with its corresponding parameters
        tasks = []
        for i, objective in enumerate(objectives):
            seed_prompt_group = seed_prompt_groups[i] if seed_prompt_groups else None
            if seed_prompt_group and seed_prompt_group.objective and seed_prompt_group.objective.value != objective:
                raise ValueError(
                    "Attack can only specify one objective per turn. Objective parameter '%s' and seed"
                    " prompt group objective '%s' are both defined",
                    objective,
                    seed_prompt_group.objective.value,
                )
            prepended_conversation = prepended_conversations[i] if prepended_conversations else []

            task = execute_with_semaphore(
                objective=objective, seed_prompt_group=seed_prompt_group, prepended_conversation=prepended_conversation
            )
            tasks.append(task)

        # Execute all tasks in parallel with concurrency control
        return await asyncio.gather(*tasks)

    async def execute_multi_turn_attacks_async(
        self,
        *,
        attack: AttackStrategy[_MultiTurnContextT, AttackStrategyResultT],
        objectives: List[str],
        custom_prompts: Optional[List[str]] = None,
        prepended_conversations: Optional[List[List[PromptRequestResponse]]] = None,
        memory_labels: Optional[Dict[str, str]] = None,
    ) -> List[AttackStrategyResultT]:
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
            prepended_conversations (Optional[List[List[PromptRequestResponse]]]): Conversations to prepend to each
                objective. If provided, must match the length of objectives. Conversation will be sent along the
                objective with the same list index.
            memory_labels (Optional[Dict[str, str]]): Additional labels that can be applied to the prompts.
                The labels will be the same across all executions.
        Returns:
            List[AttackStrategyResultT]: List of attack results in the same order as the objectives list.

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
            objective: str, custom_prompt: Optional[str], prepended_conversation: Optional[List[PromptRequestResponse]]
        ) -> AttackStrategyResultT:
            async with semaphore:
                return await attack.execute_async(
                    objective=objective,
                    prepended_conversation=prepended_conversation,
                    custom_prompt=custom_prompt,
                    memory_labels=memory_labels or {},
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
        return await asyncio.gather(*tasks)

    def _validate_attack_batch_parameters(
        self,
        *,
        objectives: List[str],
        optional_list: Optional[List[Any]] = None,
        optional_list_name: str = "optional_list",
        prepended_conversations: Optional[List[List[PromptRequestResponse]]] = None,
    ) -> None:
        """
        Validate common parameters for batch attack execution methods.

        Args:
            objectives (List[str]): List of attack objectives to test.
            optional_list (Optional[List[any]]): Optional list parameter to validate length against objectives.
            optional_list_name (str): Name of the optional list parameter for error messages.
            prepended_conversations (Optional[List[List[PromptRequestResponse]]]): Conversations to prepend.

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
    ) -> List[AttackStrategyResultT]:
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

        Returns:
            List[AttackStrategyResultT]: List of attack results in the same order as the objectives list.
                Each result corresponds to the execution of the strategy with the objective
                at the same index.

        Raises:
            AttributeError: If the context_template doesn't have required 'duplicate()' method
                or 'objective' attribute.
            Any exceptions raised during strategy execution will be propagated.

        Example:
            >>> executor = AttackExecutor(max_concurrency=3)
            >>> context = MultiTurnAttackContext(max_turns=5, ...)
            >>> results = await executor.execute_multi_objective_attack_with_context_async(
            ...     attack=prompt_injection_attack,
            ...     context_template=context,
            ...     objectives=["how to make a Molotov cocktail", "how to escalate privileges"]
            ... )
        """
        contexts = []

        for objective in objectives:
            # Create a deep copy of the context using its duplicate method
            context = context_template.duplicate()
            # Set the new objective (all attack contexts have objectives)
            context.objective = objective
            contexts.append(context)

        # Run strategies in parallel
        results = await self._execute_parallel_async(attack=attack, contexts=contexts)
        return results

    async def _execute_parallel_async(
        self,
        *,
        attack: AttackStrategy[AttackStrategyContextT, AttackStrategyResultT],
        contexts: List[AttackStrategyContextT],
    ) -> List[AttackStrategyResultT]:
        """
        Execute the same attack strategy against multiple contexts in parallel with concurrency control.

        This method uses asyncio semaphores to limit the number of concurrent executions.

        Args:
            attack (AttackStrategy[AttackStrategyContextT, AttackStrategyResultT]): The attack strategy to execute
                against all contexts.
            contexts (List[AttackStrategyContextT]): List of attack contexts to execute the strategy against.

        Returns:
            List[AttackStrategyResultT]: List of attack results in the same order as the input contexts.
                Each result corresponds to the execution of the strategy against the context
                at the same index.

        Raises:
            Any exceptions raised by the strategy execution will be propagated.
        """
        semaphore = asyncio.Semaphore(self._max_concurrency)

        # anonymous function to execute strategy with semaphore
        async def execute_with_semaphore(
            attack: AttackStrategy[AttackStrategyContextT, AttackStrategyResultT], ctx: AttackStrategyContextT
        ) -> AttackStrategyResultT:
            async with semaphore:
                return await attack.execute_with_context_async(context=ctx)

        tasks = [execute_with_semaphore(attack, ctx) for ctx in contexts]

        return await asyncio.gather(*tasks)
