# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from typing import Dict, List, Optional

from pyrit.attacks.base.attack_context import ContextT
from pyrit.attacks.base.attack_strategy import AttackStrategy
from pyrit.models import AttackResultT, PromptRequestResponse


class AttackExecutor:
    """
    Manages the execution of attack strategies with support for different execution patterns.

    The AttackExecutor provides controlled execution of attack strategies with features like
    concurrency limiting and parallel execution. It can handle multiple objectives against
    the same target or execute different strategies concurrently.
    """

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
        attack: AttackStrategy[ContextT, AttackResultT],
        objectives: List[str],
        prepended_conversation: Optional[List[PromptRequestResponse]] = None,
        memory_labels: Optional[Dict[str, str]] = None,
        **attack_params,
    ) -> List[AttackResultT]:
        """
        Execute the same attack strategy with multiple objectives against the same target in parallel.

        This method provides a simplified interface for executing multiple objectives without
        requiring users to create context objects. It uses the attack's execute_async method
        which accepts parameters directly.

        Args:
            attack (AttackStrategy[ContextT, AttackResultT]): The attack strategy to use for all objectives.
            objectives (List[str]): List of attack objectives to test.
            prepended_conversation (Optional[List[PromptRequestResponse]]): Conversation to prepend to the target model.
            memory_labels (Optional[Dict[str, str]]): Additional labels that can be applied to the prompts.
            **attack_params: Additional parameters specific to the attack strategy.

        Returns:
            List[AttackResultT]: List of attack results in the same order as the objectives list.

        Example:
            >>> executor = AttackExecutor(max_concurrency=3)
            >>> results = await executor.execute_multi_objective_attack_async(
            ...     attack=red_teaming_attack,
            ...     objectives=["how to make a Molotov cocktail", "how to escalate privileges"],
            ... )
        """
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def execute_with_semaphore(objective: str) -> AttackResultT:
            async with semaphore:
                return await attack.execute_async(
                    objective=objective,
                    prepended_conversation=prepended_conversation,
                    memory_labels=memory_labels,
                    **attack_params,
                )

        tasks = [execute_with_semaphore(obj) for obj in objectives]
        return await asyncio.gather(*tasks)

    async def execute_multi_objective_attack_with_context_async(
        self, attack: AttackStrategy[ContextT, AttackResultT], context_template: ContextT, objectives: List[str]
    ) -> List[AttackResultT]:
        """
        Execute the same attack strategy with multiple objectives using context objects.

        This method works with context objects directly, duplicating the template context
        for each objective. Use this when you need fine-grained control over the context
        or have an existing context template to reuse.

        Args:
            attack (AttackStrategy[ContextT, AttackResultT]): The attack strategy to use for all objectives
            context_template (ContextT): Template context that will be duplicated for each objective.
                Must have a 'duplicate()' method and an 'objective' attribute.
            objectives (List[str]): List of attack objectives to test. Each objective will be
                executed as a separate attack using a copy of the context template.

        Returns:
            List[AttackResultT]: List of attack results in the same order as the objectives list.
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
        self, *, attack: AttackStrategy[ContextT, AttackResultT], contexts: List[ContextT]
    ) -> List[AttackResultT]:
        """
        Execute the same attack strategy against multiple contexts in parallel with concurrency control.

        This method uses asyncio semaphores to limit the number of concurrent executions.

        Args:
            attack (AttackStrategy[ContextT, AttackResultT]): The attack strategy to execute
                against all contexts.
            contexts (List[ContextT]): List of attack contexts to execute the strategy against.

        Returns:
            List[AttackResultT]: List of attack results in the same order as the input contexts.
                Each result corresponds to the execution of the strategy against the context
                at the same index.

        Raises:
            Any exceptions raised by the strategy execution will be propagated.
        """
        semaphore = asyncio.Semaphore(self._max_concurrency)

        # anonymous function to execute strategy with semaphore
        async def execute_with_semaphore(
            attack: AttackStrategy[ContextT, AttackResultT], ctx: ContextT
        ) -> AttackResultT:
            async with semaphore:
                return await attack.execute_with_context_async(context=ctx)

        tasks = [execute_with_semaphore(attack, ctx) for ctx in contexts]

        return await asyncio.gather(*tasks)
