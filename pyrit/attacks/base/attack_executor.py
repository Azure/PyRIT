# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from typing import List

from pyrit.attacks.base.attack_strategy import AttackStrategy
from pyrit.attacks.base.context import ContextT
from pyrit.attacks.base.result import ResultT


class AttackExecutor:
    """
    Manages the execution of attack strategies with support for different execution patterns.

    The AttackExecutor provides controlled execution of attack strategies with features like
    concurrency limiting and parallel execution. It can handle multiple objectives against
    the same target or execute different strategies concurrently.
    """

    def __init__(self, *, max_concurrency: int = 5):
        """
        Initialize the attack executor with configurable concurrency control.

        Args:
            max_concurrency (int): Maximum number of concurrent attack executions allowed.
                Must be a positive integer. Defaults to 5 to balance performance and
                resource usage.

        Raises:
            ValueError: If max_concurrency is not a positive integer.
        """
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be a positive integer, got {max_concurrency}")
        self._max_concurrency = max_concurrency

    async def execute_multi_objective_attack(
        self, attack: AttackStrategy[ContextT, ResultT], context_template: ContextT, objectives: List[str]
    ) -> List[ResultT]:
        """
        Execute the same attack strategy with multiple objectives against the same target in parallel.

        This is a convenience method for the common use case of testing multiple attack objectives
        against the same target using the same strategy. It creates a separate context for each
        objective by duplicating the template context and setting the objective.

        Args:
            attack (AttackStrategy[ContextT, ResultT]): The attack strategy to use for all objectives
            context_template (ContextT): Template context that will be duplicated for each objective.
                Must have a 'duplicate()' method and an 'objective' attribute.
            objectives (List[str]): List of attack objectives to test. Each objective will be
                executed as a separate attack using a copy of the context template.

        Returns:
            List[ResultT]: List of attack results in the same order as the objectives list.
                Each result corresponds to the execution of the strategy with the objective
                at the same index.

        Raises:
            AttributeError: If the context_template doesn't have required 'duplicate()' method
                or 'objective' attribute.
            Any exceptions raised during strategy execution will be propagated.

        Example:
            >>> executor = AttackExecutor(max_concurrency=3)
            >>> results = await executor.execute_multi_objective_attack(
            ...     attack=prompt_injection_attack,
            ...     context_template=base_context,
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
        results = await self._execute_parallel(attack=attack, contexts=contexts)
        return results

    async def _execute_parallel(
        self, *, attack: AttackStrategy[ContextT, ResultT], contexts: List[ContextT]
    ) -> List[ResultT]:
        """
        Execute the same attack strategy against multiple contexts in parallel with concurrency control.

        This method uses asyncio semaphores to limit the number of concurrent executions,
        preventing resource exhaustion while maintaining parallelism.

        Args:
            attack (AttackStrategy[ContextT, ResultT]): The attack strategy to execute
                against all contexts
            contexts (List[ContextT]): List of attack contexts to execute the strategy against

        Returns:
            List[ResultT]: List of attack results in the same order as the input contexts.
                Each result corresponds to the execution of the strategy against the context
                at the same index.

        Raises:
            Any exceptions raised by the strategy execution will be propagated.
        """
        semaphore = asyncio.Semaphore(self._max_concurrency)

        # anonymous function to execute strategy with semaphore
        async def execute_with_semaphore(attack: AttackStrategy[ContextT, ResultT], ctx: ContextT) -> ResultT:
            async with semaphore:
                return await attack.execute_async(context=ctx)

        tasks = [execute_with_semaphore(attack, ctx) for ctx in contexts]

        return await asyncio.gather(*tasks)
