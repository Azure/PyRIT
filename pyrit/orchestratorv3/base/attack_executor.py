# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from typing import List, TypeVar

from pyrit.orchestratorv3.base.attack_strategy import AttackStrategy
from pyrit.orchestratorv3.base.core import (
    MultiTurnAttackContext,
    MultiTurnAttackResult,
)

MT_Context = TypeVar("MT_Context", bound=MultiTurnAttackContext)
MT_Result = TypeVar("MT_Result", bound=MultiTurnAttackResult)


class AttackExecutor:
    """
    Provides different type of attack execution patterns (e.g. we could add execution with timeouts, etc.)
    """

    async def execute_parallel(
        self, *, strategy: AttackStrategy[MT_Context, MT_Result], contexts: List[MT_Context], max_concurrency: int = 5
    ) -> List[MT_Result]:
        """
        Execute multiple strategies in parallel with concurrency control.

        Args:
            strategy: The attack strategy to use
            contexts: List of contexts to execute the strategy against
            max_concurrency: Maximum number of concurrent executions

        Returns:
            List of attack results
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        # annonymous function to execute strategy with semaphore
        async def execute_with_semaphore(strategy: AttackStrategy[MT_Context, MT_Result], ctx: MT_Context) -> MT_Result:
            async with semaphore:
                return await strategy.execute(context=ctx)

        tasks = [execute_with_semaphore(strategy, ctx) for ctx in contexts]

        return await asyncio.gather(*tasks)

    async def execute_multi_objective_attack(
        self,
        strategy: AttackStrategy[MT_Context, MT_Result],
        context_template: MT_Context,
        objectives: List[str],
        max_concurrency: int = 5,
    ) -> List[MT_Result]:
        """
        Execute multi-turn attacks with multiple objectives against the same target in parallel

        This is a simplified method designed for the common case of running the same strategy
        with multiple different objectives. Works with any context type that extends
        MultiTurnAttackContext.

        Args:
            strategy: The attack strategy to use for all objectives
            context_template: Template context to duplicate for each objective
            objectives: List of objectives to attack
            max_concurrency: Maximum number of concurrent executions

        Returns:
            List of attack results
        """
        contexts = []

        for objective in objectives:
            # Create a deep copy of the context using its duplicate method
            context = context_template.duplicate()
            # Set the new objective (all multi-turn contexts have objectives)
            context.objective = objective
            contexts.append(context)

        # Run strategies in parallel
        results = await self.execute_parallel(strategy=strategy, contexts=contexts, max_concurrency=max_concurrency)
        return results
