# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from enum import Enum
from typing import List, TypeVar

from pyrit.common.logger import logger
from pyrit.orchestratorv3.base.attack_strategy import AttackStrategy
from pyrit.orchestratorv3.base.core import (
    AttackContext,
    AttackResult,
)

T_Context = TypeVar("T_Context", bound=AttackContext)
T_Result = TypeVar("T_Result", bound=AttackResult)


class ResultSelectionStrategy(Enum):
    """Strategies for selecting a result from parallel executions."""

    # First result that achieved objective
    FIRST_SUCCESS = "first_success"
    # Result with the highest score
    BEST_SCORE = "best_score"


class AttackPipeline(AttackStrategy[T_Context, T_Result]):
    """
    Executes multiple strategies in parallel with proper context isolation

    This class allows running multiple attack strategies at the same time against
    the same context (each with its own duplicate), and then selecting the
    result based on a configurable selection strategy.
    """

    def __init__(
        self,
        *,
        strategies: List[AttackStrategy[T_Context, T_Result]],
        name: str = "parallel_attack_pipeline",
        max_concurrency: int = 5,
        result_selection: ResultSelectionStrategy = ResultSelectionStrategy.FIRST_SUCCESS,
    ):
        """
        Initialize the pipeline with a list of strategies

        Args:
            strategies: List of strategies to execute in parallel
            name: Name identifier for this pipeline
            max_concurrency: Maximum number of strategies to run concurrently
            result_selection: How to select the final result from multiple parallel executions
        """
        super().__init__(logger=logger)

        if not strategies:
            raise ValueError("Pipeline contains no strategies")

        self._strategies = strategies
        self._name = name
        self._max_concurrency = max_concurrency
        self._result_selection = result_selection

    async def execute(self, *, context: T_Context) -> T_Result:
        """
        Execute all strategies in parallel (duplicating context for each)

        Args:
            context: The initial context

        Returns:
            Result from the strategy with the best outcome based on selection criteria
        """
        self._logger.info(f"Starting parallel attack pipeline '{self._name}' with {len(self._strategies)} strategies")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def execute_strategy_with_semaphore(
            *, strategy: AttackStrategy[T_Context, T_Result], ctx: T_Context
        ) -> T_Result:
            """Execute a strategy with semaphore-based concurrency control"""
            async with semaphore:
                self._logger.info(f"Executing strategy: {strategy.__class__.__name__}")
                return await strategy.execute(context=ctx)

        # Create tasks for each strategy with its own duplicate context
        tasks = []
        for strategy in self._strategies:
            strategy_context = context.duplicate()
            tasks.append(execute_strategy_with_semaphore(strategy=strategy, ctx=strategy_context))

        # Execute all strategies in parallel and wait for all results
        self._logger.info(f"Executing {len(tasks)} strategies in parallel (max concurrency: {self._max_concurrency})")
        results = await asyncio.gather(*tasks)

        # Select result based on configured selection strategy
        final_result = await self._select_result(results=results)
        return final_result

    async def _select_result(self, *, results: List[T_Result]) -> T_Result:
        """
        Select the result to return based on the configured selection strategy

        Args:
            results: List of results from all strategies

        Returns:
            The selected result based on the configured strategy
        """
        if not results:
            raise ValueError("No results available from execution")

        # If there is only one result, return it
        if len(results) == 1:
            return results[0]

        # Select appropriate result-selection strategy
        match self._result_selection:
            case ResultSelectionStrategy.FIRST_SUCCESS:
                return await self._select_first_success_result(results=results)
            case ResultSelectionStrategy.BEST_SCORE:
                return await self._select_best_score_result(results=results)
            case _:
                self._logger.warning(f"Unknown result selection strategy: {self._result_selection}, using first result")
                return results[0]

    async def _select_first_success_result(self, *, results: List[T_Result]) -> T_Result:
        """
        Select the first result that achieved its objective

        Args:
            results: List of results from all strategies

        Returns:
            The first successful result or the first result if none were successful
        """
        # For multi-turn results, check achieved_objective
        if all(hasattr(r, "achieved_objective") for r in results):
            successful_results = [r for r in results if getattr(r, "achieved_objective", False)]
            if successful_results:
                self._logger.info(f"Found {len(successful_results)} successful results, returning the first one")
                return successful_results[0]

        # If no successful results or not multi-turn, return the first result
        self._logger.info("No successful results found, returning the first result")
        return results[0]

    async def _select_best_score_result(self, *, results: List[T_Result]) -> T_Result:
        """
        Select the result with the highest score

        Args:
            results: List of results from all strategies

        Returns:
            The result with the highest score or falls back to first_success if no scores
        """
        # For scored results, select the one with highest score
        results_with_scores = []
        for r in results:
            if hasattr(r, "last_score") and getattr(r, "last_score") is not None:
                last_score = getattr(r, "last_score")
                # Check if score has a numeric value
                if hasattr(last_score, "value") and isinstance(last_score.value, (int, float)):
                    results_with_scores.append((r, last_score.value))

        if results_with_scores:
            # Sort by score (descending) and return the highest
            results_with_scores.sort(key=lambda x: x[1], reverse=True)
            best_result, best_score = results_with_scores[0]
            self._logger.info(f"Selected result with highest score: {best_score}")
            return best_result

        # If no scores available, fall back to first_success strategy
        self._logger.warning("No scores available, falling back to first_success selection")
        return await self._select_first_success_result(results=results)

    # The base AttackStrategy methods still need to be implemented
    # but theyn will never be called due to execute() override

    async def _setup(self, *, context: T_Context) -> None:
        """This is not used in the pipeline."""
        pass

    async def _perform_attack(self, *, context: T_Context) -> T_Result:
        """This is not used in the pipeline."""
        raise NotImplementedError("This method should never be called - execute() is overridden")

    async def _teardown(self, *, context: T_Context) -> None:
        """This is not used in the pipeline."""
        pass
