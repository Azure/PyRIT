# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Scenario class for grouping and executing multiple AttackRuns.

This module provides the Scenario class that orchestrates the execution of multiple
AttackRun instances sequentially, enabling comprehensive security testing campaigns.
"""

import logging
from typing import Dict, List, Optional

from pyrit.models import AttackResult
from pyrit.scenarios.attack_run import AttackRun

logger = logging.getLogger(__name__)


class Scenario:
    """
    Groups and executes multiple AttackRun instances sequentially.

    A Scenario represents a comprehensive testing campaign composed of multiple
    atomic attack tests (AttackRuns). It executes each AttackRun in sequence and
    aggregates the results.

    Example:
        >>> from pyrit.scenarios import Scenario, AttackRun
        >>> from pyrit.setup import ConfigurationPaths
        >>> from pyrit.prompt_target import OpenAIChatTarget
        >>>
        >>> target = OpenAIChatTarget()
        >>> attack_run1 = AttackRun(
        ...     attack_config=ConfigurationPaths.attack.foundry.crescendo,
        ...     dataset_config=ConfigurationPaths.dataset.harm_bench,
        ...     objective_target=target
        ... )
        >>> attack_run2 = AttackRun(
        ...     attack_config=ConfigurationPaths.attack.foundry.ascii_art,
        ...     dataset_config=ConfigurationPaths.dataset.harm_bench,
        ...     objective_target=target
        ... )
        >>> scenario = Scenario(
        ...     name="Foundry Tests",
        ...     attack_runs=[attack_run1, attack_run2]
        ... )
        >>> results = await scenario.run_async()
    """

    def __init__(
        self,
        *,
        name: str,
        attack_runs: List[AttackRun],
        memory_labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize a scenario with a collection of attack runs.

        Args:
            name (str): Descriptive name for the scenario.
            attack_runs (List[AttackRun]): List of AttackRun instances to execute.
            memory_labels (Optional[Dict[str, str]]): Additional labels to apply to all
                attack runs in the scenario. These help track and categorize the scenario.

        Raises:
            ValueError: If attack_runs list is empty.
        """
        if not attack_runs:
            raise ValueError("Scenario must contain at least one AttackRun")

        self._name = name
        self._attack_runs = attack_runs
        self._memory_labels = memory_labels or {}

        logger.info(f"Initialized scenario '{name}' with {len(attack_runs)} attack runs")

    @property
    def name(self) -> str:
        """Get the name of the scenario."""
        return self._name

    @property
    def attack_run_count(self) -> int:
        """Get the number of attack runs in this scenario."""
        return len(self._attack_runs)

    async def run_async(self, *, max_concurrency: int = 1) -> List[AttackResult]:
        """
        Execute all attack runs in the scenario sequentially.

        Each AttackRun is executed in order, and all results are aggregated
        into a single list.

        Args:
            max_concurrency (int): Maximum number of concurrent attack executions
                within each AttackRun. Defaults to 1 for sequential execution.

        Returns:
            List[AttackResult]: Aggregated list of all attack results from all runs.

        Example:
            >>> results = await scenario.run_async(max_concurrency=3)
            >>> print(f"Total results: {len(results)}")
            >>> for result in results:
            ...     print(f"Objective: {result.objective}, Outcome: {result.outcome}")
        """
        logger.info(f"Starting scenario '{self._name}' execution with {len(self._attack_runs)} attack runs")

        all_results: List[AttackResult] = []

        for i, attack_run in enumerate(self._attack_runs, start=1):
            logger.info(f"Executing attack run {i}/{len(self._attack_runs)} in scenario '{self._name}'")

            try:
                results = await attack_run.run_async(max_concurrency=max_concurrency)
                all_results.extend(results)
                logger.info(
                    f"Attack run {i}/{len(self._attack_runs)} completed with {len(results)} results"
                )
            except Exception as e:
                logger.error(
                    f"Attack run {i}/{len(self._attack_runs)} failed in scenario '{self._name}': {str(e)}"
                )
                raise ValueError(
                    f"Failed to execute attack run {i} in scenario '{self._name}': {str(e)}"
                ) from e

        logger.info(
            f"Scenario '{self._name}' completed successfully with {len(all_results)} total results"
        )
        return all_results
