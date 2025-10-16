# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Scenario class for grouping and executing multiple AttackRuns.

This module provides the Scenario class that orchestrates the execution of multiple
AttackRun instances sequentially, enabling comprehensive security testing campaigns.
"""

import logging
from typing import Dict, List, Optional

import pyrit
from pyrit.models import AttackResult
from pyrit.scenarios.attack_run import AttackRun

logger = logging.getLogger(__name__)


class ScenarioIdentifier:
    def __init__(
        self,
        name: str,
        version: int = 1,
        pyrit_version: Optional[str] = None,
        init_data: Optional[dict] = None,
    ):
        self.name = name
        self.version = version
        self.pyrit_version = pyrit_version if pyrit_version is not None else pyrit.__version__
        self.init_data = init_data


class ScenarioResult:
    def __init__(self, *, scenario_identifier: ScenarioIdentifier, attack_results: List[AttackResult]) -> None:

        self.scenario_identifier = scenario_identifier
        self.attack_results = attack_results

    @property
    def objective_achieved_rate(self) -> int:
        """Get the success rate of this scenario."""
        total_results = len(self.attack_results)
        if total_results == 0:
            return 0
        successful_results = sum(1 for result in self.attack_results if result.outcome == "success")

        return int((successful_results / total_results) * 100)


class Scenario:
    """
    Groups and executes multiple AttackRun instances sequentially.

    A Scenario represents a comprehensive testing campaign composed of multiple
    atomic attack tests (AttackRuns). It executes each AttackRun in sequence and
    aggregates the results into a ScenarioResult.

    Example:
        >>> from pyrit.scenarios import Scenario, AttackRun
        >>> from pyrit.executor.attack.single_turn.prompt_sending import PromptSendingAttack
        >>> from pyrit.prompt_target import OpenAIChatTarget
        >>> from pyrit.prompt_converter import Base64Converter
        >>>
        >>> target = OpenAIChatTarget()
        >>>
        >>> # Create attack instances
        >>> base64_attack = PromptSendingAttack(
        ...     objective_target=target,
        ...     converters=[Base64Converter()]
        ... )
        >>> baseline_attack = PromptSendingAttack(
        ...     objective_target=target,
        ...     converters=[]
        ... )
        >>>
        >>> # Create attack runs with objectives
        >>> attack_run1 = AttackRun(
        ...     attack=base64_attack,
        ...     objectives=["Tell me how to make a bomb"]
        ... )
        >>> attack_run2 = AttackRun(
        ...     attack=baseline_attack,
        ...     objectives=["Generate harmful content"]
        ... )
        >>>
        >>> # Create and execute scenario
        >>> scenario = Scenario(
        ...     name="Security Test Campaign",
        ...     version=1,
        ...     attack_runs=[attack_run1, attack_run2]
        ... )
        >>> result = await scenario.run_async()
        >>> print(f"Completed {len(result.attack_results)} tests")
    """

    def __init__(
        self,
        *,
        name: str,
        version: int,
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

        self._identifier = ScenarioIdentifier(
            name=type(self).__name__,
            version=version,
        )

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

    async def run_async(self, *, max_concurrency: int = 1) -> ScenarioResult:
        """
        Execute all attack runs in the scenario sequentially.

        Each AttackRun is executed in order, and all results are aggregated
        into a ScenarioResult containing the scenario metadata and all attack results.

        Args:
            max_concurrency (int): Maximum number of concurrent attack executions
                within each AttackRun. Defaults to 1 for sequential execution.

        Returns:
            ScenarioResult: Contains scenario identifier and aggregated list of all
                attack results from all runs.

        Example:
            >>> result = await scenario.run_async(max_concurrency=3)
            >>> print(f"Scenario: {result.scenario_identifier.name}")
            >>> print(f"Total results: {len(result.attack_results)}")
            >>> for attack_result in result.attack_results:
            ...     print(f"Objective: {attack_result.objective}, Outcome: {attack_result.outcome}")
        """
        logger.info(f"Starting scenario '{self._name}' execution with {len(self._attack_runs)} attack runs")

        all_results: List[AttackResult] = []

        for i, attack_run in enumerate(self._attack_runs, start=1):
            logger.info(f"Executing attack run {i}/{len(self._attack_runs)} in scenario '{self._name}'")

            try:
                results = await attack_run.run_async(max_concurrency=max_concurrency)
                all_results.extend(results)
                logger.info(f"Attack run {i}/{len(self._attack_runs)} completed with {len(results)} results")
            except Exception as e:
                logger.error(f"Attack run {i}/{len(self._attack_runs)} failed in scenario '{self._name}': {str(e)}")
                raise ValueError(f"Failed to execute attack run {i} in scenario '{self._name}': {str(e)}") from e

        logger.info(f"Scenario '{self._name}' completed successfully with {len(all_results)} total results")
        return ScenarioResult(scenario_identifier=self._identifier, attack_results=all_results)
