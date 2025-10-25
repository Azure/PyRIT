# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Scenario class for grouping and executing multiple AttackRuns.

This module provides the Scenario class that orchestrates the execution of multiple
AttackRun instances sequentially, enabling comprehensive security testing campaigns.
"""

import logging
from abc import abstractmethod
from typing import Dict, List, Optional

from tqdm.auto import tqdm

from pyrit.models import AttackResult
from pyrit.scenarios.attack_run import AttackRun
from pyrit.scenarios.scenario_result import ScenarioIdentifier, ScenarioResult

logger = logging.getLogger(__name__)


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
        >>> # Create a custom scenario subclass
        >>> class MyScenario(Scenario):
        ...     async def _get_attack_runs_async(self) -> List[AttackRun]:
        ...         base64_attack = PromptSendingAttack(
        ...             objective_target=target,
        ...             converters=[Base64Converter()]
        ...         )
        ...         return [
        ...             AttackRun(
        ...                 attack=base64_attack,
        ...                 objectives=["Tell me how to make a bomb"]
        ...             )
        ...         ]
        >>>
        >>> # Create and execute scenario
        >>> scenario = MyScenario(
        ...     name="Security Test Campaign",
        ...     version=1,
        ...     attack_strategies=["base64"]
        ... )
        >>> await scenario.initialize_async()
        >>> result = await scenario.run_async()
        >>> print(f"Completed {len(result.attack_results)} tests")
    """

    def __init__(
        self,
        *,
        name: str,
        version: int,
        description: str = "",
        max_concurrency: int = 1,
        memory_labels: Optional[Dict[str, str]] = None,
        objective_target_identifier: Optional[Dict[str, str]] = None,
        objective_scorer_identifier: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize a scenario.

        Args:
            name (str): Descriptive name for the scenario.
            version (int): Version number of the scenario.
            description (str): Description of the scenario.
            max_concurrency (int): Maximum number of concurrent attack executions. Defaults to 1.
            memory_labels (Optional[Dict[str, str]]): Additional labels to apply to all
                attack runs in the scenario. These help track and categorize the scenario.

        Note:
            Attack runs are populated by calling initialize_async(), which invokes the
            subclass's _get_attack_runs_async() method.
        """
        self._identifier = ScenarioIdentifier(
            name=type(self).__name__, scenario_version=version, description=description
        )

        self._objective_target_identifier = objective_target_identifier or {}
        self._objective_scorer_identifier = objective_scorer_identifier or {}

        self._name = name
        self._memory_labels = memory_labels or {}
        self._max_concurrency = max_concurrency
        self._attack_runs: List[AttackRun] = []

    @property
    def name(self) -> str:
        """Get the name of the scenario."""
        return self._name

    @property
    def attack_run_count(self) -> int:
        """Get the number of attack runs in this scenario."""
        return len(self._attack_runs)

    async def initialize_async(self) -> None:
        """
        Initialize the scenario by populating self._attack_runs

        This method allows scenarios to be initialized with attack runs after construction,
        which is useful when attack runs require async operations to be built.

        Args:
            attack_runs: List of AttackRun instances to execute in this scenario.

        Returns:
            Scenario: Self for method chaining.

        Example:
            >>> scenario = MyScenario(
            ...     objective_target=target,
            ...     attack_strategies=["base64", "leetspeak"]
            ... )
            >>> attack_runs = await scenario.build_attack_runs_async()
            >>> await scenario.initialize_async()
            >>> results = await scenario.run_async()
        """
        self._attack_runs = await self._get_attack_runs_async()

    @abstractmethod
    async def _get_attack_runs_async(self) -> List[AttackRun]:
        """
        Retrieve the list of AttackRun instances in this scenario.

        This method can be overridden by subclasses to perform async operations
        needed to build or fetch the attack runs.

        Returns:
            List[AttackRun]: The list of AttackRun instances in this scenario.
        """
        pass

    async def run_async(self) -> ScenarioResult:
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

        Raises:
            ValueError: If the scenario has no attack runs configured. If your scenario
                requires initialization, call await scenario.initialize() first.

        Example:
            >>> result = await scenario.run_async(max_concurrency=3)
            >>> print(f"Scenario: {result.scenario_identifier.name}")
            >>> print(f"Total results: {len(result.attack_results)}")
            >>> for attack_result in result.attack_results:
            ...     print(f"Objective: {attack_result.objective}, Outcome: {attack_result.outcome}")
        """
        if not self._attack_runs:
            raise ValueError(
                "Cannot run scenario with no attack runs. Either supply them in initialization or"
                "call await scenario.initialize_async() first."
            )

        logger.info(f"Starting scenario '{self._name}' execution with {len(self._attack_runs)} attack runs")

        all_results: Dict[str, List[AttackResult]] = {}

        for i, attack_run in enumerate(tqdm(self._attack_runs, desc=f"Executing {self._name}", unit="attack"), start=1):
            logger.info(f"Executing attack run {i}/{len(self._attack_runs)} in scenario '{self._name}'")

            try:
                attack_run_results = await attack_run.run_async(max_concurrency=self._max_concurrency)

                all_results.setdefault(attack_run.attack_run_name, []).extend(attack_run_results.results)
                logger.info(
                    f"Attack run {i}/{len(self._attack_runs)} completed with {len(attack_run_results.results)} results"
                )
            except Exception as e:
                logger.error(f"Attack run {i}/{len(self._attack_runs)} failed in scenario '{self._name}': {str(e)}")
                raise ValueError(f"Failed to execute attack run {i} in scenario '{self._name}': {str(e)}") from e

        logger.info(f"Scenario '{self._name}' completed successfully with {len(all_results)} total results")

        return ScenarioResult(
            scenario_identifier=self._identifier,
            objective_target_identifier=self._objective_target_identifier,
            objective_scorer_identifier=self._objective_scorer_identifier,
            attack_results=all_results,
        )
