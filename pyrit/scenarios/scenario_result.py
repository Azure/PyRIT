# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import List, Optional

import pyrit
from pyrit.models import AttackOutcome, AttackResult

logger = logging.getLogger(__name__)


class ScenarioIdentifier:
    """
    Scenario result class for aggregating results from multiple AttackRuns.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        scenario_version: int = 1,
        init_data: Optional[dict] = None,
    ):
        """
        Initialize a ScenarioIdentifier.

        Args:
            name (str): Name of the scenario.
            description (str): Description of the scenario.
            scenario_version (int): Version of the scenario.
            pyrit_version (Optional[str]): PyRIT version string.
            init_data (Optional[dict]): Initialization data.
        """
        self.name = name
        self.description = description
        self.version = scenario_version
        self.pyrit_version = pyrit.__version__
        self.init_data = init_data


class ScenarioResult:
    """
    Scenario result class for aggregating scenario results.
    """

    def __init__(
        self,
        *,
        scenario_identifier: ScenarioIdentifier,
        objective_target_identifier: dict,
        attack_results: dict[str, List[AttackResult]],
        objective_scorer_identifier: Optional[dict] = None,
    ) -> None:
        self.scenario_identifier = scenario_identifier
        self.objective_target_identifier = objective_target_identifier
        self.objective_scorer_identifier = objective_scorer_identifier
        self.attack_results = attack_results

    def get_strategies_used(self) -> List[str]:
        """Get the list of strategies used in this scenario."""
        return list(self.attack_results.keys())

    def get_objectives(self, *, attack_run_name: Optional[str] = None) -> List[str]:
        """
        Get the list of unique objectives for this scenario.

        Args:
            attack_run_name (Optional[str]): Name of specific attack run to include.
                If None, includes objectives from all attack runs. Defaults to None.

        Returns:
            List[str]: Deduplicated list of objectives.
        """
        objectives: List[str] = []
        strategies_to_process: List[List[AttackResult]]

        if not attack_run_name:
            # Include all attack runs
            strategies_to_process = list(self.attack_results.values())
        else:
            # Include only specified attack run
            if attack_run_name in self.attack_results:
                strategies_to_process = [self.attack_results[attack_run_name]]
            else:
                strategies_to_process = []

        for results in strategies_to_process:
            for result in results:
                objectives.append(result.objective)

        return list(set(objectives))

    def objective_achieved_rate(self, *, attack_run_name: Optional[str] = None) -> int:
        """
        Get the success rate of this scenario.

        Args:
            attack_run_name (Optional[str]): Name of specific attack run to calculate rate for.
                If None, calculates rate across all attack runs. Defaults to None.

        Returns:
            int: Success rate as a percentage (0-100).
        """
        if not attack_run_name:
            # Calculate rate across all attack runs
            all_results = []
            for results in self.attack_results.values():
                all_results.extend(results)
        else:
            # Calculate rate for specific attack run
            if attack_run_name in self.attack_results:
                all_results = self.attack_results[attack_run_name]
            else:
                return 0

        total_results = len(all_results)
        if total_results == 0:
            return 0

        successful_results = sum(1 for result in all_results if result.outcome == AttackOutcome.SUCCESS)
        return int((successful_results / total_results) * 100)
