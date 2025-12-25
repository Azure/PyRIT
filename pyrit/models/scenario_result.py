# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Literal, Optional

import pyrit
from pyrit.models import AttackOutcome, AttackResult

if TYPE_CHECKING:
    from pyrit.score.scorer_evaluation.scorer_evaluator import ScorerMetrics

logger = logging.getLogger(__name__)


class ScenarioIdentifier:
    """
    Scenario result class for aggregating results from multiple AtomicAttacks.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        scenario_version: int = 1,
        init_data: Optional[dict] = None,
        pyrit_version: Optional[str] = None,
    ):
        """
        Initialize a ScenarioIdentifier.

        Args:
            name (str): Name of the scenario.
            description (str): Description of the scenario.
            scenario_version (int): Version of the scenario.
            init_data (Optional[dict]): Initialization data.
            pyrit_version (Optional[str]): PyRIT version string. If None, uses current version.
        """
        self.name = name
        self.description = description
        self.version = scenario_version
        self.pyrit_version = pyrit_version if pyrit_version is not None else pyrit.__version__
        self.init_data = init_data


ScenarioRunState = Literal["CREATED", "IN_PROGRESS", "COMPLETED", "FAILED"]


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
        scenario_run_state: ScenarioRunState = "CREATED",
        labels: Optional[dict[str, str]] = None,
        completion_time: Optional[datetime] = None,
        number_tries: int = 0,
        id: Optional[uuid.UUID] = None,
    ) -> None:
        self.id = id if id is not None else uuid.uuid4()
        self.scenario_identifier = scenario_identifier
        self.objective_target_identifier = objective_target_identifier
        self.objective_scorer_identifier = objective_scorer_identifier or {}
        self.scenario_run_state = scenario_run_state
        self.attack_results = attack_results
        self.labels = labels if labels is not None else {}
        self.completion_time = completion_time if completion_time is not None else datetime.now(timezone.utc)
        self.number_tries = number_tries

    def get_strategies_used(self) -> List[str]:
        """Get the list of strategies used in this scenario."""
        return list(self.attack_results.keys())

    def get_objectives(self, *, atomic_attack_name: Optional[str] = None) -> List[str]:
        """
        Get the list of unique objectives for this scenario.

        Args:
            atomic_attack_name (Optional[str]): Name of specific atomic attack to include.
                If None, includes objectives from all atomic attacks. Defaults to None.

        Returns:
            List[str]: Deduplicated list of objectives.
        """
        objectives: List[str] = []
        strategies_to_process: List[List[AttackResult]]

        if not atomic_attack_name:
            # Include all atomic attacks
            strategies_to_process = list(self.attack_results.values())
        else:
            # Include only specified atomic attack
            if atomic_attack_name in self.attack_results:
                strategies_to_process = [self.attack_results[atomic_attack_name]]
            else:
                strategies_to_process = []

        for results in strategies_to_process:
            for result in results:
                objectives.append(result.objective)

        return list(set(objectives))

    def objective_achieved_rate(self, *, atomic_attack_name: Optional[str] = None) -> int:
        """
        Get the success rate of this scenario.

        Args:
            atomic_attack_name (Optional[str]): Name of specific atomic attack to calculate rate for.
                If None, calculates rate across all atomic attacks. Defaults to None.

        Returns:
            int: Success rate as a percentage (0-100).
        """
        if not atomic_attack_name:
            # Calculate rate across all atomic attacks
            all_results = []
            for results in self.attack_results.values():
                all_results.extend(results)
        else:
            # Calculate rate for specific atomic attack
            if atomic_attack_name in self.attack_results:
                all_results = self.attack_results[atomic_attack_name]
            else:
                return 0

        total_results = len(all_results)
        if total_results == 0:
            return 0

        successful_results = sum(1 for result in all_results if result.outcome == AttackOutcome.SUCCESS)
        return int((successful_results / total_results) * 100)

    def get_scorer_evaluation_metrics(self) -> Optional["ScorerMetrics"]:
        """
        Get the evaluation metrics for the scenario's scorer from the scorer evaluation registry.

        Returns:
            ScorerMetrics: The evaluation metrics object, or None if not found.
        """
        # import here to avoid circular imports
        from pyrit.score.scorer_evaluation.scorer_metrics_io import find_objective_metrics_by_hash

        # Use the stored hash directly for lookup (avoids needing to reconstruct ScorerIdentifier)
        scorer_hash = self.objective_scorer_identifier.get("hash")
        if not scorer_hash:
            return None

        return find_objective_metrics_by_hash(hash=scorer_hash)
