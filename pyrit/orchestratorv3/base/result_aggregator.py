# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import statistics
from collections import defaultdict
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    cast,
)

from pyrit.orchestratorv3.base.core import (
    AttackResult,
    MultiTurnAttackResult,
    SingleTurnAttackResult,
)

T_Result = TypeVar("T_Result", bound=AttackResult)


class ResultAggregator(Generic[T_Result]):
    """
    Aggregates and analyzes attack results with type-safe methods

    This class handles different types of attack results (SingleTurn, MultiTurn)
    and provides appropriate analysis methods based on the result type
    (could be subclassed for more specific behavior)
    """

    def __init__(self, results: List[T_Result]):
        """
        Initialize the aggregator with a list of attack results

        Args:
            results: List of attack results to analyze
        """
        self.results = results
        self._result_type: Optional[Type[AttackResult]] = None

        # Determine result type if list is not empty
        if results:
            self._result_type = type(results[0])

    def is_multi_turn_results(self) -> bool:
        """Check if the results are from multi-turn attacks"""
        return bool(self.results and isinstance(self.results[0], MultiTurnAttackResult))

    def is_single_turn_results(self) -> bool:
        """Check if the results are from single-turn attacks"""
        return bool(self.results and isinstance(self.results[0], SingleTurnAttackResult))

    def success_rate(self) -> float:
        """
        Calculate success rate of attacks (for multi-turn results only)

        Returns:
            Success rate as a float between 0.0 and 1.0, or 0.0 if not applicable
        """
        if not self.results or not self.is_multi_turn_results():
            return 0.0

        multi_turn_results = cast(List[MultiTurnAttackResult], self.results)
        success_count = sum(1 for r in multi_turn_results if r.achieved_objective)
        return success_count / len(multi_turn_results)

    def average_turns(self) -> float:
        """
        Calculate average number of turns for attacks (multi-turn results only)

        Returns:
            Average number of turns, or 0.0 if not applicable
        """
        if not self.results or not self.is_multi_turn_results():
            return 0.0

        multi_turn_results = cast(List[MultiTurnAttackResult], self.results)
        return statistics.mean(r.executed_turns for r in multi_turn_results)

    def attack_duration_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics for attack duration (turns)

        Returns:
            Dictionary with turn statistics
        """
        if not self.results or not self.is_multi_turn_results():
            return {}

        multi_turn_results = cast(List[MultiTurnAttackResult], self.results)
        turns = [r.executed_turns for r in multi_turn_results]

        if not turns:
            return {}

        return {
            "mean": statistics.mean(turns),
            "median": statistics.median(turns),
            "min": min(turns),
            "max": max(turns),
            "stdev": statistics.stdev(turns) if len(turns) > 1 else 0,
        }

    def score_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics for score values (any result with scores)

        Returns:
            Dictionary with score statistics
        """
        scores = self._extract_scores()
        if not scores:
            return {}

        score_values = [s for s in scores if s is not None]
        if not score_values:
            return {}

        return {
            "mean": statistics.mean(score_values),
            "median": statistics.median(score_values),
            "min": min(score_values),
            "max": max(score_values),
            "stdev": statistics.stdev(score_values) if len(score_values) > 1 else 0,
        }

    def _extract_scores(self) -> List[Optional[float]]:
        """
        Extract numeric scores from results, handling different result types

        Returns:
            List of score values, with None for results without scores
        """
        scores = []

        for result in self.results:
            score_value = None

            if hasattr(result, "last_score") and getattr(result, "last_score") is not None:
                # For multi-turn results with last_score
                last_score = getattr(result, "last_score")
                if hasattr(last_score, "value") and isinstance(getattr(last_score, "value"), (int, float)):
                    score_value = getattr(last_score, "value")

            elif isinstance(result, SingleTurnAttackResult):
                # For single-turn results, use the average score of prompt responses
                single_turn_result = cast(SingleTurnAttackResult, result)
                prompt_scores = []

                # No need for hasattr check since we've cast to SingleTurnAttackResult
                for prompt_response in single_turn_result.prompt_list:
                    for piece in prompt_response.request_pieces:
                        for score in piece.scores:
                            if hasattr(score, "value") and isinstance(getattr(score, "value"), (int, float)):
                                prompt_scores.append(getattr(score, "value"))

                if prompt_scores:
                    score_value = statistics.mean(prompt_scores)

            scores.append(score_value)
        return scores

    def group_by_orchestrator(self) -> Dict[str, List[T_Result]]:
        """
        Group results by orchestrator type

        Returns:
            Dictionary mapping orchestrator types to results
        """
        grouped = defaultdict(list)
        for result in self.results:
            if "__type__" in result.orchestrator_identifier:
                orchestrator_type = result.orchestrator_identifier["__type__"]
                grouped[orchestrator_type].append(result)
        return dict(grouped)

    def group_by_objective(self) -> Dict[str, List[MultiTurnAttackResult]]:
        """
        Group multi-turn results by objective

        Returns:
            Dictionary mapping objectives to results, or empty dict for single-turn
        """
        if not self.is_multi_turn_results():
            return {}

        grouped = defaultdict(list)
        multi_turn_results = cast(List[MultiTurnAttackResult], self.results)

        for result in multi_turn_results:
            if result.objective:
                grouped[result.objective].append(result)
            else:
                grouped["<no objective>"].append(result)

        return dict(grouped)

    def compare_success_by_objective(self) -> Dict[str, float]:
        """
        Calculate success rate by objective

        Returns:
            Dictionary mapping objectives to success rates
        """
        if not self.is_multi_turn_results():
            return {}

        grouped = self.group_by_objective()
        success_rates = {}

        for objective, results in grouped.items():
            success_count = sum(1 for r in results if r.achieved_objective)
            success_rates[objective] = success_count / len(results) if results else 0

        return success_rates

    def single_turn_prompt_statistics(self) -> Dict[str, Any]:
        """
        Analyze prompts from single-turn results

        Returns:
            Statistics about prompts and their responses
        """
        if not self.is_single_turn_results():
            return {}

        single_turn_results = cast(List[SingleTurnAttackResult], self.results)
        prompt_counts = 0
        error_counts = 0
        blocked_counts = 0

        for result in single_turn_results:
            for prompt_response in result.prompt_list:
                prompt_counts += 1
                for piece in prompt_response.request_pieces:
                    if piece.has_error():
                        error_counts += 1
                    if piece.is_blocked():
                        blocked_counts += 1

        return {
            "total_prompts": prompt_counts,
            "error_count": error_counts,
            "blocked_count": blocked_counts,
            "success_rate": (prompt_counts - error_counts) / prompt_counts if prompt_counts > 0 else 0,
            "block_rate": blocked_counts / prompt_counts if prompt_counts > 0 else 0,
        }

    def summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of all attack results

        Returns:
            Dictionary with summary statistics
        """
        summary_data = {
            "total_attacks": len(self.results),
            "result_type": self._result_type.__name__ if self._result_type else "Unknown",
        }

        # Add multi-turn specific statistics
        if self.is_multi_turn_results():
            summary_data.update(
                {
                    "success_rate": self.success_rate(),
                    "average_turns": self.average_turns(),
                    "attack_duration_statistics": self.attack_duration_statistics(),
                    "success_by_objective": self.compare_success_by_objective(),
                }
            )

        # Add single-turn specific statistics
        if self.is_single_turn_results():
            summary_data["prompt_statistics"] = self.single_turn_prompt_statistics()

        # Add score statistics if available
        score_stats = self.score_statistics()
        if score_stats:
            summary_data["score_statistics"] = score_stats

        return summary_data
