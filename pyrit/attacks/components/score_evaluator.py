# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Optional

from pyrit.models.score import Score
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class ScoreEvaluator:
    """
    Evaluates scores for orchestrator responses.

    Responsible for:
    - Evaluating if objectives are achieved
    - Checking if responses are refused
    - Tracking score history
    - Providing score feedback
    """

    def __init__(
        self, *, scorer: Scorer, use_score_as_feedback: bool = True, objective_achieved_score_threshold: float = 0.8
    ):
        """
        Initialize the score evaluator.

        Args:
            scorer: The scorer to use for evaluating responses
            use_score_as_feedback: Whether to use the score as feedback for future prompts
            objective_achieved_score_threshold: The score threshold to consider an objective achieved
        """
        self._scorer = scorer
        self._use_score_as_feedback = use_score_as_feedback
        self._objective_achieved_score_threshold = objective_achieved_score_threshold

    def is_objective_achieved(self, score: Score) -> bool:
        """
        Determine if the objective is achieved based on score.

        Args:
            score: The score to check, or the latest score if None

        Returns:
            True if the objective is achieved, False otherwise
        """
        if not score:
            return False

        score_type = score.score_type
        score_value = score.get_value()

        # Handle true_false scores
        if score_type == "true_false":
            return bool(score_value)

        # Handle numeric scores
        if score_type == "numeric":
            return float(score_value) >= self._objective_achieved_score_threshold

        # For other score types, assume not achieved
        return False

    def get_feedback(self, score: Score) -> Optional[str]:
        """
        Get feedback from a score for use in future prompts.

        Args:
            score: The score to get feedback from, or the latest score if None

        Returns:
            Feedback string, or None if no suitable feedback exists
        """
        if not score or not self._use_score_as_feedback:
            return None

        return score.score_rationale

    @property
    def scorer_type(self) -> str:
        """Get the type of the scorer."""
        return self._scorer.get_identifier()["__type__"]
