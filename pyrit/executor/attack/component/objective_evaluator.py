# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Optional

from pyrit.models import Score
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class ObjectiveEvaluator:
    """
    Evaluates scores from a Scorer to determine objective achievement and provide feedback.

    This component wraps a Scorer and provides logic for:
    - Determining if an objective is achieved based on score values and thresholds.
    - Extracting feedback from score rationales for use in subsequent prompts.
    - Supporting both true/false and numeric score types.
    """

    def __init__(
        self, *, scorer: Scorer, use_score_as_feedback: bool = True, successful_objective_threshold: float = 0.8
    ):
        """
        Initialize the objective evaluator.

        Args:
            scorer (Scorer): The scorer to use for evaluating responses
            use_score_as_feedback (bool): Whether to use the score as feedback for future prompts
            successful_objective_threshold (float): The score threshold to consider an
                                                   objective successful [0.0 to 1.0, inclusive].

        Raises:
            ValueError: If successful_objective_threshold is not between 0.0 and 1.0 (inclusive).
        """
        self._scorer = scorer
        self._use_score_as_feedback = use_score_as_feedback

        if not 0.0 <= successful_objective_threshold <= 1.0:
            raise ValueError("successful_objective_threshold must be between 0.0 and 1.0 (inclusive)")

        self._successful_objective_threshold = successful_objective_threshold

    def is_objective_achieved(self, score: Optional[Score] = None) -> bool:
        """
        Determine if the objective is achieved based on score.

        Args:
            score (Optional[Score]): The score to check.

        Returns:
            bool: True if the objective is achieved, False otherwise.

        Raises:
            ValueError: If score value cannot be converted to the expected type.
        """
        if not score:
            return False

        score_type = score.score_type
        score_value = score.get_value()

        # Handle true_false scores
        if score_type == "true_false":
            return bool(score_value)

        # Handle numeric scores
        if score_type == "float_scale":
            return float(score_value) >= self._successful_objective_threshold

        # For other score types, assume not achieved
        return False

    def get_feedback(self, score: Score) -> Optional[str]:
        """
        Get feedback from a score for use in future prompts.

        Args:
            score (Optional[Score]): The score to get feedback from.

        Returns:
            Optional[str]: Feedback string, or None if no suitable feedback exists.
        """
        if not score or not self._use_score_as_feedback:
            return None

        return score.score_rationale

    @property
    def scorer_type(self) -> str:
        """
        Get the type of the scorer.

        Returns:
            str: The type identifier of the scorer.
        """
        return str(self._scorer.get_identifier()["__type__"])
