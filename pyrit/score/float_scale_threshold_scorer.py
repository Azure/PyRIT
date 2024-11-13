# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional
import uuid

from pyrit.models import PromptRequestPiece, Score
from pyrit.score.scorer import Scorer


class FloatScaleThresholdScorer(Scorer):
    """A scorer that applies a threshold to a float scale score to make it a true/false score."""

    def __init__(self, *, scorer: Scorer, threshold: float) -> None:
        self._scorer = scorer
        self._threshold = threshold

        if not scorer.scorer_type == "float_scale":
            raise ValueError("The scorer must be a float scale scorer")

        if threshold <= 0 or threshold >= 1:
            raise ValueError("The threshold must be between 0 and 1")

        self.scorer_type = "true_false"

    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        """Scores the piece using the underlying float-scale scorer and thresholds the resulting score.

        Args:
            request_response (PromptRequestPiece): The piece to score.
            task (str): The task based on which the text should be scored (the original attacker model's objective).

        Returns:
            list[Score]: The scores.
        """
        scores = await self._scorer.score_async(request_response, task=task)
        for score in scores:
            score_value = score.get_value()
            score.score_value = str(score_value >= self._threshold)
            if score_value > self._threshold:
                comparison_symbol = ">"
            if score_value < self._threshold:
                comparison_symbol = "<"
            if score_value == self._threshold:
                comparison_symbol = "="
            score.score_rationale = (
                f"Normalized scale score: {score_value} {comparison_symbol} threshold {self._threshold}\n"
                f"Rationale for scale score: {score.score_rationale}"
            )
            score.score_type = self.scorer_type
            score.id = uuid.uuid4()
            score.scorer_class_identifier = self.get_identifier()
            score.scorer_class_identifier["sub_identifier"] = self._scorer.get_identifier()
        self._memory.add_scores_to_memory(scores=scores)
        return scores

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> None:
        """Validates the request response for scoring."""
        self._scorer.validate(request_response, task=task)
