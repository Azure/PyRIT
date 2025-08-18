# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from typing import Optional

from pyrit.models import PromptRequestPiece, Score
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.score.float_scale.float_scale_score_aggregator import MAX_, FloatScaleScoreAggregator
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer
from pyrit.score.float_scale.float_scale_scorer import FloatScaleScorer


class FloatScaleThresholdScorer(TrueFalseScorer):
    """A scorer that applies a threshold to a float scale score to make it a true/false score."""

    def __init__(
        self,
        *,
        scorer: FloatScaleScorer,
        threshold: float,
        float_scale_aggregator: FloatScaleScoreAggregator = MAX_,

    ) -> None:
        self._scorer = scorer
        self._threshold = threshold
        self._float_scale_aggregator = float_scale_aggregator

        # Validation is used by sub-scorers
        super().__init__(validator=ScorerPromptValidator())

        if threshold <= 0 or threshold >= 1:
            raise ValueError("The threshold must be between 0 and 1")

    async def _score_async(self, request_response: PromptRequestResponse, *, objective: Optional[str] = None) -> list[Score]:
        """Scores the piece using the underlying float-scale scorer and thresholds the resulting score.

        Args:
            request_response (PromptRequestResponse): The piece to score.
            objective (str): The task based on which the text should be scored (the original attacker model's objective).

        Returns:
            list[Score]: The score.
        """
        scores = await self._scorer.score_async(request_response, objective=objective)

        aggregate_score = self._float_scale_aggregator(scores)

        score = scores[0]
        score.score_type = "true_false"

        aggregate_value = aggregate_score.value


        score.score_value = str(aggregate_value >= self._threshold)
        if aggregate_value > self._threshold:
            comparison_symbol = ">"
        if aggregate_value < self._threshold:
            comparison_symbol = "<"
        else:
            comparison_symbol = "="
        score.score_rationale = (
            f"Normalized scale score: {aggregate_value} {comparison_symbol} threshold {self._threshold}\n"
            f"Rationale for scale score: {score.score_rationale}"
        )

        score.score_value_description = aggregate_score.description

        score.id = uuid.uuid4()
        score.scorer_class_identifier = self.get_identifier()
        score.scorer_class_identifier["sub_identifier"] = self._scorer.get_identifier()
        return scores
