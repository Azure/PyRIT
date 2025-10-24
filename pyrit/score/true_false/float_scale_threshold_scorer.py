# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from typing import Optional

from pyrit.models import ChatMessageRole, Message, MessagePiece, Score
from pyrit.score.float_scale.float_scale_score_aggregator import (
    FloatScaleAggregatorFunc,
    FloatScaleScoreAggregator,
)
from pyrit.score.float_scale.float_scale_scorer import FloatScaleScorer
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer


class FloatScaleThresholdScorer(TrueFalseScorer):
    """A scorer that applies a threshold to a float scale score to make it a true/false score."""

    def __init__(
        self,
        *,
        scorer: FloatScaleScorer,
        threshold: float,
        float_scale_aggregator: FloatScaleAggregatorFunc = FloatScaleScoreAggregator.MAX,
    ) -> None:
        """Initialize the FloatScaleThresholdScorer.

        Args:
            scorer (FloatScaleScorer): The underlying float scale scorer to use.
            threshold (float): The threshold value between 0 and 1. Scores >= threshold are True, otherwise False.
            float_scale_aggregator (FloatScaleAggregatorFunc): The aggregator function to use for combining
                multiple float scale scores. Defaults to FloatScaleScoreAggregator.MAX.
        """
        self._scorer = scorer
        self._threshold = threshold
        self._float_scale_aggregator = float_scale_aggregator

        # Validation is used by sub-scorers
        super().__init__(validator=ScorerPromptValidator())

        if threshold <= 0 or threshold >= 1:
            raise ValueError("The threshold must be between 0 and 1")

    async def _score_async(
        self,
        message: Message,
        *,
        objective: Optional[str] = None,
        role_filter: Optional[ChatMessageRole] = None,
    ) -> list[Score]:
        """Scores the piece using the underlying float-scale scorer and thresholds the resulting score.

        Args:
            message (Message): The message to score.
            objective (Optional[str]): The objective to evaluate against (the original attacker model's objective).
                Defaults to None.
            role_filter (Optional[ChatMessageRole]): Optional filter for message roles. Defaults to None.

        Returns:
            list[Score]: A list containing a single true/false Score object based on the threshold comparison.
        """
        scores = await self._scorer.score_async(
            message,
            objective=objective,
            role_filter=role_filter,
        )

        # Aggregator now returns a list of results
        aggregate_results = self._float_scale_aggregator(scores)
        # For threshold scoring, we expect a single aggregated result
        aggregate_score = aggregate_results[0]

        score = scores[0]
        score.score_type = "true_false"

        aggregate_value = aggregate_score.value

        score.score_value = str(aggregate_value >= self._threshold)
        if aggregate_value > self._threshold:
            comparison_symbol = ">"
        elif aggregate_value < self._threshold:
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
        return [score]

    async def _score_piece_async(self, message_piece: MessagePiece, *, objective: Optional[str] = None) -> list[Score]:
        """Float Scale scorers do not support piecewise scoring.

        Args:
            message_piece (MessagePiece): Unused.
            objective (Optional[str]): Unused.

        Raises:
            NotImplementedError: Always, since composite scoring operates at the response level.
        """
        raise NotImplementedError("TrueFalseCompositeScorer does not support piecewise scoring.")

    def _get_sub_identifier(self):
        """
        Returns the identifier of the underlying float scale scorer.

        Returns:
            dict: The identifier dictionary of the wrapped scorer.
        """
        return self._scorer.get_identifier()
