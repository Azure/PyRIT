# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

from pyrit.models import Score
from pyrit.models.message import Message
from pyrit.score.scorer import Scorer
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator
from pyrit.score.true_false.true_false_score_aggregator import (
    TrueFalseAggregatorFunc,
    TrueFalseScoreAggregator,
)


class TrueFalseScorer(Scorer):
    """
    Base class for scorers that return true/false binary scores.

    This scorer evaluates prompt responses and returns a single boolean score indicating
    whether the response meets a specific criterion. Multiple pieces in a request response
    are aggregated using a TrueFalseAggregatorFunc function (default: TrueFalseScoreAggregator.OR).
    """

    def __init__(
        self,
        *,
        validator: ScorerPromptValidator,
        score_aggregator: TrueFalseAggregatorFunc = TrueFalseScoreAggregator.OR,
    ) -> None:
        super().__init__(validator=validator)
        self._score_aggregator = score_aggregator

    def validate_return_scores(self, scores: list[Score]):
        if len(scores) != 1:
            raise ValueError("TrueFalseScorer should return exactly one score.")

        if scores[0].score_value.lower() not in ["true", "false"]:
            raise ValueError("TrueFalseScorer score value must be True or False.")

    async def _score_async(
        self, message: Message, *, objective: Optional[str] = None
    ) -> list[Score]:
        """
        Score the given request response asynchronously.

        For TrueFalseScorer, multiple piece scores are aggregated into a single true/false score.

        Args:
            message (Message): The message to score.
            objective (Optional[str]): The objective to evaluate against. Defaults to None.

        Returns:
            list[Score]: A list containing a single true/false Score object.
        """
        # Get individual scores for all supported pieces using base implementation logic
        score_list = await super()._score_async(message, objective=objective)

        if not score_list:
            # If no pieces matched (e.g., due to role filter), return False
            # Use the first request piece's ID (or original_prompt_id as fallback)
            first_piece = message.message_pieces[0]
            piece_id = first_piece.id or first_piece.original_prompt_id
            if piece_id is None:
                raise ValueError("Cannot create score: request piece has no id or original_prompt_id")

            return_score = Score(
                score_value=str(False).lower(),
                score_value_description="No pieces to score after filtering; returning false.",
                score_type="true_false",
                score_category=None,
                score_metadata=None,
                score_rationale="No supported pieces (possibly filtered by role).",
                scorer_class_identifier=self.get_identifier(),
                prompt_request_response_id=piece_id,
                objective=objective,
            )
            return [return_score]

        # Use score aggregator to combine multiple piece scores into a single score
        result = self._score_aggregator(score_list)

        # Use the prompt_request_response_id from the first score
        return_score = Score(
            score_value=str(result.value).lower(),
            score_value_description=result.description,
            score_type="true_false",
            score_category=result.category,
            score_metadata=result.metadata,
            score_rationale=result.rationale,
            scorer_class_identifier=self.get_identifier(),
            prompt_request_response_id=score_list[0].prompt_request_response_id,
            objective=objective,
        )

        return [return_score]
