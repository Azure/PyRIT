# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Optional

from pyrit.models import PromptRequestPiece, Score
from pyrit.score.score_aggregator import ScoreAggregator
from pyrit.score.scorer import Scorer


class CompositeScorer(Scorer):
    """A scorer that aggregates other true_false scorers using a specified aggregation function.

    It returns a single score of True or False based on the aggregation of the scores of the constituent
    scorers.

    Args:
        aggregator: The aggregation function to use (e.g. `AND_`, `OR_`, `MAJORITY_`)
        scorers: List of true_false scorers to combine
        score_category: Optional category for the score
    """

    def __init__(
        self, *, aggregator: ScoreAggregator, scorers: List[Scorer], score_category: Optional[str] = None
    ) -> None:
        self.scorer_type = "true_false"
        self._aggregator = aggregator
        self._score_category = score_category

        if not scorers:
            raise ValueError("At least one scorer must be provided.")

        for scorer in scorers:
            if scorer.scorer_type != "true_false":
                raise ValueError("All scorers must be true_false scorers.")

        self._scorers = scorers

    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> List[Score]:
        """Scores the request response by combining results from all constituent scorers.

        Args:
            request_response: The request response to be scored
            task: Optional task description for scoring context

        Returns:
            List containing a single Score object representing the combined result
        """
        self.validate(request_response, task=task)
        scores = await self._score_all_async(request_response, task=task)

        identifier_dict = self.get_identifier()
        identifier_dict["sub_identifier"] = [scorer.get_identifier() for scorer in self._scorers]

        result = self._aggregator(scores)

        return_score = Score(
            score_value=str(result.value),
            score_value_description=None,
            score_type=self.scorer_type,
            score_category=self._score_category,
            score_metadata=None,
            score_rationale=result.rationale,
            scorer_class_identifier=identifier_dict,
            prompt_request_response_id=request_response.id,
            task=task,
        )

        return [return_score]

    async def _score_all_async(
        self, request_response: PromptRequestPiece, *, task: Optional[str] = None
    ) -> List[Score]:
        """Scores the request_response using all constituent scorers sequentially.

        Args:
            request_response: The request response to be scored
            task: Optional task description for scoring context

        Returns:
            List of scores from all constituent scorers
        """
        if not self._scorers:
            return []

        all_scores = []
        for scorer in self._scorers:
            scores = await scorer.score_async(request_response=request_response, task=task)
            all_scores.extend(scores)

        return all_scores

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> None:
        """Validates the request response for scoring.

        Args:
            request_response: The request response to validate
            task: Optional task description for validation context
        """
        pass
