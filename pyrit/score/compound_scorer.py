# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Literal, Optional

from pyrit.models import PromptRequestPiece, Score
from pyrit.score.scorer import Scorer

LogicalCombination = Literal["AND", "OR"]


class CompoundScorer(Scorer):
    """A scorer that aggregates other true_false scorers, either using AND or OR logic.

    It returns a single score of True or False based on the logical combination of the scores of the constituent
    scorers.

    Args:
        logical_combination: The logical operation to combine scores ("AND" or "OR")
        scorers: List of true_false scorers to combine
        score_category: Optional category for the score
    """

    def __init__(
        self, *, logical_combination: LogicalCombination, scorers: List[Scorer], score_category: Optional[str] = None
    ) -> None:
        self.scorer_type = "true_false"
        self._logical_combination = logical_combination
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

        return_score = Score(
            score_value=str(True),
            score_value_description=None,
            score_type=self.scorer_type,
            score_category=self._score_category,
            score_metadata=None,
            score_rationale="",
            scorer_class_identifier=identifier_dict,
            prompt_request_response_id=request_response.id,
            task=task,
        )

        sub_rationale = "\n\n".join(f"{score.scorer_class_identifier} {score.score_rationale}" for score in scores)

        # Determine final score and rationale based on logical combination
        if self._logical_combination == "AND":
            return_score.score_value = str(all(score.get_value() for score in scores))
            top_rationale = (
                "All constituent scorers returned True in an AND compound scorer."
                if return_score.get_value() is True
                else "At least one constituent scorer returned False in an AND compound scorer."
            )
        else:
            return_score.score_value = str(any(score.get_value() for score in scores))
            top_rationale = (
                "At least one constituent scorer returned True in an OR compound scorer."
                if return_score.get_value() is True
                else "All constituent scorers returned False in an OR compound scorer."
            )

        return_score.score_rationale = f"{top_rationale}\n\n{sub_rationale}"
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
