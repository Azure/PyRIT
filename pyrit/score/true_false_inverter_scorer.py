# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional
import uuid

from pyrit.models import PromptRequestPiece, Score
from pyrit.score.scorer import Scorer


class TrueFalseInverterScorer(Scorer):
    """A scorer that inverts a true false score."""

    def __init__(self, *, scorer: Scorer) -> None:
        self._scorer = scorer

        if not scorer.scorer_type == "true_false":
            raise ValueError("The scorer must be a true false scorer")

        self.scorer_type = "true_false"

    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        """Scores the piece using the underlying true-false scorer and returns the opposite score.

        Args:
            request_response (PromptRequestPiece): The piece to score.
            task (str): The task based on which the text should be scored (the original attacker model's objective).

        Returns:
            list[Score]: The scores.
        """
        scores = await self._scorer.score_async(request_response, task=task)
        for score in scores:
            score.score_value = str(True) if not score.get_value() else str(False)
            score.score_value_description = "Inverted score: " + str(score.score_value_description)
            score.score_rationale = f"Inverted score: {score.score_value}\n{score.score_rationale}"

            score.id = uuid.uuid4()

            score.scorer_class_identifier = self.get_identifier()
            score.scorer_class_identifier["sub_identifier"] = self._scorer.get_identifier()

        self._memory.add_scores_to_memory(scores=scores)
        return scores

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> None:
        """Validates the request response for scoring."""
        self._scorer.validate(request_response, task=task)
