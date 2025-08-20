# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from typing import Optional

from pyrit.models import PromptRequestPiece, Score
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.score.scorer import Scorer
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer


class TrueFalseInverterScorer(TrueFalseScorer):
    """A scorer that inverts a true false score."""

    def __init__(self, *, scorer: TrueFalseScorer, validator: Optional[ScorerPromptValidator] = None) -> None:
        if (type(scorer) is not TrueFalseScorer):
            raise ValueError("The scorer must be a true false scorer")
        self._scorer = scorer


    async def _score_async(self, request_response: PromptRequestResponse, *, objective: Optional[str] = None) -> list[Score]:
        """Scores the piece using the underlying true-false scorer and returns the opposite score.

        Args:
            request_response (PromptRequestPiece): The piece to score.
            task (str): The task based on which the text should be scored (the original attacker model's objective).

        Returns:
            list[Score]: The scores.
        """
        scores = await self._scorer.score_async(request_response, objective=objective)

        # TrueFalseScorers only have a single score
        score = scores[0]

        score.score_value = str(True) if not score.get_value() else str(False)
        score.score_value_description = "Inverted score: " + str(score.score_value_description)
        score.score_rationale = f"Inverted score: {score.score_value}\n{score.score_rationale}"

        score.id = uuid.uuid4()

        score.scorer_class_identifier = self.get_identifier()
        score.scorer_class_identifier["sub_identifier"] = self._scorer.get_identifier()

        return [score]

