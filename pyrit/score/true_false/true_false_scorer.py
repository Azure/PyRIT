# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from typing import Dict, Optional

from pyrit.models import PromptRequestPiece, Score
from pyrit.models.literals import ChatMessageRole
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.score.scorer import Scorer
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator
from pyrit.score.true_false.true_false_score_aggregator import (
    OR_,
    TrueFalseScoreAggregator,
)


class TrueFalseScorer(Scorer):

    def __init__(self, *, validator: ScorerPromptValidator, score_aggregator: TrueFalseScoreAggregator = OR_) -> None:
        super().__init__(validator=validator)
        self._score_aggregator = score_aggregator

    def validate_return_scores(self, scores: list[Score]):
        if len(scores) != 1:
            raise ValueError("TrueFalseScorer should return exactly one score.")

        if scores[0].score_value.lower() not in ["true", "false"]:
            raise ValueError("TrueFalseScorer score value must be True or False.")

    async def _score_async(
        self, request_response: PromptRequestResponse, *, objective: Optional[str] = None
    ) -> list[Score]:
        """
        Score the given request response asynchronously.

        For TrueFalseScorer, the scoring is a single score.
        """

        tasks = [
            self._score_piece_async(request_piece=piece, objective=objective)
            for piece in request_response.request_pieces
        ]

        if not tasks:
            # If no pieces matched (e.g., due to role filter), return False
            return_score = Score(
                score_value=str(False).lower(),
                score_value_description="No pieces to score after filtering; returning false.",
                score_type="true_false",
                score_category=None,
                score_metadata=None,
                score_rationale="No supported pieces (possibly filtered by role).",
                scorer_class_identifier=self.get_identifier(),
                prompt_request_response_id=request_response.request_pieces[0].original_prompt_id,
                objective=objective,
            )
            return [return_score]

        # Run all piece-level scorings concurrently
        piece_score_lists = await asyncio.gather(*tasks)

        # Use score aggregator to return a single score
        score_list = [score for sublist in piece_score_lists for score in sublist]

        if len(score_list) == 0:
            raise ValueError("No scores were generated from the request response pieces.")

        result = self._score_aggregator(score_list)

        return_score = Score(
            score_value=str(result.value).lower(),
            score_value_description=result.description,
            score_type="true_false",
            score_category=result.category,
            score_metadata=result.metadata,
            score_rationale=result.rationale,
            scorer_class_identifier=self.get_identifier(),
            prompt_request_response_id=request_response.request_pieces[0].id,
            objective=objective,
        )

        return [return_score]
