# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from typing import Dict, Optional


from pyrit.models import PromptRequestPiece, Score
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.models.literals import ChatMessageRole
from pyrit.score.scorer import Scorer
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator
from pyrit.score.true_false.true_false_score_aggregator import OR_, TrueFalseScoreAggregator

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
        self,
        request_response: PromptRequestResponse,
        *,
        objective: Optional[str] = None,
        role_filter: Optional[ChatMessageRole] = None,
    ) -> list[Score]:
        """
        Score the given request response asynchronously.

        For TrueFalseScorer, the scoring is a single score. 
        """
        # If a role filter is provided and this response's role doesn't match, immediately return False
        if role_filter is not None and request_response.request_pieces:
            if request_response.request_pieces[0].role != role_filter:
                return_score = Score(
                    score_value=str(False).lower(),
                    score_value_description="Role did not match role_filter; returning false.",
                    score_type="true_false",
                    score_category=None,
                    score_metadata=None,
                    score_rationale=f"role={request_response.request_pieces[0].role} != role_filter={role_filter}",
                    scorer_class_identifier=self.get_identifier(),
                    prompt_request_response_id=request_response.request_pieces[0].original_prompt_id,
                    objective=objective,
                )
                return [return_score]

        # score the supported pieces
        # in the future, we may want to make this configurable. E.g. ensuring every piece returns true
        supported_pieces = self._get_supported_pieces(request_response)
        # Apply role filter to supported pieces when provided
        if role_filter is not None:
            supported_pieces = [p for p in supported_pieces if p.role == role_filter]

        tasks = [
            self._score_piece_async(request_piece=piece, objective=objective)
            for piece in supported_pieces
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
            scorer_class_identifier= self.get_identifier(),
            prompt_request_response_id=request_response.request_pieces[0].id,
            objective=objective,
        )

        return [return_score]