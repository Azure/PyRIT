# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from typing import List, Optional

from pyrit.models import PromptRequestPiece, Score
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.models.literals import ChatMessageRole
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator
from pyrit.score.true_false.true_false_score_aggregator import TrueFalseScoreAggregator
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer


class TrueFalseCompositeScorer(TrueFalseScorer):
    """Composite true/false scorer that aggregates results from other true/false scorers.

    This scorer invokes a collection of constituent ``TrueFalseScorer`` instances and
    reduces their single-score outputs into one final true/false score using the supplied
    aggregation function (e.g., ``AND_``, ``OR_``, ``MAJORITY_``).
    """


    def __init__(
        self,
        *,
        aggregator: TrueFalseScoreAggregator,
        scorers: List[TrueFalseScorer],
    ) -> None:
        """Initialize the composite scorer.

        Args:
            aggregator (TrueFalseScoreAggregator): Aggregation function to combine child scores
                (e.g., ``AND_``, ``OR_``, ``MAJORITY_``).
            scorers (List[TrueFalseScorer]): The constituent true/false scorers to invoke.
        """
        # Initialize base with the selected aggregator used by TrueFalseScorer logic
        # Validation is used by sub-scorers
        super().__init__(score_aggregator=aggregator, validator=ScorerPromptValidator())


        if not scorers:
            raise ValueError("At least one scorer must be provided.")

        for scorer in scorers:
            if not isinstance(scorer, TrueFalseScorer):
                raise ValueError("All scorers must be true_false scorers.")

        self._scorers = scorers

    async def _score_async(
        self,
        request_response: PromptRequestResponse,
        *,
        objective: Optional[str] = None,
        role_filter: Optional[ChatMessageRole] = None,
    ) -> list[Score]:
        """Score a request/response by combining results from all constituent scorers.

        Args:
            request_response (PromptRequestResponse): The request/response to score.
            objective (Optional[str]): Scoring objective or context.

        Returns:
            list[Score]: A single-element list with the aggregated true/false score.
        """

        tasks = [
            scorer.score_async(request_response=request_response, objective=objective, role_filter=role_filter)
            for scorer in self._scorers
        ]
    
        # Run all response scorings concurrently
        score_list = await asyncio.gather(*tasks)

        for score in score_list:
            if len(score) != 1:
                raise ValueError("Each TrueFalseScorer must return exactly one score.")

        # Use score aggregator to return a single score
        score_list = [score[0] for score in score_list]

        if len(score_list) == 0:
            raise ValueError("No scores were generated from the request response pieces.")

        result = self._score_aggregator(score_list)

        return_score = Score(
            score_value=str(result.value),
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

    async def _score_piece_async(self, request_piece: PromptRequestPiece, *, objective: Optional[str] = None) -> list[Score]:
        """Composite scorers do not support piecewise scoring.

        Args:
            request_piece (PromptRequestPiece): Unused.
            objective (Optional[str]): Unused.

        Raises:
            NotImplementedError: Always, since composite scoring operates at the response level.
        """
        raise NotImplementedError("TrueFalseCompositeScorer does not support piecewise scoring.")

    

