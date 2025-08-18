# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

from pyrit.models import PromptRequestPiece, Score
from pyrit.score.scorer import Scorer
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator
from pyrit.score.true_false.true_false_score_aggregator import TrueFalseScoreAggregator
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer


class SubStringScorer(TrueFalseScorer):
    """Scorer that checks if a given substring is present in the text.

    This scorer performs simple substring matching to determine if specific
    text patterns are present in the response. It's useful for detecting
    specific phrases, keywords, or patterns in generated content.
    """

    _default_validator: ScorerPromptValidator = ScorerPromptValidator(supported_data_types=["text"])

    def __init__(
            self,
            *,
            substring: str,
            category: str = "",
            aggregator: TrueFalseScoreAggregator,
            validator: Optional[ScorerPromptValidator] = None
        ) -> None:
        """Initialize the SubStringScorer.

        Args:
            substring: The substring to search for in the text.
            category: Optional category for the score. Defaults to empty string.
        """
        super().__init__(score_aggregator=aggregator, validator=validator or self._default_validator)
        self._substring = substring
        self._score_category = category

    async def _score_piece_async(self, request_piece: PromptRequestPiece, *, objective: Optional[str] = None) -> list[Score]:
        """Score the given request_response based on presence of the substring.

        Args:
            request_response: The piece to score.
            task: The task based on which the text should be scored (unused).

        Returns:
            A list containing a single Score object with a boolean value indicating
            whether the substring is present in the text.
        """
        expected_output_substring_present = self._substring in request_piece.converted_value

        score = [
            Score(
                score_value=str(expected_output_substring_present),
                score_value_description="",
                score_metadata=None,
                score_type="true_false",
                score_category=self._score_category,
                score_rationale="",
                scorer_class_identifier=self.get_identifier(),
                prompt_request_response_id=request_piece.id,
                objective=objective,
            )
        ]

        return score

