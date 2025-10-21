# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

from pyrit.models import MessagePiece, Score
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator
from pyrit.score.true_false.true_false_score_aggregator import (
    TrueFalseAggregatorFunc,
    TrueFalseScoreAggregator,
)
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
        categories: Optional[list[str]] = None,
        aggregator: TrueFalseAggregatorFunc = TrueFalseScoreAggregator.OR,
        validator: Optional[ScorerPromptValidator] = None,
    ) -> None:
        """Initialize the SubStringScorer.

        Args:
            substring (str): The substring to search for in the text.
            categories (Optional[list[str]]): Optional list of categories for the score. Defaults to None.
            aggregator (TrueFalseAggregatorFunc): The aggregator function to use.
                Defaults to TrueFalseScoreAggregator.OR.
            validator (Optional[ScorerPromptValidator]): Custom validator. Defaults to None.
        """
        super().__init__(score_aggregator=aggregator, validator=validator or self._default_validator)
        self._substring = substring
        self._score_categories = categories if categories else []

    async def _score_piece_async(self, message_piece: MessagePiece, *, objective: Optional[str] = None) -> list[Score]:
        """Score the given message piece based on presence of the substring.

        Args:
            message_piece (MessagePiece): The message piece to score.
            objective (Optional[str]): The objective to evaluate against. Defaults to None.
                Currently not used for this scorer.

        Returns:
            list[Score]: A list containing a single Score object with a boolean value indicating
                whether the substring is present in the text.
        """
        expected_output_substring_present = self._substring in message_piece.converted_value

        score = [
            Score(
                score_value=str(expected_output_substring_present),
                score_value_description="",
                score_metadata=None,
                score_type="true_false",
                score_category=self._score_categories,
                score_rationale="",
                scorer_class_identifier=self.get_identifier(),
                message_piece_id=message_piece.id,
                objective=objective,
            )
        ]

        return score
