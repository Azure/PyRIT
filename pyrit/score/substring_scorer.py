# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

from pyrit.models import PromptRequestPiece, Score
from pyrit.score.scorer import Scorer


class SubStringScorer(Scorer):
    """Scorer that checks if a given substring is present in the text.

    This scorer performs simple substring matching to determine if specific
    text patterns are present in the response. It's useful for detecting
    specific phrases, keywords, or patterns in generated content.
    """

    def __init__(self, *, substring: str, category: str = "") -> None:
        """Initialize the SubStringScorer.

        Args:
            substring: The substring to search for in the text.
            category: Optional category for the score. Defaults to empty string.
        """
        self._substring = substring
        self._score_category = category
        self.scorer_type = "true_false"

    async def _score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        """Score the given request_response based on presence of the substring.

        Args:
            request_response: The piece to score.
            task: The task based on which the text should be scored (unused).

        Returns:
            A list containing a single Score object with a boolean value indicating
            whether the substring is present in the text.
        """
        expected_output_substring_present = self._substring in request_response.converted_value

        score = [
            Score(
                score_value=str(expected_output_substring_present),
                score_value_description=None,
                score_metadata=None,
                score_type=self.scorer_type,
                score_category=self._score_category,
                score_rationale=None,
                scorer_class_identifier=self.get_identifier(),
                prompt_request_response_id=request_response.id,
                task=task,
            )
        ]

        return score

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        if request_response.converted_value_data_type != "text":
            raise ValueError("Expected text data type")
