# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

from pyrit.models import PromptRequestPiece, Score
from pyrit.score.scorer import Scorer


class SubStringScorer(Scorer):
    """
    Scorer that checks if a given substring is present in the text.
    """

    def __init__(self, *, substring: str, category: str = "") -> None:

        self._substring = substring
        self._score_category = category
        self.scorer_type = "true_false"

    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:

        self.validate(request_response, task=task)

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

        self._memory.add_scores_to_memory(scores=score)
        return score

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        if request_response.converted_value_data_type != "text":
            raise ValueError("Expected text data type")
