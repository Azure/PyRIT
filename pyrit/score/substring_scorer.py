# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.score import Score, TrueFalseScorer


class SubStringScorer(TrueFalseScorer):
    """
    Scorer that checks if a given substring is present in the text.
    """
    def __init__(self, *, expected_output_substring: str, category:str = None) -> None:
        super().__init__()
        self._expected_output_substring = expected_output_substring
        self._category = category

    def score(self, request_response: PromptRequestPiece) -> list[Score]:

        self.validate(request_response)

        expected_output_substring_present = self._expected_output_substring \
            in request_response.converted_value

        return [
            Score(
                score_value=str(expected_output_substring_present),
                scorer_type=self._score_type,
                score_category=self._category,
                score_rationale=None,
                scorer_class_identifier=self.get_identifier(),
                prompt_request_response_id=request_response.id,
            )
        ]

    def validate(self, request_response: PromptRequestPiece):
        if request_response.converted_value_data_type != "text":
            raise ValueError("Expected text data type")
