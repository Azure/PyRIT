# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.score import Score, Scorer


class SubStringScorer(Scorer):
    """
    Scorer that checks if a given substring is present in the text.
    """

    def __init__(self, *, substring: str, category: str = None) -> None:
        super().__init__()
        self._substring = substring
        self._category = category
        self.scorer_type = "true_false"

    async def score_async(self, request_response: PromptRequestPiece) -> list[Score]:

        await asyncio.sleep(0)

        self.validate(request_response)

        expected_output_substring_present = self._substring in request_response.converted_value

        return [
            Score(
                score_value=str(expected_output_substring_present),
                score_value_description=None,
                score_metadata=None,
                score_type=self.scorer_type,
                score_category=self._category,
                score_rationale=None,
                scorer_class_identifier=self.get_identifier(),
                prompt_request_response_id=request_response.id,
            )
        ]

    def validate(self, request_response: PromptRequestPiece):
        if request_response.converted_value_data_type != "text":
            raise ValueError("Expected text data type")
