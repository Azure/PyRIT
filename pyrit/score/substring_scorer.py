# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.score import Score, SupportTextClassification


class SubStringScorer(SupportTextClassification):
    def __init__(self, *, expected_output_substring: str) -> None:
        super().__init__()
        self._expected_output_substring = expected_output_substring

    def score_text(self, text: str) -> Score:
        expected_output_substring_present = self._expected_output_substring in text
        optional_not_text = "not " if not expected_output_substring_present else ""
        return Score(
            score_type="bool",
            score_value=expected_output_substring_present,
            score_description=f"The expected output substring is {optional_not_text}present in the text.",
            score_explanation=f"The expected output substring {self._expected_output_substring} is "
            f"{optional_not_text}present in the text:\n{text}",
        )
