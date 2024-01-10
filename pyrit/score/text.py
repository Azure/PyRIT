# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.interfaces import SupportTextClassification
from pyrit.models import Score


class TextConversationTokenFinder(SupportTextClassification):
    def __init__(self, token: str):
        self._token = token

    def score_text(self, text: str) -> Score:
        if self._token in text:
            return Score(
                score_type="bool",
                score_value=True,
                score_description=f"Return True if the token {self._token} is found in the text",
                score_explanation=f"Found token {self._token} in text",
            )
        else:
            return Score(
                score_type="bool",
                score_value=False,
                score_description=f"Return True if the token {self._token} is found in the text",
                score_explanation=f"Did not find token {self._token} in text",
            )
