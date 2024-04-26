# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Dict, Literal


class Score:

    # The data type of the score value
    score_type: Literal["int", "float", "str", "bool"]
    # The score value
    score_value: int | float | str | bool
    # A description of the meaning of the score value
    score_description: str = ""
    # An explanation of how the score was calculated
    score_explanation: str = ""
    # The raw input text that was scored
    raw_input_score_text: str = ""
    # The raw output of the scoring engine that was used to generate the score
    raw_output_score_text: str = ""

    id: int
    # The value the scorer ended up with; e.g. "No hate speech"
    score_value: str

    # Extra data the scorer provides around the rationale of the score
    score_explanation: str

    # Metadata a scorer might use
    metadata: str

    # The class of the scorer
    scoring_class_identifier: str

    # The criteria the scorer used to determine a score; e.g. "current_events.yaml"
    scoring_criteria: str

    # Optional: the request used for scoring
    prompt_request_id: str

    prompt_response_id: str




    def __init__(self,
                 id: int,
                 score_value: str,
                 scoring_class_identifier: Dict[str, str],
                 scoring_criteria: str,
                 ):
        self.id
