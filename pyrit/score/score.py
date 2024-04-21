from dataclasses import dataclass
from typing import Literal


@dataclass
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