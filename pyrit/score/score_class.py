# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
import json
from typing import Dict, Literal
import uuid


ScorerType = Literal["true_false", "float_scale"]

class Score:

    id: int

    # The value the scorer ended up with; e.g. True (if bool) or 0 (if float_scale)
    score_value: str

    # The type of the scorer; e.g. "bool" or "float_scale"
    scorer_type: ScorerType

    # The type of the harms category (e.g. "hate" or "violence")
    score_category: str

    # Extra data the scorer provides around the rationale of the score
    score_rationale: str

    # Custom metadata a scorer might use
    metadata: str

    # The identifier of the scorer class, including relavent information
    # e.g. {"scorer_name": "SelfAskScorer", "classifier": "current_events.yml"}
    scorer_class_identifier: Dict[str, str]

    # This is the prompt_request_response_id that the score is scoring
    # Note a scorer can generate an additional request. This is NOT that, but
    # the request associated with what we're scoring.
    prompt_request_response_id: str




    def __init__(self,
                 score_value: str,
                 scorer_type: ScorerType,
                 score_category: str,
                 score_rationale: str,
                 metadata: str,
                 scorer_class_identifier: Dict[str, str],
                 prompt_request_response_id: str,
                 ):
        self.id = uuid.uuid4()

        self.score_value = score_value
        self.scorer_type = scorer_type
        self.score_category = score_category
        self.score_rationale = score_rationale
        self.metadata = metadata
        self.scorer_class_identifier = scorer_class_identifier
        self.prompt_request_response_id = prompt_request_response_id

