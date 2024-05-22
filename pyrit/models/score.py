# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datetime import datetime
from typing import Dict, Literal, Optional, get_args
import uuid


ScoreType = Literal["true_false", "float_scale"]


class Score:

    id: uuid.UUID | str

    # The value the scorer ended up with; e.g. True (if true_false) or 0 (if float_scale)
    score_value: str

    # Value that can include a description of the score value
    score_value_description: str

    # The type of the scorer; e.g. "true_false" or "float_scale"
    score_type: ScoreType

    # The type of the harms category (e.g. "hate" or "violence")
    score_category: str

    # Extra data the scorer provides around the rationale of the score
    score_rationale: str

    # Custom metadata a scorer might use. This is left undefined other than for the
    # specific scorer that uses it.
    score_metadata: str

    # The identifier of the scorer class, including relavent information
    # e.g. {"scorer_name": "SelfAskScorer", "classifier": "current_events.yml"}
    scorer_class_identifier: Dict[str, str]

    # This is the prompt_request_response_id that the score is scoring
    # Note a scorer can generate an additional request. This is NOT that, but
    # the request associated with what we're scoring.
    prompt_request_response_id: uuid.UUID | str

    # Timestamp of when the score was created
    timestamp: datetime

    def __init__(
        self,
        *,
        id: Optional[uuid.UUID] = None,
        score_value: str,
        score_value_description: str,
        score_type: ScoreType,
        score_category: str,
        score_rationale: str,
        score_metadata: str,
        scorer_class_identifier: Dict[str, str] = None,
        prompt_request_response_id: str | uuid.UUID,
        date_time: Optional[datetime] = datetime.now(),
    ):
        self.id = id if id else uuid.uuid4()

        self._validate(score_type, score_value)

        self.score_value = score_value
        self.score_value_description = score_value_description

        if score_type not in get_args(ScoreType):
            raise ValueError(f"Score type {score_type} is not a valid score type.")

        self.score_type = score_type
        self.score_category = score_category
        self.score_rationale = score_rationale
        self.score_metadata = score_metadata
        self.scorer_class_identifier = scorer_class_identifier
        self.prompt_request_response_id = prompt_request_response_id
        self.date_time = date_time

    def get_value(self):
        """
        Returns the value of the score based on its type.

        If the score type is "true_false", it returns True if the score value is "true" (case-insensitive),
        otherwise it returns False.

        If the score type is "float_scale", it returns the score value as a float.

        Raises:
            ValueError: If the score type is unknown.

        Returns:
            The value of the score based on its type.
        """
        if self.score_type == "true_false":
            return self.score_value.lower() == "true"
        elif self.score_type == "float_scale":
            return float(self.score_value)

        raise ValueError(f"Unknown scorer type: {self.score_type}")

    def __str__(self):
        if self.scorer_class_identifier:
            return f"{self.scorer_class_identifier['__type__']}: {self.score_category}: {self.score_value}"
        return f": {self.score_category}: {self.score_value}"

    def _validate(self, scorer_type, score_value):
        if scorer_type == "true_false" and str(score_value).lower() not in ["true", "false"]:
            raise ValueError(f"True False scorers must have a score value of 'true' or 'false' not {score_value}")
        elif scorer_type == "float_scale":
            try:
                score = float(score_value)
                if not (0 <= score <= 1):
                    raise ValueError(f"Float scale scorers must have a score value between 0 and 1. Got {score_value}")
            except ValueError:
                raise ValueError(f"Float scale scorers require a numeric score value. Got {score_value}")
