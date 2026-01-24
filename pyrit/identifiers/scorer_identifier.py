# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pyrit.identifiers.identifier import Identifier, MAX_STORAGE_LENGTH
from pyrit.models.score import ScoreType


@dataclass(frozen=True)
class ScorerIdentifier(Identifier):
    """
    Identifier for Scorer instances.

    This frozen dataclass extends Identifier with scorer-specific fields.
    Long prompt templates are automatically truncated for storage display.

    Attributes:
        scorer_type: The type of scorer ("true_false", "float_scale", or "unknown").
        system_prompt_template: The system prompt template used by the scorer.
            Truncated for storage if > 100 characters.
        user_prompt_template: The user prompt template used by the scorer.
            Truncated for storage if > 100 characters.
        sub_identifier: List of sub-scorer identifiers for composite scorers.
        target_info: Information about the prompt target used by the scorer.
        score_aggregator: The name of the score aggregator function.
        scorer_specific_params: Additional scorer-specific parameters.
    """

    scorer_type: ScoreType = "unknown"
    system_prompt_template: Optional[str] = field(default=None, metadata={MAX_STORAGE_LENGTH: 100})
    user_prompt_template: Optional[str] = field(default=None, metadata={MAX_STORAGE_LENGTH: 100})
    sub_identifier: Optional[List[Dict[str, Any]]] = None
    target_info: Optional[Dict[str, Any]] = None
    score_aggregator: Optional[str] = None
    scorer_specific_params: Optional[Dict[str, Any]] = None
