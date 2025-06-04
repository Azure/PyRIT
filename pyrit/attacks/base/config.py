# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass, field
from typing import List

from pyrit.prompt_normalizer.prompt_converter_configuration import (
    PromptConverterConfiguration,
)
from pyrit.score.scorer import Scorer


@dataclass
class AttackScoringConfig:
    """
    Configuration for scoring attacks in PyRIT.

    This class defines the scoring components used to evaluate attack effectiveness,
    detect refusals, and perform auxiliary scoring operations.
    """

    # Additional scorers for auxiliary metrics or custom evaluations
    auxiliary_scorers: List[Scorer] = field(default_factory=list)

    # Whether to use scoring results as feedback for iterative attacks
    use_score_as_feedback: bool = False

    # Threshold for considering an objective achieved (0.0 to 1.0)
    # Only applies to float_scale scorers
    objective_achieved_score_threshold: float = 0.8
    
    def __post_init__(self):
        """Validate configuration values."""
        if not 0.0 <= self.objective_achieved_score_threshold <= 1.0:
            raise ValueError(
                f"objective_achieved_score_threshold must be between 0.0 and 1.0, "
                f"got {self.objective_achieved_score_threshold}"
            )


@dataclass
class AttackConverterConfig:
    """
    Configuration for prompt converters used in attacks.

    This class defines the converter configurations that transform prompts
    during the attack process, both for requests and responses.
    """

    # List of converter configurations to apply to attack requests/prompts
    request_converters: List[PromptConverterConfiguration] = field(default_factory=list)

    # List of converter configurations to apply to target responses
    response_converters: List[PromptConverterConfiguration] = field(default_factory=list)
