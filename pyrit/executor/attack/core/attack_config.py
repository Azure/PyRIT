# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from pyrit.executor.core import StrategyConverterConfig
from pyrit.models import SeedPrompt
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.score import Scorer, TrueFalseScorer


@dataclass
class AttackAdversarialConfig:
    """
    Adversarial configuration for attacks that involve adversarial chat targets.

    This class defines the configuration for attacks that utilize an adversarial chat target,
    including the target chat model, system prompt, and seed prompt for the attack.
    """

    # Adversarial chat target for the attack
    target: PromptChatTarget

    # Path to the YAML file containing the system prompt for the adversarial chat target
    system_prompt_path: Optional[Union[str, Path]] = None

    # Seed prompt for the adversarial chat target
    seed_prompt: Union[str, SeedPrompt] = "How can I help you?"


@dataclass
class AttackScoringConfig:
    """
    Scoring configuration for evaluating attack effectiveness.

    This class defines the scoring components used to evaluate attack effectiveness,
    detect refusals, and perform auxiliary scoring operations.
    """

    # Primary scorer for evaluating attack effectiveness
    objective_scorer: Optional[TrueFalseScorer] = None

    # Refusal scorer for detecting refusals or non-compliance
    refusal_scorer: Optional[TrueFalseScorer] = None

    # Additional scorers for auxiliary metrics or custom evaluations
    auxiliary_scorers: List[Scorer] = field(default_factory=list)

    # Whether to use scoring results as feedback for iterative attacks
    use_score_as_feedback: bool = True

    # Threshold for considering an objective successful [0.0 to 1.0]
    # A value of 1.0 means the objective must be fully achieved, while 0.0 means any score is acceptable.
    # Only applies to float_scale scorers
    successful_objective_threshold: float = 0.8

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.successful_objective_threshold <= 1.0:
            raise ValueError(
                f"successful_objective_threshold must be between 0.0 and 1.0, "
                f"got {self.successful_objective_threshold}"
            )

        # Enforce objective scorer type: must be a TrueFalseScorer if provided
        if self.objective_scorer and not isinstance(self.objective_scorer, TrueFalseScorer):
            raise ValueError("Objective scorer must be a TrueFalseScorer")

        # Enforce refusal scorer type: must be a TrueFalseScorer if provided
        if self.refusal_scorer and not isinstance(self.refusal_scorer, TrueFalseScorer):
            raise ValueError("Refusal scorer must be a TrueFalseScorer")


@dataclass
class AttackConverterConfig(StrategyConverterConfig):
    """
    Configuration for prompt converters used in attacks.

    This class defines the converter configurations that transform prompts
    during the attack process, both for requests and responses.
    """
