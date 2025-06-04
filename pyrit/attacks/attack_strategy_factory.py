# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional, Union, overload

from pyrit.attacks.base.attack_strategy import AttackStrategy
from pyrit.attacks.base.config import AttackConverterConfig, AttackScoringConfig
from pyrit.attacks.multi_turn.red_teaming import RedTeamingAttack
from pyrit.attacks.single_turn.prompt_injection import PromptInjectionAttack
from pyrit.models.seed_prompt import SeedPrompt
from pyrit.orchestrator.multi_turn.red_teaming_orchestrator import RTOSystemPromptPaths
from pyrit.prompt_normalizer.prompt_converter_configuration import (
    PromptConverterConfiguration,
)
from pyrit.prompt_normalizer.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.prompt_target.common.prompt_target import PromptTarget
from pyrit.score.scorer import Scorer


class AttackStrategyType(Enum):
    """Types of attack strategies"""

    # Single-turn Attacks
    PROMPT_INJECTION = "prompt_injection"

    # Multi-turn Attacks
    RED_TEAMING = "red_teaming"


class AttackStrategyFactory:
    """
    Factory for creating attack strategies using method overloading for type safety
    """

    @overload
    @classmethod
    def create(
        cls,
        *,
        # Overload Identifier (Red Teaming)
        strategy_type: Literal[AttackStrategyType.RED_TEAMING],
        # Red Teaming Specific Parameters
        objective_target: PromptTarget,
        objective_scorer: Scorer,
        adversarial_chat: PromptChatTarget,
        adversarial_chat_system_prompt_path: Union[str, Path] = RTOSystemPromptPaths.TEXT_GENERATION.value,
        adversarial_chat_seed_prompt: Union[str, SeedPrompt] = "How can I help you?",
        prompt_normalizer: Optional[PromptNormalizer] = None,
        use_score_as_feedback: bool = False,
    ) -> RedTeamingAttack: ...

    @overload
    @classmethod
    def create(
        cls,
        *,
        # Overload Identifier (Prompt Injection)
        strategy_type: Literal[AttackStrategyType.PROMPT_INJECTION],
        # Prompt Injection Specific Parameters
        objective_target: PromptTarget,
        objective_scorer: Optional[Scorer] = None,
        request_converter_configurations: Optional[List[PromptConverterConfiguration]] = None,
        response_converter_configurations: Optional[List[PromptConverterConfiguration]] = None,
        auxiliary_scorers: Optional[List[Scorer]] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,
    ) -> PromptInjectionAttack: ...

    @classmethod
    def create(
        cls,
        *,
        strategy_type: AttackStrategyType,
        **kwargs,
    ) -> AttackStrategy:
        """
        Create an attack strategy and its corresponding context based on the specified type and parameters

        Args:
            strategy_type (StrategyType): The type of strategy to create
            **kwargs: Additional arguments specific to the strategy type

        Returns:
            AttackStrategy: An instance of the specified attack strategy

        Raises:
            ValueError: If required arguments are missing or invalid
        """
        match strategy_type:
            case AttackStrategyType.RED_TEAMING:
                return cls._build_red_teaming_attack(**kwargs)
            case AttackStrategyType.PROMPT_INJECTION:
                return cls._build_prompt_injection_attack(**kwargs)
            case _:
                raise ValueError(f"Unknown strategy type: {strategy_type}")

    @classmethod
    def _build_red_teaming_attack(cls, **kwargs) -> RedTeamingAttack:
        """
        Build a RedTeamingAttack instance from the provided parameters.

        Args:
            **kwargs: Parameters for RedTeamingAttack

        Returns:
            RedTeamingAttack: Configured red teaming attack instance

        Raises:
            ValueError: If required parameters are missing
        """
        # Extract red teaming specific parameters
        objective_target = kwargs.get("objective_target")
        objective_scorer = kwargs.get("objective_scorer")
        adversarial_chat = kwargs.get("adversarial_chat")
        adversarial_chat_system_prompt_path = kwargs.get(
            "adversarial_chat_system_prompt_path", RTOSystemPromptPaths.TEXT_GENERATION.value
        )
        adversarial_chat_seed_prompt = kwargs.get("adversarial_chat_seed_prompt", "How can I help you?")
        prompt_normalizer = kwargs.get("prompt_normalizer")
        use_score_as_feedback = kwargs.get("use_score_as_feedback", False)

        # Validate required parameters
        if not objective_target or not objective_scorer or not adversarial_chat:
            raise ValueError("Red teaming attack requires objective_target, objective_scorer, and adversarial_chat")

        # Create configuration objects
        attack_scoring_config = AttackScoringConfig(use_score_as_feedback=use_score_as_feedback)

        return RedTeamingAttack(
            objective_target=objective_target,
            objective_scorer=objective_scorer,
            adversarial_chat=adversarial_chat,
            adversarial_chat_system_prompt_path=adversarial_chat_system_prompt_path,
            adversarial_chat_seed_prompt=adversarial_chat_seed_prompt,
            attack_scoring_config=attack_scoring_config,
            prompt_normalizer=prompt_normalizer,
        )

    @classmethod
    def _build_prompt_injection_attack(cls, **kwargs) -> PromptInjectionAttack:
        """
        Build a PromptInjectionAttack instance from the provided parameters.

        Args:
            **kwargs: Parameters for PromptInjectionAttack

        Returns:
            PromptInjectionAttack: Configured prompt injection attack instance

        Raises:
            ValueError: If required parameters are missing
        """
        # Extract prompt injection specific parameters
        objective_target = kwargs.get("objective_target")
        objective_scorer = kwargs.get("objective_scorer")
        request_converter_configurations = kwargs.get("request_converter_configurations")
        response_converter_configurations = kwargs.get("response_converter_configurations")
        auxiliary_scorers = kwargs.get("auxiliary_scorers")
        prompt_normalizer = kwargs.get("prompt_normalizer")

        # Validate required parameters
        if not objective_target:
            raise ValueError("Prompt injection attack requires objective_target")

        # Create configuration objects
        attack_converter_cfg = None
        if request_converter_configurations or response_converter_configurations:
            attack_converter_cfg = AttackConverterConfig(
                request_converters=request_converter_configurations or [],
                response_converters=response_converter_configurations or [],
            )

        attack_scoring_cfg = None
        if auxiliary_scorers:
            attack_scoring_cfg = AttackScoringConfig(auxiliary_scorers=auxiliary_scorers or [])

        return PromptInjectionAttack(
            objective_target=objective_target,
            objective_scorer=objective_scorer,
            attack_converter_cfg=attack_converter_cfg,
            attack_scoring_cfg=attack_scoring_cfg,
            prompt_normalizer=prompt_normalizer,
        )
