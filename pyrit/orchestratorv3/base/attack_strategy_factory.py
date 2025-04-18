# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional, Union, overload

from pyrit.models.seed_prompt import SeedPrompt
from pyrit.orchestrator.multi_turn.red_teaming_orchestrator import RTOSystemPromptPaths
from pyrit.orchestratorv3.base.attack_strategy import (
    AttackStrategy,
    BacktrackingStrategy,
)
from pyrit.orchestratorv3.base.core import MultiTurnAttackContext
from pyrit.orchestratorv3.multi_turn.red_teaming import RedTeamingStrategy
from pyrit.orchestratorv3.multi_turn.tree_of_attacks import TreeOfAttacksStrategy
from pyrit.orchestratorv3.single_turn.prompt_sending import PromptSendingAttackStrategy
from pyrit.prompt_converter.prompt_converter import PromptConverter
from pyrit.prompt_normalizer.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.prompt_target.common.prompt_target import PromptTarget
from pyrit.score.scorer import Scorer


class AttackStrategyType(Enum):
    """Types of attack strategies"""

    # Multi-turn Attacks
    RED_TEAMING = "red_teaming"
    TREE_OF_ATTACKS = "tree_of_attackS"

    # Single-turn Attacks
    PROMPT_SENDING = "prompt_sending"


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
        prompt_converters: Optional[List[PromptConverter]] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        backtracking_strategy: Optional[BacktrackingStrategy[MultiTurnAttackContext]] = None,
        use_score_as_feedback: bool = False,
    ) -> RedTeamingStrategy: ...

    @overload
    @classmethod
    def create(
        cls,
        *,
        # Overload Identifier (Prompt Sending)
        strategy_type: Literal[AttackStrategyType.PROMPT_SENDING],
        # Prompt Sending Specific Parameters
        objective_target: PromptTarget,
        prompt_converters: Optional[List[PromptConverter]] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        scorers: Optional[List[Scorer]] = None,
    ) -> PromptSendingAttackStrategy: ...

    @overload
    @classmethod
    def create(
        cls,
        *,
        # Overload Identifier (Prompt Sending)
        strategy_type: Literal[AttackStrategyType.TREE_OF_ATTACKS],
        # Prompt Sending Specific Parameters
        objective_target: PromptChatTarget,
        adversarial_chat: PromptChatTarget,
        scoring_target: PromptChatTarget,
        adversarial_chat_system_prompt_path: Optional[Path] = None,
        adversarial_chat_seed_prompt: Optional[SeedPrompt] = None,
        prompt_converters: Optional[List[PromptConverter]] = None,
        on_topic_checking_enabled: bool = True,
        desired_response_prefix: Optional[str] = None,
    ) -> TreeOfAttacksStrategy: ...

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
                # TODO: you could add validation here to check if all required arguments are present
                # and valid for the Red Teaming strategy
                return RedTeamingStrategy(**kwargs)
            case AttackStrategyType.PROMPT_SENDING:
                return PromptSendingAttackStrategy(**kwargs)
            case AttackStrategyType.TREE_OF_ATTACKS:
                return TreeOfAttacksStrategy(**kwargs)
            case _:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
