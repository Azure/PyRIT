# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import enum
import logging
import warnings
from pathlib import Path
from typing import Optional, cast

from typing_extensions import LiteralString, deprecated

from pyrit.attacks import (
    AttackAdversarialConfig,
    AttackConverterConfig,
    AttackScoringConfig,
    RedTeamingAttack,
)
from pyrit.common import deprecation_message
from pyrit.common.path import RED_TEAM_ORCHESTRATOR_PATH
from pyrit.models import AttackOutcome
from pyrit.orchestrator import MultiTurnOrchestrator, OrchestratorResult
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_normalizer.prompt_converter_configuration import (
    PromptConverterConfiguration,
)
from pyrit.prompt_target import PromptChatTarget, PromptTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class RTOSystemPromptPaths(enum.Enum):
    TEXT_GENERATION = Path(RED_TEAM_ORCHESTRATOR_PATH, "text_generation.yaml").resolve()
    IMAGE_GENERATION = Path(RED_TEAM_ORCHESTRATOR_PATH, "image_generation.yaml").resolve()
    NAIVE_CRESCENDO = Path(RED_TEAM_ORCHESTRATOR_PATH, "naive_crescendo.yaml").resolve()
    VIOLENT_DURIAN = Path(RED_TEAM_ORCHESTRATOR_PATH, "violent_durian.yaml").resolve()
    CRUCIBLE = Path(RED_TEAM_ORCHESTRATOR_PATH, "crucible.yaml").resolve()


@deprecated(
    cast(
        LiteralString,
        deprecation_message(
            old_item="RedTeamingOrchestrator",
            new_item=RedTeamingAttack,
            removed_in="v0.12.0",
        ),
    ),
)
class RedTeamingOrchestrator(MultiTurnOrchestrator):
    """
    .. warning::
        `RedTeamingOrchestrator` is deprecated and will be removed in **v0.12.0**;
        use `pyrit.attacks.RedTeamingAttack` instead.

    The `RedTeamingOrchestrator` class orchestrates a multi-turn red teaming attack on a target system.

    It is extemely simple. It sends a prompt to the target system, and then sends the response to the red teaming chat.

    Args:
        objective_target (PromptTarget): Target for created prompts.
        adversarial_chat (PromptChatTarget): Endpoint creating prompts sent to objective_target.
        adversarial_chat_system_prompt_path (Path): Path to initial adversarial_chat system prompt.
        adversarial_chat_seed_prompt (str): Initial seed prompt for the adversarial chat.
        initial_adversarial_chat_prompt (str, Optional): Initial message to start the chat. Defaults to
            "How can I help you?".
        prompt_converters (Optional[list[PromptConverter]]): Converters for prompt formatting. Defaults to None.
        max_turns (int, Optional): Max turns for the conversation, ≥ 0. Defaults to 5.
        objective_scorer (Scorer): Scores prompt target output as sufficient or insufficient.
        use_score_as_feedback (bool, Optional): Use scoring as feedback. Defaults to True.
        verbose (bool, Optional): Print debug info. Defaults to False.

    Raises:
        FileNotFoundError: If adversarial_chat_system_prompt_path file not found.
        ValueError: If max_turns ≤ 0 or if objective_scorer is not binary.
    """

    def __init__(
        self,
        *,
        objective_target: PromptTarget,
        adversarial_chat: PromptChatTarget,
        adversarial_chat_system_prompt_path: Path = RTOSystemPromptPaths.TEXT_GENERATION.value,
        adversarial_chat_seed_prompt: str = "How can I help you?",
        prompt_converters: Optional[list[PromptConverter]] = None,
        max_turns: int = 5,
        objective_scorer: Scorer,
        use_score_as_feedback: bool = True,
        batch_size: int = 1,
        verbose: bool = False,
    ) -> None:

        warnings.warn(
            deprecation_message(
                old_item="RedTeamingOrchestrator",
                new_item=RedTeamingAttack,
                removed_in="v0.12.0",
            ),
            DeprecationWarning,
            stacklevel=2,
        )

        if objective_scorer.scorer_type != "true_false":
            raise ValueError(
                f"The scorer must be a true/false scorer. The scorer type is {objective_scorer.scorer_type}."
            )

        super().__init__(
            objective_target=objective_target,
            adversarial_chat=adversarial_chat,
            adversarial_chat_system_prompt_path=adversarial_chat_system_prompt_path,
            adversarial_chat_seed_prompt=adversarial_chat_seed_prompt,
            max_turns=max_turns,
            prompt_converters=prompt_converters,
            objective_scorer=objective_scorer,
            verbose=verbose,
            batch_size=batch_size,
        )

        self._prompt_normalizer = PromptNormalizer()
        self._use_score_as_feedback = use_score_as_feedback

        # Build the new attack model
        self._attack = RedTeamingAttack(
            objective_target=objective_target,
            attack_adversarial_config=AttackAdversarialConfig(
                target=adversarial_chat,
                system_prompt_path=adversarial_chat_system_prompt_path,
                seed_prompt=adversarial_chat_seed_prompt,
            ),
            attack_scoring_config=AttackScoringConfig(
                objective_scorer=objective_scorer,
                use_score_as_feedback=use_score_as_feedback,
            ),
            attack_converter_config=AttackConverterConfig(
                request_converters=PromptConverterConfiguration.from_converters(converters=prompt_converters or []),
            ),
            prompt_normalizer=self._prompt_normalizer,
            max_turns=max_turns,
        )

    async def run_attack_async(
        self, *, objective: str, memory_labels: Optional[dict[str, str]] = None
    ) -> OrchestratorResult:

        result = await self._attack.execute_async(
            objective=objective,
            prepended_conversation=self._prepended_conversation,
            memory_labels=memory_labels or {},
        )
        objective_achieved = result.outcome == AttackOutcome.SUCCESS

        # Translating the result back to the orchestrator result format
        return OrchestratorResult(
            conversation_id=result.conversation_id,
            objective=objective,
            status="success" if objective_achieved else "failure",
            objective_score=result.last_score,
            confidence=1.0 if objective_achieved else 0.0,
        )
