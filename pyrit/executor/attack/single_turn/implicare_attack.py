# Licensed under the MIT license.

import logging
import pathlib
import uuid
from typing import Optional

from pyrit.common.path import DATASETS_PATH
from pyrit.executor.attack.core import AttackConverterConfig, AttackScoringConfig
from pyrit.executor.attack.single_turn.prompt_sending import PromptSendingAttack
from pyrit.executor.attack.single_turn.single_turn_attack_strategy import (
    SingleTurnAttackContext,
)
from pyrit.models import (
    AttackResult,
    PromptRequestResponse,
    SeedPrompt,
    SeedPromptGroup,
)
from pyrit.prompt_normalizer import PromptConverterConfiguration, PromptNormalizer
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)

class ImplicareAttack(PromptSendingAttack):
    """
    Basically, a single system prompt attack
    """

    def __init__(
        self,
        objective_target: PromptChatTarget,
        attack_converter_config: Optional[AttackConverterConfig] = None,
        attack_scoring_config: Optional[AttackScoringConfig] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        max_attempts_on_failure: int = 0,
    ) -> None:
        """
        Args:
            objective_target (PromptChatTarget): The target system to attack.
            attack_converter_config (AttackConverterConfig, Optional): Configuration for the prompt converters.
            attack_scoring_config (AttackScoringConfig, Optional): Configuration for scoring components.
            prompt_normalizer (PromptNormalizer, Optional): Normalizer for handling prompts.
            max_attempts_on_failure (int, Optional): Maximum number of attempts to retry on failure.
        """
        super().__init__(
            objective_target=objective_target,
            attack_converter_config=attack_converter_config,
            attack_scoring_config=attack_scoring_config,
            prompt_normalizer=prompt_normalizer,
            max_attempts_on_failure=max_attempts_on_failure,
        )
        # This system prompt is sent to the target for getting the requested information
        system_prompt_path = pathlib.Path(DATASETS_PATH) / "executors" / "implicare_attack.yaml"
        system_prompt = SeedPrompt.from_yaml_file(system_prompt_path).value

    def _validate_context(self, *, context: SingleTurnAttackContext) -> None:
        """
        Validate the context before executing the attack.

        Args:
            context (SingleTurnAttackContext): The attack context containing parameters and objective.

        Raises:
            ValueError: If the context is invalid.
        """
        if context.prepended_conversation:
            raise ValueError("Implicare does not support prepended conversations.")
        super()._validate_context(context=context)

     async def _perform_async(self, *, context: SingleTurnAttackContext) -> AttackResult:
        """
        Perform the FlipAttack.

        Args:
            context (SingleTurnAttackContext): The attack context containing attack parameters.

        Returns:
            AttackResult: The result of the attack.
        """
        initial_prompt = "TASK is '⟪" + context.objective.replace("'", "") + "⟫'"
        seed_prompt_group = SeedPromptGroup(prompts=[SeedPrompt(value=initial_prompt, data_type="text")])
        context.seed_prompt_group = seed_prompt_group

        return await super()._perform_async(context=context)
