# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib
import uuid
from typing import Optional

from pyrit.attacks.base.attack_config import AttackConverterConfig, AttackScoringConfig
from pyrit.attacks.base.attack_context import SingleTurnAttackContext
from pyrit.attacks.single_turn.prompt_sending import PromptSendingAttack
from pyrit.common.path import DATASETS_PATH
from pyrit.common.utils import combine_dict
from pyrit.models import (
    AttackResult,
    PromptRequestResponse,
    SeedPrompt,
    SeedPromptGroup,
)
from pyrit.prompt_converter import FlipConverter
from pyrit.prompt_normalizer import PromptConverterConfiguration, PromptNormalizer
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class FlipAttack(PromptSendingAttack):
    """
    This attack implements the FlipAttack method found here:
    https://arxiv.org/html/2410.02832v1.

    Essentially, it adds a system prompt to the beginning of the conversation to flip each word in the prompt.
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

        flip_converter = PromptConverterConfiguration.from_converters(converters=[FlipConverter()])
        self._request_converters = flip_converter + self._request_converters

        # This system prompt is sent to the target to flip the words in the prompt.
        system_prompt_path = pathlib.Path(DATASETS_PATH) / "orchestrators" / "flip_attack.yaml"
        system_prompt = SeedPrompt.from_yaml_file(system_prompt_path).value

        self._system_prompt = PromptRequestResponse.from_system_prompt(system_prompt=system_prompt)

    def _validate_context(self, *, context: SingleTurnAttackContext) -> None:
        """
        Validate the context before executing the attack.

        Args:
            context (SingleTurnAttackContext): The attack context containing parameters and objective.

        Raises:
            ValueError: If the context is invalid.
        """
        if context.prepended_conversation:
            raise ValueError("FlipAttack does not support prepended conversations as it uses its own system prompt.")
        super()._validate_context(context=context)

    async def _setup_async(self, *, context: SingleTurnAttackContext) -> None:
        """
        Set up the FlipAttack by preparing conversation context.

        Args:
            context (SingleTurnAttackContext): The attack context containing attack parameters.
        """
        # Ensure the context has a conversation ID
        context.conversation_id = str(uuid.uuid4())
        context.prepended_conversation = [self._system_prompt]

        # Combine memory labels from context and attack strategy
        context.memory_labels = combine_dict(self._memory_labels, context.memory_labels)

        # System prompt should not be converted, and the new implementation correctly
        # skips converters for system messages, so we can pass empty converter lists
        await self._conversation_manager.update_conversation_state_async(
            conversation_id=context.conversation_id, prepended_conversation=context.prepended_conversation
        )

    async def _perform_attack_async(self, *, context: SingleTurnAttackContext) -> AttackResult:
        """
        Perform the FlipAttack.

        Args:
            context (SingleTurnAttackContext): The attack context containing attack parameters.

        Returns:
            OrchestratorResult: The result of the orchestrated attack.
        """
        initial_prompt = "TASK is '⟪" + context.objective.replace("'", "") + "⟫'"
        seed_prompt_group = SeedPromptGroup(prompts=[SeedPrompt(value=initial_prompt, data_type="text")])
        context.seed_prompt_group = seed_prompt_group

        return await super()._perform_attack_async(context=context)
