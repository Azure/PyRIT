# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
import logging
import pathlib
from typing import Optional

from pyrit.common.path import DATASETS_PATH
from pyrit.executor.attack.core import AttackConverterConfig, AttackScoringConfig
from pyrit.executor.attack.single_turn.prompt_sending import PromptSendingAttack
from pyrit.executor.attack.single_turn.single_turn_attack_strategy import (
    SingleTurnAttackContext,
)
from pyrit.models import (
    Message,
    SeedDataset,
)
from pyrit.prompt_converter import LLMGenericTextConverter
from pyrit.prompt_normalizer import PromptConverterConfiguration, PromptNormalizer
from pyrit.prompt_target import PromptChatTarget, PromptTarget

logger = logging.getLogger(__name__)


class RolePlayPaths(enum.Enum):
    VIDEO_GAME = pathlib.Path(DATASETS_PATH) / "executors" / "role_play" / "video_game.yaml"
    MOVIE_SCRIPT = pathlib.Path(DATASETS_PATH) / "executors" / "role_play" / "movie_script.yaml"
    TRIVIA_GAME = pathlib.Path(DATASETS_PATH) / "executors" / "role_play" / "trivia_game.yaml"
    PERSUASION_SCRIPT = pathlib.Path(DATASETS_PATH) / "executors" / "role_play" / "persuasion_script.yaml"


class RolePlayAttack(PromptSendingAttack):
    """
    Implementation of single-turn role-play attack strategy.

    This class orchestrates a role-play attack where malicious objectives are rephrased
    into role-playing contexts to make them appear more benign and bypass content filters.
    The strategy uses an adversarial chat target to transform the objective into a role-play
    scenario before sending it to the target system.

    The attack flow consists of:
    1. Loading role-play scenarios from a YAML file.
    2. Using an adversarial chat target to rephrase the objective into the role-play context.
    3. Sending the rephrased objective to the target system.
    4. Evaluating the response with scorers if configured.
    5. Retrying on failure up to the configured number of retries.
    6. Returning the attack result

    The strategy supports customization through prepended conversations, converters,
    and multiple scorer types.
    """

    def __init__(
        self,
        *,
        objective_target: PromptTarget,
        adversarial_chat: PromptChatTarget,
        role_play_definition_path: pathlib.Path,
        attack_converter_config: Optional[AttackConverterConfig] = None,
        attack_scoring_config: Optional[AttackScoringConfig] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        max_attempts_on_failure: int = 0,
    ) -> None:
        """
        Initializes the role-play attack strategy.

        Args:
            objective_target (PromptTarget): The target system to attack.
            adversarial_chat (PromptChatTarget): The adversarial chat target used to rephrase
                objectives into role-play scenarios.
            role_play_definition_path (pathlib.Path): Path to the YAML file containing role-play
                definitions (rephrase instructions, user start turn, assistant start turn).
            attack_converter_config (Optional[AttackConverterConfig]): Configuration for prompt converters.
            attack_scoring_config (Optional[AttackScoringConfig]): Configuration for scoring components.
            prompt_normalizer (Optional[PromptNormalizer]): Normalizer for handling prompts.
            max_attempts_on_failure (int): Maximum number of attempts to retry the attack

        Raises:
            ValueError: If the objective scorer is not a true/false scorer.
            FileNotFoundError: If the role_play_definition_path does not exist.
        """
        # Initialize the parent class first
        super().__init__(
            objective_target=objective_target,
            attack_converter_config=attack_converter_config,
            attack_scoring_config=attack_scoring_config,
            prompt_normalizer=prompt_normalizer,
            max_attempts_on_failure=max_attempts_on_failure,
        )

        # Store the adversarial chat for role-play rephrasing
        self._adversarial_chat = adversarial_chat

        # Load role-play definitions
        role_play_definition = SeedDataset.from_yaml_file(role_play_definition_path)

        # Validate role-play definition structure
        self._parse_role_play_definition(role_play_definition)

        # Create the rephrase converter configuration
        rephrase_converter = PromptConverterConfiguration.from_converters(
            converters=[
                LLMGenericTextConverter(
                    converter_target=self._adversarial_chat,
                    user_prompt_template_with_objective=self._rephrase_instructions,
                )
            ]
        )

        # Prepend the rephrase converter to existing request converters
        self._request_converters = rephrase_converter + self._request_converters

    async def _setup_async(self, *, context: SingleTurnAttackContext) -> None:
        """
        Sets up the attack by preparing conversation context with role-play start.

        Args:
            context (SingleTurnAttackContext): The attack context containing attack parameters.
        """
        # Get role-play conversation start
        context.prepended_conversation = await self._get_conversation_start() or []

        # Call parent setup which handles conversation ID generation, memory labels, etc.
        await super()._setup_async(context=context)

    def _validate_context(self, *, context: SingleTurnAttackContext) -> None:
        """
        Validate the context before executing the attack.
        Args:
            context (SingleTurnAttackContext): The attack context containing parameters and objective.
        Raises:
            ValueError: If the context is invalid.
        """
        if context.prepended_conversation:
            raise ValueError("RolePlayAttack does not support prepended conversations.")
        super()._validate_context(context=context)

    async def _get_conversation_start(self) -> Optional[list[Message]]:
        """
        Get the role-play conversation start messages.

        Returns:
            Optional[list[Message]]: List containing user and assistant start turns
                for the role-play scenario.
        """
        return [
            Message.from_prompt(
                prompt=self._user_start_turn.value,
                role="user",
            ),
            Message.from_prompt(
                prompt=self._assistant_start_turn.value,
                role="assistant",
            ),
        ]

    def _parse_role_play_definition(self, role_play_definition: SeedDataset):
        """
        Parses and validates the role-play definition structure.

        Args:
            role_play_definition (SeedDataset): The role-play definition dataset to validate.

        Raises:
            ValueError: If the definition does not contain exactly 3 prompts or if any prompt is empty.
        """
        if len(role_play_definition.prompts) != 3:
            raise ValueError(
                f"Role-play definition must contain 3 prompts, but found {len(role_play_definition.prompts)}. "
                "Expected: [rephrase_instructions, user_start_turn, assistant_start_turn]"
            )
        for i, prompt in enumerate(role_play_definition.prompts):
            if not prompt.value or not prompt.value.strip():
                prompt_names = ["rephrase_instructions", "user_start_turn", "assistant_start_turn"]
                raise ValueError(f"Role-play definition prompt '{prompt_names[i]}' cannot be empty")

        self._rephrase_instructions = role_play_definition.prompts[0]
        self._user_start_turn = role_play_definition.prompts[1]
        self._assistant_start_turn = role_play_definition.prompts[2]
