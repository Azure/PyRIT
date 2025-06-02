# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
import logging
import pathlib
from typing import Optional

from pyrit.common.path import DATASETS_PATH
from pyrit.models import PromptRequestResponse, SeedPromptDataset
from pyrit.orchestrator import OrchestratorResult, PromptSendingOrchestrator
from pyrit.prompt_converter import LLMGenericTextConverter
from pyrit.prompt_normalizer import PromptConverterConfiguration
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class RolePlayPaths(enum.Enum):
    VIDEO_GAME = pathlib.Path(DATASETS_PATH) / "orchestrators" / "role_play" / "video_game.yaml"
    MOVIE_SCRIPT = pathlib.Path(DATASETS_PATH) / "orchestrators" / "role_play" / "movie_script.yaml"
    TRIVIA_GAME = pathlib.Path(DATASETS_PATH) / "orchestrators" / "role_play" / "trivia_game.yaml"
    PERSUASION_SCRIPT = pathlib.Path(DATASETS_PATH) / "orchestrators" / "role_play" / "persuasion_script.yaml"


class RolePlayOrchestrator(PromptSendingOrchestrator):
    """
    This orchestrator implements a role-playing attack where the objective is rephrased into a game or script context.
    It uses an adversarial chat target to rephrase the objective into a more benign form that fits within the role-play
    scenario, making it harder for the target to detect the true intent.
    """

    def __init__(
        self,
        objective_target: PromptChatTarget,
        adversarial_chat: PromptChatTarget,
        role_play_definition_path: pathlib.Path,
        request_converter_configurations: Optional[list[PromptConverterConfiguration]] = None,
        response_converter_configurations: Optional[list[PromptConverterConfiguration]] = None,
        objective_scorer: Optional[Scorer] = None,
        auxiliary_scorers: Optional[list[Scorer]] = None,
        should_convert_prepended_conversation: bool = True,
        batch_size: int = 10,
        retries_on_objective_failure: int = 0,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            objective_target (PromptChatTarget): The target for sending prompts.
            adversarial_chat (PromptChatTarget): The target used to rephrase objectives into role-play scenarios.
            role_play_definition_path (pathlib.Path): Path to the YAML file containing role-play definitions.
            request_converter_configurations (list[PromptConverterConfiguration], Optional): List of prompt
                converters.
            response_converter_configurations (list[PromptConverterConfiguration], Optional): List of response
                converters.
            objective_scorer (Scorer, Optional): Scorer to use for evaluating if the objective was achieved.
            auxiliary_scorers (list[Scorer], Optional): List of additional scorers to use for each prompt request
                response.
            should_convert_prepended_conversation (bool, Optional): Whether to convert the prepended conversation.
            batch_size (int, Optional): The (max) batch size for sending prompts. Defaults to 10.
                Note: If providing max requests per minute on the prompt_target, this should be set to 1 to
                ensure proper rate limit management.
            retries_on_objective_failure (int, Optional): Number of retries to attempt if objective fails. Defaults to
                0.
            verbose (bool, Optional): Whether to log debug information. Defaults to False.
        """

        self._adversarial_chat = adversarial_chat

        role_play_definition: SeedPromptDataset = SeedPromptDataset.from_yaml_file(role_play_definition_path)

        self._rephrase_instructions = role_play_definition.prompts[0]
        self._user_start_turn = role_play_definition.prompts[1]
        self._assistant_start_turn = role_play_definition.prompts[2]

        rephrase_turn_converter = PromptConverterConfiguration.from_converters(
            converters=[
                LLMGenericTextConverter(
                    converter_target=adversarial_chat,
                    user_prompt_template_with_objective=self._rephrase_instructions,
                )
            ]
        )

        super().__init__(
            objective_target=objective_target,
            request_converter_configurations=rephrase_turn_converter + (request_converter_configurations or []),
            response_converter_configurations=response_converter_configurations,
            objective_scorer=objective_scorer,
            auxiliary_scorers=auxiliary_scorers,
            should_convert_prepended_conversation=should_convert_prepended_conversation,
            batch_size=batch_size,
            retries_on_objective_failure=retries_on_objective_failure,
            verbose=verbose,
        )

    async def run_attack_async(  # type: ignore[override]
        self,
        *,
        objective: str,
        memory_labels: Optional[dict[str, str]] = None,
    ) -> OrchestratorResult:

        prepended_conversation = await self._get_conversation_start(objective=objective)
        return await super().run_attack_async(
            objective=objective,
            prepended_conversation=prepended_conversation,
            memory_labels=memory_labels,
        )

    async def run_attacks_async(  # type: ignore[override]
        self,
        *,
        objectives: list[str],
        memory_labels: Optional[dict[str, str]] = None,
    ) -> list[OrchestratorResult]:
        return await super()._run_attacks_with_only_objectives_async(
            objectives=objectives,
            memory_labels=memory_labels,
        )

    async def _get_conversation_start(self, objective: str = None) -> Optional[list[PromptRequestResponse]]:

        return [
            PromptRequestResponse.from_prompt(
                prompt=self._user_start_turn.value,
                role="user",
            ),
            PromptRequestResponse.from_prompt(
                prompt=self._assistant_start_turn.value,
                role="assistant",
            ),
        ]
