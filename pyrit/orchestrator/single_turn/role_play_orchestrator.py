# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
import logging
import pathlib
from typing import Optional

from pyrit.prompt_target.batch_helper import batch_task_async


from pyrit.common.path import DATASETS_PATH
from pyrit.models import PromptRequestPiece, PromptRequestResponse, SeedPromptDataset, SeedPrompt, SeedPromptGroup
from pyrit.orchestrator import PromptSendingOrchestrator, OrchestratorResult
from pyrit.prompt_converter import LLMGenericTextConverter, PromptConverter
from pyrit.prompt_normalizer import NormalizerRequest, PromptConverterConfiguration
from pyrit.prompt_normalizer.prompt_converter_configuration import convert_to_configurations
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
    This orchestrator implements a game role play
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
        batch_size: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            objective_target (PromptTarget): The target for sending prompts.
            prompt_converters (list[PromptConverter], Optional): List of prompt converters. These are stacked in
                order.
            scorers (list[Scorer], Optional): List of scorers to use for each prompt request response, to be
                scored immediately after receiving response. Default is None.
            batch_size (int, Optional): The (max) batch size for sending prompts. Defaults to 10.
                Note: If providing max requests per minute on the prompt_target, this should be set to 1 to
                ensure proper rate limit management.
            verbose (bool, Optional): Whether to log debug information. Defaults to False.
        """

        self._adversarial_chat = adversarial_chat

        role_play_definition: SeedPromptDataset = SeedPromptDataset.from_yaml_file(role_play_definition_path)

        self._rephrase_instructions = role_play_definition.prompts[0]
        self._user_start_turn = role_play_definition.prompts[1]
        self._assistant_start_turn = role_play_definition.prompts[2]

        rephrase_turn_converter = convert_to_configurations([
            LLMGenericTextConverter(
                converter_target=adversarial_chat,
                user_prompt_template_with_objective=self._rephrase_instructions,
            )]
        )

        super().__init__(
            objective_target=objective_target,
            request_converter_configurations= rephrase_turn_converter + (request_converter_configurations or []),
            response_converter_configurations=response_converter_configurations,
            objective_scorer=objective_scorer,
            auxiliary_scorers=auxiliary_scorers,
            batch_size=batch_size,
            verbose=verbose,
        )


    async def run_attack_async(
        self,
        *,
        objective: str,
        retries_on_objective_failure: int = 0,
        memory_labels: Optional[dict[str, str]] = None,
    ) -> OrchestratorResult:
        return await super().run_attack_async(
            objective=objective,
            prepended_conversation=self._get_conversation_start(),
            retries_on_objective_failure=retries_on_objective_failure,
            memory_labels=memory_labels,
        )
    
    async def run_attacks_async(
        self,
        *,
        objectives: list[str],
        memory_labels: Optional[dict[str, str]] = None,
    ) -> list[OrchestratorResult]:
        """
        Runs multiple role play attacks in parallel using batch_size.

        Args:
            objectives (list[str]): List of objectives for the attacks.
            memory_labels (dict[str, str], Optional): The memory labels to use for the attacks.
        Returns:
            list[OrchestratorResult]: List of results from each attack.
        """

        batch_items = [
            objectives,
        ]

        batch_item_keys = [
            "objective",
        ]

        results = await batch_task_async(
            prompt_target=self._objective_target,
            batch_size=self._batch_size,
            items_to_batch=batch_items,
            task_func=self.run_attack_async,
            task_arguments=batch_item_keys,
            memory_labels=memory_labels
        )

        return results


    def _get_conversation_start(self):

        return [
            PromptRequestResponse(
                request_pieces=[
                    PromptRequestPiece(
                        role="user",
                        original_value=self._user_start_turn.value,
                    )
                ]
            ),
            PromptRequestResponse(
                request_pieces=[
                    PromptRequestPiece(
                        role="assistant",
                        original_value=self._assistant_start_turn.value,
                    )
                ]
            ),
        ]

