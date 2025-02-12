# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
import logging
import pathlib
from typing import Optional

from pyrit.common.path import DATASETS_PATH
from pyrit.models import (
    PromptRequestPiece,
    PromptRequestResponse,
    SeedPrompt,
    SeedPromptDataset,
    SeedPromptGroup,
)
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_normalizer import NormalizerRequest
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class RolePlayPaths(enum.Enum):
    VIDEO_GAME = pathlib.Path(DATASETS_PATH) / "orchestrators" / "role_play" / "video_game.yaml"
    MOVIE_SCRIPT = pathlib.Path(DATASETS_PATH) / "orchestrators" / "role_play" / "movie_script.yaml"
    TRIVIA_GAME = pathlib.Path(DATASETS_PATH) / "orchestrators" / "role_play" / "trivia_game.yaml"


class RolePlayOrchestrator(PromptSendingOrchestrator):
    """
    This orchestrator implements a game role play
    """

    def __init__(
        self,
        objective_target: PromptChatTarget,
        adversarial_chat: PromptChatTarget,
        role_play_definition_path: pathlib.Path,
        prompt_converters: Optional[list[PromptConverter]] = None,
        scorers: Optional[list[Scorer]] = None,
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
                ensure proper rate limit management.\
            verbose (bool, Optional): Whether to log debug information. Defaults to False.
        """

        super().__init__(
            objective_target=objective_target,
            prompt_converters=prompt_converters,
            scorers=scorers,
            batch_size=batch_size,
            verbose=verbose,
        )

        self._adversarial_chat = adversarial_chat

        role_play_definition: SeedPromptDataset = SeedPromptDataset.from_yaml_file(role_play_definition_path)

        self._rephrase_instructions = role_play_definition.prompts[0]
        self._user_start_turn = role_play_definition.prompts[1]
        self._assistant_start_turn = role_play_definition.prompts[2]

        self._set_default_conversation_start()

    async def send_prompts_async(  # type: ignore[override]
        self,
        *,
        prompt_list: list[str],
        memory_labels: Optional[dict[str, str]] = None,
        metadata: Optional[dict[str, str] | None] = None,
    ) -> list[PromptRequestResponse]:
        """
        Sends the prompts to the prompt target using a defined role playing scenario.

        Args:
            prompt_list (list[str]): The list of prompts (objectives) to be sent.
            memory_labels (dict[str, str], Optional): A free-form dictionary of additional labels to apply to the
                prompts. Any labels passed in will be combined with self._global_memory_labels with the passed
                in labels taking precedence in the case of collisions. Defaults to None.
            metadata: Any additional information to be added to the memory entry corresponding to the prompts sent.

        Returns:
            list[PromptRequestResponse]: The responses from sending the prompts.
        """

        role_playing_prompts = await self._get_role_playing_prompts_async(prompt_list)

        return await super().send_prompts_async(
            prompt_list=role_playing_prompts, prompt_type="text", memory_labels=memory_labels, metadata=metadata
        )

    async def _get_role_playing_prompts_async(self, objective_list: list[str]) -> list[str]:
        """
        Returns the role playing prompts for the given list of prompts.

        Args:
            prompt_list (list[str]): The list of prompts to be role played.

        Returns:
            list[str]: The role playing prompts.
        """
        requests = []

        for objective in objective_list:
            normalizer_request = NormalizerRequest(
                seed_prompt_group=SeedPromptGroup(
                    prompts=[
                        SeedPrompt(
                            value=self._rephrase_instructions.render_template_value(objective=objective),
                            data_type="text",
                        )
                    ]
                )
            )

            requests.append(normalizer_request)

        role_playing_prompts: list[PromptRequestResponse] = (
            await self._prompt_normalizer.send_prompt_batch_to_target_async(
                requests=requests,
                target=self._adversarial_chat,
                batch_size=self._batch_size,
            )
        )

        return [role_playing_prompt.request_pieces[0].original_value for role_playing_prompt in role_playing_prompts]

    def _set_default_conversation_start(self):

        prepended_conversation = [
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

        self.set_prepended_conversation(prepended_conversation=prepended_conversation)
