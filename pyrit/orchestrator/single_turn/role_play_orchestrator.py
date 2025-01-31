# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
import logging
import pathlib
from typing import Optional

from pyrit.common.path import DATASETS_PATH
from pyrit.models import PromptRequestResponse, SeedPrompt
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.seed_prompt import SeedPromptDataset
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter.flip_converter import FlipConverter
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.score import Scorer
from pyrit.common.utils import combine_dict


logger = logging.getLogger(__name__)


class RolePlayPaths(enum.Enum):
    VIDEO_GAME = pathlib.Path(DATASETS_PATH) / "orchestrators" / "role_play" / "video_game.yaml"



class RolePlayOrchestrator(PromptSendingOrchestrator):
    """
    This orchestrator implements a game role play
    """

    def __init__(
        self,
        objective_target: PromptTarget,
        adversarial_chat: PromptChatTarget,
        role_play_definition: pathlib.Path,
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

        # TODO converters
        super().__init__(
            objective_target=objective_target,
            prompt_converters=[],
            scorers=scorers,
            batch_size=batch_size,
            verbose=verbose,
        )

        self._adversarial_chat = adversarial_chat

        role_play_definition: SeedPromptDataset = SeedPromptDataset.from_yaml_file(
            pathlib.Path(DATASETS_PATH) / "orchestrators" / "role_play" / "video_game.yaml"
        )

        self._rephrase_instructions = role_play_definition.prompts[0]
        self._user_start_turn = role_play_definition.prompts[1]
        self._assistant_start_turn = role_play_definition.prompts[2]

    async def send_prompts_async(  # type: ignore[override]
        self,
        *,
        prompt_list: list[str],
        memory_labels: Optional[dict[str, str]] = None,
        metadata: Optional[str] = None,
    ) -> list[PromptRequestResponse]:
        """
        Sends the prompts to the prompt target using flip attack.

        Args:
            prompt_list (list[str]): The list of prompts (objectives) to be sent.
            memory_labels (dict[str, str], Optional): A free-form dictionary of additional labels to apply to the
                prompts. Any labels passed in will be combined with self._global_memory_labels with the passed
                in labels taking precedence in the case of collisions. Defaults to None.
            metadata: Any additional information to be added to the memory entry corresponding to the prompts sent.

        Returns:
            list[PromptRequestResponse]: The responses from sending the prompts.
        """

        self._set_default_conversation_start()

        for i in range(0, len(prompt_list)):
            prompt_list[i] = self.rephrase_as_roleplay_template.render_template_value(objective=prompt_list[i])


        # TODO do we have labels on adversarial chats?
        specific_objective_roleplay_requests = await self._prompt_normalizer.send_prompt_async(
            requests=prompt_list,
            target=self._adversarial_chat,
            labels=combine_dict(existing_dict=self._global_memory_labels, new_dict=memory_labels),
            orchestrator_identifier=self.get_identifier(),
            batch_size=self._batch_size,
        )

        # setting the first two turns manually
        # TODO allow for specific system prompt?

        return await super().send_prompts_async(
            prompt_list=prompt_list, prompt_type="text", memory_labels=memory_labels, metadata=metadata
        )

    def _set_default_conversation_start(self):

        default_user_prompt = SeedPrompt.from_yaml_file(
            pathlib.Path(DATASETS_PATH) / "orchestrators" / "role_play" / "user_turn_video_game.yaml"
        ).render_template_value()

        default_assistant_prompt = SeedPrompt.from_yaml_file(
            pathlib.Path(DATASETS_PATH) / "orchestrators" / "role_play" /  "assistant_turn_video_game.yaml"
        ).render_template_value()


        prepended_conversation = [
            PromptRequestResponse(
                request_pieces=[
                    PromptRequestPiece(
                        role="user",
                        original_value=default_user_prompt,
                        )
                ]
            ),
            PromptRequestResponse(
                request_pieces=[
                    PromptRequestPiece(
                        role="assistant",
                        original_value=default_assistant_prompt,
                    )
                ]
            )
        ]

        self.set_prepended_conversation(prepended_conversation)

