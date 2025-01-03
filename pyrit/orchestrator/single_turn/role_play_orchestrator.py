# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib
from typing import Optional

from pyrit.common.path import DATASETS_PATH
from pyrit.models import PromptRequestResponse, SeedPrompt
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter.flip_converter import FlipConverter
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class RolePlayOrchestrator(PromptSendingOrchestrator):
    """
    This orchestrator implements a game role play:
    https://arxiv.org/html/2410.02832v1.

    Essentially, adds a system prompt to the beginning of the conversation to flip each word in the prompt.
    """

    def __init__(
        self,
        objective_target: PromptTarget,
        adversarial_chat: PromptChatTarget,
        scorers: Optional[list[Scorer]] = None,
        batch_size: int = 10,
        rephrase_as_role_play_prompt: SeedPrompt = None,
        prepended_conversation: list[PromptRequestPiece] = None,
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

    

        if not rephrase_as_role_play_prompt:
            self.rephrase_as_roleplay_template = SeedPrompt.from_yaml_file(
                pathlib.Path(DATASETS_PATH) / "orchestrator" / "role_play" / "rephrase_video_game.yaml"
            )

        super().set_prepended_conversation(prepended_conversation=[system_prompt])

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
            prompt_list (list[str]): The list of prompts to be sent.
            memory_labels (dict[str, str], Optional): A free-form dictionary of additional labels to apply to the
                prompts. Any labels passed in will be combined with self._global_memory_labels with the passed
                in labels taking precedence in the case of collisions. Defaults to None.
            metadata: Any additional information to be added to the memory entry corresponding to the prompts sent.

        Returns:
            list[PromptRequestResponse]: The responses from sending the prompts.
        """


        for i in range(0, len(prompt_list)):
            prompt_list[i] = self.rephrase_as_roleplay_template.render_template_value(objective=prompt_list[i])

        return await super().send_prompts_async(
            prompt_list=prompt_list, prompt_type="text", memory_labels=memory_labels, metadata=metadata
        )
    
    def _set_default_conversation_start(self):
        default_system_prompt = SeedPrompt.from_yaml_file(
                pathlib.Path(DATASETS_PATH) / "prompt_templates" / "role_play.yaml"
            ).render_template_value()

        default_user_prompt = SeedPrompt.from_yaml_file(
            pathlib.Path(DATASETS_PATH) / "prompt_templates" / "role_play_user_prompt.yaml"
        ).render_template_value()

        default_assistant_prompt = SeedPrompt.from_yaml_file(
            pathlib.Path(DATASETS_PATH) / "prompt_templates" / "role_play_assistant_prompt.yaml"
        ).render_template_value()

        prepended_conversation = []


        prepended_conversation = [
            PromptRequestResponse(
                request_pieces=[
                    PromptRequestPiece(
                        role="system",
                        original_value=default_system_prompt,
                        )
                ]
            ),
            PromptRequestResponse(
                request_pieces=[
                    PromptRequestPiece(
                        role="user",
                        original_value=default_system_prompt,
                    )
                ]
            )
        ]

