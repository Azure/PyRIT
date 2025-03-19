# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
import logging
import pathlib
from typing import Optional

from pyrit.common.path import DATASETS_PATH
from pyrit.models import PromptRequestPiece, PromptRequestResponse, SeedPromptDataset
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter import LLMGenericTextConverter, PromptConverter
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
                ensure proper rate limit management.
            verbose (bool, Optional): Whether to log debug information. Defaults to False.
        """

        self._adversarial_chat = adversarial_chat

        role_play_definition: SeedPromptDataset = SeedPromptDataset.from_yaml_file(role_play_definition_path)

        self._rephrase_instructions = role_play_definition.prompts[0]
        self._user_start_turn = role_play_definition.prompts[1]
        self._assistant_start_turn = role_play_definition.prompts[2]

        rephrase_turn_converter = LLMGenericTextConverter(
            converter_target=adversarial_chat,
            user_prompt_template_with_objective=self._rephrase_instructions,
        )

        super().__init__(
            objective_target=objective_target,
            prompt_converters=[rephrase_turn_converter] + (prompt_converters or []),
            scorers=scorers,
            batch_size=batch_size,
            verbose=verbose,
        )

        self._set_default_conversation_start()

    def validate_normalizer_requests(self, *, prompt_request_list: list[NormalizerRequest]):
        if not prompt_request_list:
            raise ValueError("No normalizer requests provided")

        for request in prompt_request_list:
            if len(request.seed_prompt_group.prompts) > 1:
                raise ValueError("Multi-part messages not supported")
            if request.seed_prompt_group.prompts[0].data_type != "text":
                raise ValueError("Non text messages not supported")

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
