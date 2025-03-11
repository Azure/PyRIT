# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib
from typing import Optional, Union

from pyrit.common.path import DATASETS_PATH
from pyrit.models import PromptRequestResponse, SeedPrompt
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter import FlipConverter, PromptConverter
from pyrit.prompt_normalizer import NormalizerRequest
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class FlipAttackOrchestrator(PromptSendingOrchestrator):
    """
    This orchestrator implements the Flip Attack method found here:
    https://arxiv.org/html/2410.02832v1.

    Essentially, adds a system prompt to the beginning of the conversation to flip each word in the prompt.
    """

    def __init__(
        self,
        objective_target: PromptChatTarget,
        prompt_converters: Optional[list[PromptConverter]] = None,
        scorers: Optional[list[Scorer]] = None,
        batch_size: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            objective_target (PromptChatTarget): The target for sending prompts.
            prompt_converters (list[PromptConverter], Optional): List of prompt converters. These are stacked in
                order on top of the flip converter.
            scorers (list[Scorer], Optional): List of scorers to use for each prompt request response, to be
                scored immediately after receiving response. Default is None.
            batch_size (int, Optional): The (max) batch size for sending prompts. Defaults to 10.
                Note: If providing max requests per minute on the objective_target, this should be set to 1 to
                ensure proper rate limit management.\
            verbose (bool, Optional): Whether to log debug information. Defaults to False.
        """

        super().__init__(
            objective_target=objective_target,
            prompt_converters=[FlipConverter()] + (prompt_converters or []),
            scorers=scorers,
            batch_size=batch_size,
            verbose=verbose,
        )

        # This is sent to the target
        system_prompt_path = pathlib.Path(DATASETS_PATH) / "orchestrators" / "flip_attack.yaml"
        self.system_prompt = SeedPrompt.from_yaml_file(system_prompt_path).value

        system_prompt = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="system",
                    original_value=self.system_prompt,
                )
            ]
        )

        super().set_prepended_conversation(prepended_conversation=[system_prompt])

    async def send_prompts_async(  # type: ignore[override]
        self,
        *,
        prompt_list: list[str],
        memory_labels: Optional[dict[str, str]] = None,
        metadata: Optional[dict[str, Union[str, int]]] = None,
    ) -> list[PromptRequestResponse]:
        """
        Sends the prompts to the prompt target using flip attack.

        Args:
            prompt_list (list[str]): The list of prompts to be sent.
            memory_labels (dict[str, str], Optional): A free-form dictionary of additional labels to apply to the
                prompts. Any labels passed in will be combined with self._global_memory_labels with the passed
                in labels taking precedence in the case of collisions. Defaults to None.
            metadata (Optional(dict[str, str | int]): Any additional information to be added to the memory entry
                corresponding to the prompts sent.

        Returns:
            list[PromptRequestResponse]: The responses from sending the prompts.
        """

        # Format follows the paper https://github.com/yueliu1999/FlipAttack/blob/main/result/FlipAttack-gpt-4.json

        for i in range(0, len(prompt_list)):
            prompt_list[i] = "TASK is '⟪" + prompt_list[i].replace("'", "") + "⟫'"

        return await super().send_prompts_async(
            prompt_list=prompt_list, prompt_type="text", memory_labels=memory_labels, metadata=metadata
        )

    def validate_normalizer_requests(self, *, prompt_request_list: list[NormalizerRequest]):
        if not prompt_request_list:
            raise ValueError("No normalizer requests provided")

        for request in prompt_request_list:
            if len(request.seed_prompt_group.prompts) > 1:
                raise ValueError("Multi-part messages not supported")
            if request.seed_prompt_group.prompts[0].data_type != "text":
                raise ValueError("Non text messages not supported")
