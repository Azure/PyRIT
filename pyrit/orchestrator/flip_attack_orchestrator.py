# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
import logging

from typing import Optional

from pyrit.common.path import DATASETS_PATH
from pyrit.models import PromptRequestResponse
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models import SeedPrompt
from pyrit.orchestrator.prompt_sending_orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter.flip_converter import FlipConverter
from pyrit.prompt_target import PromptTarget
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
        prompt_target: PromptTarget,
        scorers: Optional[list[Scorer]] = None,
        memory_labels: Optional[dict[str, str]] = None,
        batch_size: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            prompt_target (PromptTarget): The target for sending prompts.
            scorers (list[Scorer], Optional): List of scorers to use for each prompt request response, to be
                scored immediately after receiving response. Default is None.
            memory_labels (dict[str, str], Optional): A free-form dictionary for tagging prompts with custom labels.
            These labels can be used to track all prompts sent as part of an operation, score prompts based on
            the operation ID (op_id), and tag each prompt with the relevant Responsible AI (RAI) harm category.
            Users can define any key-value pairs according to their needs. Defaults to None.
            batch_size (int, Optional): The (max) batch size for sending prompts. Defaults to 10.
                Note: If providing max requests per minute on the prompt_target, this should be set to 1 to
                ensure proper rate limit management.\
            verbose (bool, Optional): Whether to log debug information. Defaults to False.
        """

        super().__init__(
            prompt_target=prompt_target,
            prompt_converters=[FlipConverter()],
            scorers=scorers,
            memory_labels=memory_labels,
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
        metadata: Optional[str] = None,
    ) -> list[PromptRequestResponse]:
        """
        Sends the prompts to the prompt target using flip attack.

        Args:
            prompt_list (list[str]): The list of prompts to be sent.
            memory_labels (dict[str, str], Optional): A free-form dictionary of additional labels to apply to the
                prompts.
            These labels will be merged with the instance's global memory labels. Defaults to None.
            metadata: Any additional information to be added to the memory entry corresponding to the prompts sent.

        Returns:
            list[PromptRequestResponse]: The responses from sending the prompts.
        """

        # Format follows the paper https://github.com/yueliu1999/FlipAttack/blob/main/result/FlipAttack-gpt-4.json

        for i in range(0, len(prompt_list)):
            prompt_list[i] = "TASK is '⟪" + prompt_list[i].replace("'", "") + "⟫'"

        return await super().send_prompts_async(
            prompt_list=prompt_list, prompt_type="text", memory_labels=memory_labels, metadata=metadata
        )
