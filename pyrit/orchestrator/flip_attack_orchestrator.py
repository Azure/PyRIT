# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from colorama import Fore, Style
import logging

from typing import Optional

from pyrit.common.display_response import display_response
from pyrit.memory import MemoryInterface
from pyrit.models import PromptDataType
from pyrit.models import PromptRequestResponse
from pyrit.orchestrator import Orchestrator
from pyrit.orchestrator.prompt_sending_orchestrator import PromptSendingOrchestrator
from pyrit.orchestrator.scoring_orchestrator import ScoringOrchestrator
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_target.prompt_chat_target.prompt_chat_target import PromptChatTarget
from pyrit.score import Scorer


logger = logging.getLogger(__name__)


class FlipAttackOrchestrator(PromptSendingOrchestrator):
    """
    This orchestrator takes a list of prompts and sends them to a target.
    """

    def __init__(
        self,
        prompt_target: PromptTarget,
        scorers: Optional[list[Scorer]] = None,
        memory: MemoryInterface = None,
        memory_labels: Optional[dict[str, str]] = None,
        batch_size: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            prompt_target (PromptTarget): The target for sending prompts.
            scorers (list[Scorer], optional): List of scorers to use for each prompt request response, to be
                scored immediately after receiving response. Default is None.
            memory (MemoryInterface, optional): The memory interface. Defaults to None.
            memory_labels (dict[str, str], optional): A free-form dictionary for tagging prompts with custom labels.
            These labels can be used to track all prompts sent as part of an operation, score prompts based on
            the operation ID (op_id), and tag each prompt with the relevant Responsible AI (RAI) harm category.
            Users can define any key-value pairs according to their needs. Defaults to None.
            batch_size (int, optional): The (max) batch size for sending prompts. Defaults to 10.
                Note: If providing max requests per minute on the prompt_target, this should be set to 1 to
                ensure proper rate limit management.
        """

        # TODO
        super().__init__(
            prompt_converters=prompt_converters, memory=memory, memory_labels=memory_labels, verbose=verbose
        )

        self._prompt_normalizer = PromptNormalizer(memory=self._memory)
        self._scorers = scorers
        # Set the scorer and scorer._prompt_target memory to match the orchestrator's memory.
        if self._scorers:
            for scorer in self._scorers:
                scorer._memory = self._memory
                if hasattr(scorer, "_prompt_target"):
                    scorer._prompt_target._memory = self._memory

        self._prompt_target = prompt_target
        self._prompt_target._memory = self._memory

        self._batch_size = batch_size

        self._system_prompt = "TODO GET FROM TEMPLATRE"

    async def send_prompts_async(
        self,
        *,
        prompt_list: list[str],
        memory_labels: Optional[dict[str, str]] = None,
        metadata: Optional[str] = None,
    ) -> list[PromptRequestResponse]:
        """
        Sends the prompts to the prompt target.

        Args:
            prompt_list (list[str]): The list of prompts to be sent.
            prompt_type (PromptDataType): The type of prompt data. Defaults to "text".
            memory_labels (dict[str, str], optional): A free-form dictionary of additional labels to apply to the
                prompts.
            These labels will be merged with the instance's global memory labels. Defaults to None.
            metadata: Any additional information to be added to the memory entry corresponding to the prompts sent.

        Returns:
            list[PromptRequestResponse]: The responses from sending the prompts.
        """



    async def send_prompt_async(self):
        raise NotImplementedError("This method is not supported for this orchestrator.")

    async def send_normalizer_requests_async(self):
        raise NotImplementedError("This method is not supported for this orchestrator.")


