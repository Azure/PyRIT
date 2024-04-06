# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from typing import Optional
from uuid import uuid4

from pyrit.memory import MemoryInterface
from pyrit.memory.memory_models import PromptDataType
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_normalizer import PromptRequestPiece, PromptNormalizer
from pyrit.prompt_normalizer.prompt_request_piece import PromptRequestPieces
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter, NoOpConverter

logger = logging.getLogger(__name__)


class PromptSendingOrchestrator(Orchestrator):
    """
    This orchestrator takes a set of prompts, converts them using the list of PromptConverters,
    and sends them to a target.
    """

    def __init__(
        self,
        prompt_target: PromptTarget,
        prompt_converters: Optional[list[PromptConverter]] = None,
        memory: MemoryInterface = None,
        batch_size: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            prompt_target (PromptTarget): The target for sending prompts.
            prompt_converters (list[PromptConverter], optional): List of prompt converters. These are stacked in
                                    the order they are provided. E.g. the output of converter1 is the input of
                                    converter2.
            memory (MemoryInterface, optional): The memory interface. Defaults to None.
            batch_size (int, optional): The (max) batch size for sending prompts. Defaults to 10.
        """
        super().__init__(prompt_converters=prompt_converters, memory=memory, verbose=verbose)

        self._prompt_normalizer = PromptNormalizer(memory=self._memory)

        self._prompt_target = prompt_target
        self._prompt_target._memory = self._memory

        self._batch_size = batch_size

    def send_text_prompts(self, prompts: list[str]):
        """
        Sends the strings to the prompt target
        """

        responses = []

        for prompt in prompts:
            request = self._create_prompt_request(prompt, "text")
            response = self._prompt_normalizer.send_prompt(
                request=request,
                target=self._prompt_target,
                labels=self._global_memory_labels,
                orchestrator=self,
                verbose=self._verbose)
            
            responses.append(response)

        return responses

    async def send_prompts_batch_async(self, prompts: list[str]):
        """
        Sends the prompt to the prompt target.
        """

        requests = []
        for prompt in prompts:
            requests.append(self._create_prompt_request(prompt, "text"))

        await self._prompt_normalizer.send_prompt_batch_to_target_async(
            requests=requests,
            target=self._prompt_target,
            labels=self._global_memory_labels,
            orchestrator=self,
            verbose=self._verbose,
            batch_size=self._batch_size)

    def _create_prompt_request(self, text: str, prompt_type: PromptDataType):

        request_piece = PromptRequestPiece(
            prompt_converters=self._prompt_converters,
            prompt_text=text,
            prompt_data_type=prompt_type,
            )
        
        request = PromptRequestPieces([request_piece])
        return request

    def get_memory(self):
        """
        Retrieves the memory associated with the prompt normalizer.
        """
        id = self._prompt_normalizer.id
        return self._memory.get_prompt_entries_with_normalizer_id(normalizer_id=id)
