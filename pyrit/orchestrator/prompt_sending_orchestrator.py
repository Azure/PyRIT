# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from typing import Optional
from uuid import uuid4

from pyrit.memory import MemoryInterface
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_normalizer import PromptRequestPiece, PromptNormalizer
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
        include_original_prompts: bool = False,
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
            include_original_prompts (bool, optional): Whether to include original prompts to send to the target
                                    before converting. This does not include intermediate steps from converters.
                                    Defaults to False.
        """
        super().__init__(prompt_converters=prompt_converters, memory=memory, verbose=verbose)

        self._prompt_normalizer = PromptNormalizer(memory=self._memory)

        self._prompt_target = prompt_target
        self._prompt_target._memory = self._memory
        self._include_original_prompts = include_original_prompts

        self.batch_size = batch_size

    def send_prompts(self, prompts: list[str]):
        """
        Sends the prompt to the prompt target.
        """
        responses = []

        normalized_prompts = self._coalesce_prompts(prompts)

        for prompt in normalized_prompts:
            responses.append(self._prompt_normalizer.send_prompt(prompt=prompt))

        return responses

    async def send_prompts_batch_async(self, prompts: list[str]):
        """
        Sends the prompt to the prompt target.
        """

        normalized_prompts = self._coalesce_prompts(prompts)

        await self._prompt_normalizer.send_prompt_batch_async(normalized_prompts, batch_size=self.batch_size)

    def _coalesce_prompts(self, prompts):
        normalized_prompts = []

        for prompt_text in prompts:
            if self._include_original_prompts:
                original_prompt = PromptRequestPiece(
                    prompt_target=self._prompt_target,
                    prompt_converters=[NoOpConverter()],
                    prompt_text=prompt_text,
                    conversation_id=str(uuid4()),
                )

                normalized_prompts.append(original_prompt)

            converted_prompt = PromptRequestPiece(
                prompt_target=self._prompt_target,
                prompt_converters=self._prompt_converters,
                prompt_text=prompt_text,
                conversation_id=str(uuid4()),
            )
            normalized_prompts.append(converted_prompt)

        logger.info(
            f"Sending {len(normalized_prompts)} prompts to the prompt target.",
        )

        for normalized_prompt in normalized_prompts:
            logger.info(f"Prompt: {normalized_prompt}")

        return normalized_prompts

    def get_memory(self):
        """
        Retrieves the memory associated with the prompt normalizer.
        """
        id = self._prompt_normalizer.id
        return self._memory.get_prompt_entries_with_normalizer_id(normalizer_id=id)
