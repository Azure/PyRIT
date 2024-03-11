# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional
from uuid import uuid4

from pyrit.memory import MemoryInterface, FileMemory
from pyrit.prompt_normalizer import Prompt, PromptNormalizer
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter, NoOpConverter


class PromptSendingOrchestrator:
    """
    This orchestrator takes a set of prompts, converts them using the list of PromptConverters,
    and sends them to a target.
    """

    def __init__(
        self,
        prompt_target: PromptTarget,
        prompt_converters: Optional[list[PromptConverter]] = None,
        memory: MemoryInterface = None,
        include_original_prompts: bool = False,
    ) -> None:
        self._prompt_converters = prompt_converters if prompt_converters else [NoOpConverter()]
        self._memory = memory if memory else FileMemory()
        self._prompt_normalizer = PromptNormalizer(memory=self._memory)

        self._prompt_target = prompt_target
        self._prompt_target._memory = self._memory
        self._include_original_prompts = include_original_prompts

    def send_prompts(self, prompts: list[str]):
        """
        Sends the prompt to the prompt target.
        """
        responses = []
        for prompt_text in prompts:
            if self._include_original_prompts:
                original_prompt = Prompt(
                    prompt_target=self._prompt_target,
                    prompt_converters=[NoOpConverter()],
                    prompt_text=prompt_text,
                    conversation_id=str(uuid4()),
                )

                self._prompt_normalizer.send_prompt(prompt=original_prompt)

            converted_prompt = Prompt(
                prompt_target=self._prompt_target,
                prompt_converters=self._prompt_converters,
                prompt_text=prompt_text,
                conversation_id=str(uuid4()),
            )

            responses.append(
                self._prompt_normalizer.send_prompt(prompt=converted_prompt))

        return responses

    def get_memory(self):
        """
        Retrieves the memory associated with the prompt normalizer.
        """
        id = self._prompt_normalizer.id
        return self._memory.get_memories_with_normalizer_id(normalizer_id=id)
