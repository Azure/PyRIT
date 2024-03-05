# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from uuid import uuid4

from pyrit.memory import MemoryInterface, FileMemory
from pyrit.prompt_normalizer import Prompt, PromptNormalizer
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter, NoOpConverter


class PromptSendingOrchestrator:
    """
    This orchestrator takes a set of prompts, transforms them, and sends them to a target.
    """

    def __init__(
        self, prompt_target: PromptTarget, prompt_converter: PromptConverter = None, memory: MemoryInterface = None
    ) -> None:
        self._prompt_converter = prompt_converter if prompt_converter else NoOpConverter()
        self._memory = memory if memory else FileMemory()
        self._prompt_normalizer = PromptNormalizer(memory=self._memory)

        self._prompt_target = prompt_target
        self._prompt_target._memory = self._memory

    def send_prompts(self, prompts: list[str]):
        """
        Sends the prompt to the prompt target.
        """
        responses = []
        for prompt_text in prompts:
            prompt = Prompt(
                prompt_target=self._prompt_target,
                prompt_converter=self._prompt_converter,
                prompt_text=prompt_text,
                conversation_id=str(uuid4()),
            )

            responses.append(self._prompt_normalizer.send_prompt(prompt=prompt))
        return responses

    def get_memory(self):
        return self._memory.get_memories_with_normalizer_id(normalizer_id=self._prompt_normalizer.id)
