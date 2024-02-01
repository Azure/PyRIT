# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from uuid import uuid4

from pyrit.memory import MemoryInterface, file_memory
from pyrit.prompt_normalizer import Prompt, PromptNormalizer
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_transformer import PromptTransformer


class SendAllPromptsOrchestrator:
    """
    This orchestrator takes a set of prompts, transforms them, and attempts them on a target
    """

    def __init__(
        self, prompt_target: PromptTarget, prompt_transformer: PromptTransformer = None, memory: MemoryInterface = None
    ) -> None:
        self.prompts = list[str]
        self.prompt_target = prompt_target

        self.prompt_transformer = prompt_transformer if prompt_transformer else PromptTransformer()
        self.memory = memory if memory else file_memory.FileMemory()
        self.prompt_normalizer = PromptNormalizer(memory=self.memory)

        self.prompt_target.memory = self.memory

    def send_prompts(self, prompts: list[str]):
        """
        Sends the prompt to the prompt target.
        """
        for str_prompt in prompts:
            prompt = Prompt(
                prompt_target=self.prompt_target,
                prompt_transformer=self.prompt_transformer,
                prompt_text=str_prompt,
                conversation_id=str(uuid4()),
            )

            self.prompt_normalizer.send_prompt(prompt=prompt)

    def get_memory(self):
        id = self.prompt_normalizer.id
        return self.memory.get_memories_with_normalizer_id(normalizer_id=id)
