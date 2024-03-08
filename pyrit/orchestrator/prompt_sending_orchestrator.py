# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from uuid import uuid4

from pyrit.memory import MemoryInterface, file_memory
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
        prompt_converters: list[PromptConverter] = None,
        memory: MemoryInterface = None,
        batch_size: int = 10,
        include_original_prompts: bool = False,
    ) -> None:
        """
        Initialize the PromptSendingOrchestrator.

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
        self.prompts = list[str]
        self.prompt_target = prompt_target
        self.batch_size = batch_size

        self.prompt_converters = prompt_converters if prompt_converters else [NoOpConverter()]
        self.memory = memory if memory else file_memory.FileMemory()
        self.prompt_normalizer = PromptNormalizer(memory=self.memory)

        self.prompt_target.memory = self.memory
        self.include_original_prompts = include_original_prompts

    def send_prompts(self, prompts: list[str]):
        """
        Sends the prompt to the prompt target.
        """

        normalized_prompts = self._coallesce_prompts(prompts)

        for prompt in normalized_prompts:
            self.prompt_normalizer.send_prompt(prompt)

    async def send_parallel_prompts_async(self, prompts: list[str]):
        """
        Sends the prompt to the prompt target.
        """

        normalized_prompts = self._coallesce_prompts(prompts)

        await self.prompt_normalizer.send_prompt_batch_async(normalized_prompts, batch_size=self.batch_size)

    def get_memory(self):
        """
        Retrieves the memory associated with the prompt normalizer.
        """
        id = self.prompt_normalizer.id
        return self.memory.get_memories_with_normalizer_id(normalizer_id=id)

    def _coallesce_prompts(self, prompts):
        normalized_prompts = []

        for prompt_text in prompts:
            if self.include_original_prompts:
                original_prompt = Prompt(
                    prompt_target=self.prompt_target,
                    prompt_converters=[NoOpConverter()],
                    prompt_text=prompt_text,
                    conversation_id=str(uuid4()),
                )

                normalized_prompts.append(original_prompt)

            converted_prompt = Prompt(
                prompt_target=self.prompt_target,
                prompt_converters=self.prompt_converters,
                prompt_text=prompt_text,
                conversation_id=str(uuid4()),
            )

            normalized_prompts.append(converted_prompt)
        return normalized_prompts
