# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import asyncio
from uuid import uuid4
from pyrit.memory import MemoryInterface
from pyrit.prompt_normalizer.prompt_class import Prompt


class PromptNormalizer(abc.ABC):
    _memory: MemoryInterface

    def __init__(self, *, memory: MemoryInterface, verbose=False) -> None:
        self._memory = memory
        self.id = str(uuid4())

    def send_prompt(self, prompt: Prompt, memory_labels: list[str] | None = None) -> list[str]:
        """
        Sends a prompt to the prompt targets.
        """
        return prompt.send_prompt(normalizer_id=self.id, memory_labels=memory_labels)

    async def send_prompt_batch_async(self, prompts: list[Prompt], memory_labels: list[str] | None = None, batch_size: int = 10):
        """
        Sends a batch of prompts to a target
        """
        results = []

        for prompts_batch in self._chunked_prompts(prompts, batch_size):
            tasks = [prompt.send_prompt_async(normalizer_id=self.id, memory_labels=memory_labels) for prompt in prompts_batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        return results

    def _chunked_prompts(self, prompts, size):
        for i in range(0, len(prompts), size):
            yield prompts[i : i + size]
