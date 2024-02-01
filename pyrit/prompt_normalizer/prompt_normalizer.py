# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from uuid import uuid4
from pyrit.memory import MemoryInterface
from pyrit.prompt_normalizer import Prompt


class PromptNormalizer(abc.ABC):
    memory: MemoryInterface

    def __init__(self, memory: MemoryInterface, prompts: list[Prompt] = []) -> None:
        self.memory = memory
        self.id = str(uuid4())
        self.prompts = prompts

    def send_prompt(self, prompt: Prompt):
        """
        Sends a prompt to the prompt targets.
        """
        prompt.send_prompt(normalizer_id=self.id)
