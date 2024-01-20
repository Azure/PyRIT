# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from pyrit.memory import MemoryInterface
from pyrit.prompt_target.prompt_target import PromptTarget
from pyrit.prompt_transformer.prompt_transformer import PromptTransformer

class Prompt(abc.ABC):
    memory: MemoryInterface

    def __init__(self,
                 memory: MemoryInterface,
                 prompt_target: PromptTarget,
                 prompt_transformer: PromptTransformer,
                 prompt_text: str) -> None:

        if not isinstance(memory, MemoryInterface):
            raise ValueError("memory must be a MemoryInterface")

        if not isinstance(prompt_target, PromptTarget):
            raise ValueError("prompt_target must be a PromptTarget")

        if not isinstance(prompt_transformer, PromptTransformer):
            raise ValueError("prompt_transformer must be a PromptTransformer")

        if not isinstance(prompt_text, str):
            raise ValueError("prompt_text must be a str")

        self.memory = memory
        self.prompt_target = prompt_target
        self.prompt_transformer = prompt_transformer
        self.prompt_text = prompt_text

    def send_prompt(self):
        """
        Sends the prompt to the prompt target.
        """
        self.prompt_target.send_prompt(self.prompt_transformer.transform(self.prompt_text))
