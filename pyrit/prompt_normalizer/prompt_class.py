# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from pyrit.memory import MemoryInterface
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter


class Prompt(abc.ABC):
    memory: MemoryInterface

    def __init__(
        self, prompt_target: PromptTarget, prompt_converter: PromptConverter, prompt_text: str, conversation_id: str
    ) -> None:
        if not isinstance(prompt_target, PromptTarget):
            raise ValueError("prompt_target must be a PromptTarget")

        if not isinstance(prompt_converter, PromptConverter):
            raise ValueError("prompt_converter must be a Promptconverter")

        if not isinstance(prompt_text, str):
            raise ValueError("prompt_text must be a str")

        if not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a str")

        self.prompt_target = prompt_target
        self.prompt_converter = prompt_converter
        self.prompt_text = prompt_text
        self.conversation_id = conversation_id

    def send_prompt(self, normalizer_id: str):
        """
        Sends the prompt to the prompt target.
        """
        self.prompt_target.send_prompt(
            normalized_prompt=self.prompt_converter.convert(self.prompt_text),
            conversation_id=self.conversation_id,
            normalizer_id=normalizer_id,
        )
