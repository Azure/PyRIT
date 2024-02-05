# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from pyrit.memory import MemoryInterface
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_transformer import PromptTransformer


class Prompt(abc.ABC):
    memory: MemoryInterface

    def __init__(
        self, prompt_target: PromptTarget, prompt_transformer: PromptTransformer, prompt_text: str, conversation_id: str
    ) -> None:
        if not isinstance(prompt_target, PromptTarget):
            raise ValueError("prompt_target must be a PromptTarget")

        if not isinstance(prompt_transformer, PromptTransformer):
            raise ValueError("prompt_transformer must be a PromptTransformer")

        if not isinstance(prompt_text, str):
            raise ValueError("prompt_text must be a str")

        if not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a str")

        self.prompt_target = prompt_target
        self.prompt_transformer = prompt_transformer
        self.prompt_text = prompt_text
        self.conversation_id = conversation_id

    def send_prompt(self, normalizer_id: str):
        """
        Sends the prompt to the prompt target.
        """
        self.prompt_target.send_prompt(
            normalized_prompt=self.prompt_transformer.transform(self.prompt_text),
            conversation_id=self.conversation_id,
            normalizer_id=normalizer_id,
        )
