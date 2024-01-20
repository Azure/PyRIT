# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from pyrit.memory import MemoryInterface
from pyrit.prompt_normalizer.prompt import Prompt
from pyrit.prompt_target.prompt_target import PromptTarget
from pyrit.prompt_transformer.prompt_transformer import PromptTransformer

class PromptNormalizer(abc.ABC):
    memory: MemoryInterface

    def __init__(self,
                 memory: MemoryInterface,
                 session_id = None) -> None:

        self.memory = memory
        self.session_id = session_id

    def send_prompt(self, prompt: Prompt):
        """
        Sends the prompt to the prompt target.
        """
        self.prompt_target.send_prompt(self.prompt_transformer.transform(self.prompt_text))

    def send_prompts(self):
        """
        Sends the prompt to the prompt target.
        """
        self.prompt_target.send_prompt(self.prompt_transformer.transform(self.prompt_text))
