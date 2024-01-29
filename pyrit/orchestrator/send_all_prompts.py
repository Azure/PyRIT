# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from pathlib import Path
from uuid import uuid4
from pyrit.interfaces import SupportTextClassification

from pyrit.memory import MemoryInterface, file_memory
from pyrit.prompt_normalizer.prompt import Prompt
from pyrit.prompt_normalizer.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target.prompt_target import PromptTarget
from pyrit.prompt_transformer.prompt_transformer import PromptTransformer

class SendAllPromptsOrchestrator():
    """
    This orchestrator takes a set of prompts, transforms them, and attempts them on a target
    """

    def __init__(self,
                 prompts: list[str],
                 prompt_target: PromptTarget,
                 prompt_transformer: PromptTransformer = None,
                 classifier: SupportTextClassification = None,
                 memory: MemoryInterface = None) -> None:

        self.prompts = list[str]
        self.prompt_target = prompt_target
        self.classifier = classifier

        if not self.prompt_transformer:
            self.prompt_transformer = PromptTransformer()
        if not self.classifier:
            #TODO build default classifier factory
            raise("Not implemented")

        self.prompt_transformer = prompt_transformer if prompt_transformer else PromptTransformer()
        self.memory = memory if memory else file_memory()




    def send_prompts(self):
        """
        Sends the prompt to the prompt target.
        """
        # TODO
        # Create list of prompts
        # Create prompt normalizer
        # for each prompt, send it, and use classifier to check success

        self.prompt_normalizer = PromptNormalizer(memory=self.memory)


        self.prompt_normalizer.send_prompts()
        # use classifier to check success

    def _create_prompts(self):

