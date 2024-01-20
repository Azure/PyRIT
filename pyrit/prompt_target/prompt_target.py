# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from pyrit.memory import MemoryInterface

class PromptTarget(abc.ABC):
    memory: MemoryInterface
    supported_transformers: list

    def __init__(self, memory: MemoryInterface, session_id: str) -> None:
        self.memory = memory
        self.session_id = session_id

    @abc.abstractmethod
    def send_prompt(self, normalized_prompt: str) -> None:
        """
        Sennds a normalized prompt to the prompt target.
        """