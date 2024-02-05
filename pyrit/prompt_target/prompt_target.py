# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from pyrit.memory import MemoryInterface


class PromptTarget(abc.ABC):
    memory: MemoryInterface

    """
    A list of transformers that are supported by the prompt target.
    An empty list implies that the prompt target supports all transformers.
    """
    supported_transformers: list

    def __init__(self, memory: MemoryInterface) -> None:
        self.memory = memory

    @abc.abstractmethod
    def set_system_prompt(self, prompt: str, conversation_id: str, normalizer_id: str) -> None:
        """
        Sets the system prompt for the prompt target
        """

    @abc.abstractmethod
    def send_prompt(self, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        """
        Sends a normalized prompt to the prompt target.
        """
