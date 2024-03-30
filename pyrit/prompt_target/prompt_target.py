# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import json
from pyrit.memory import MemoryInterface
from pyrit.memory import DuckDBMemory


class PromptTarget(abc.ABC):
    _memory: MemoryInterface

    """
    A list of PromptConverters that are supported by the prompt target.
    An empty list implies that the prompt target supports all converters.
    """
    supported_converters: list

    def __init__(self, memory: MemoryInterface) -> None:
        self._memory = memory if memory else DuckDBMemory()

    @abc.abstractmethod
    def send_prompt(self, *, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        """
        Sends a normalized prompt to the prompt target.
        """

    @abc.abstractmethod
    async def send_prompt_async(self, *, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        """
        Sends a normalized prompt async to the prompt target.
        """

    def to_json(self):
        public_attributes = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        public_attributes["__type__"] = self.__class__.__name__
        public_attributes["__module__"] = self.__class__.__module__
        return json.dumps(public_attributes)
