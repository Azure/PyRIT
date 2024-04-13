# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import json
import logging

from pyrit.memory import MemoryInterface, DuckDBMemory
from pyrit.models import PromptRequestResponse


class PromptTarget(abc.ABC):
    _memory: MemoryInterface

    """
    A list of PromptConverters that are supported by the prompt target.
    An empty list implies that the prompt target supports all converters.
    """
    supported_converters: list

    def __init__(self, memory: MemoryInterface, verbose: bool = False) -> None:
        self._memory = memory if memory else DuckDBMemory()

    @abc.abstractmethod
    def send_prompt(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """
        Sends a normalized prompt to the prompt target and adds the request and response to memory
        """

    @abc.abstractmethod
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """
        Sends a normalized prompt async to the prompt target.
        """

    def __enter__(self):
        """Enter the runtime context related to this object."""
        return self  # You can return self or another object that should be used in the with-statement.

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context and perform any cleanup actions."""
        self.dispose_db_engine()

    def dispose_db_engine(self) -> None:
        """
        Dispose DuckDB database engine to release database connections and resources.
        """
        self._memory.dispose_engine()

    def to_dict(self):
        public_attributes = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        public_attributes["__type__"] = self.__class__.__name__
        public_attributes["__module__"] = self.__class__.__module__
        return public_attributes
