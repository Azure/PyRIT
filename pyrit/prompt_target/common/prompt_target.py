# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import logging
from typing import Optional

from pyrit.memory import MemoryInterface, CentralMemory
from pyrit.models import PromptRequestResponse, Identifier


logger = logging.getLogger(__name__)


class PromptTarget(abc.ABC, Identifier):
    _memory: MemoryInterface

    """
    A list of PromptConverters that are supported by the prompt target.
    An empty list implies that the prompt target supports all converters.
    """
    supported_converters: list

    def __init__(self, verbose: bool = False, max_requests_per_minute: Optional[int] = None) -> None:
        self._memory = CentralMemory.get_memory_instance()
        self._verbose = verbose
        self._max_requests_per_minute = max_requests_per_minute

        if self._verbose:
            logging.basicConfig(level=logging.INFO)

    @abc.abstractmethod
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """
        Sends a normalized prompt async to the prompt target.
        """

    @abc.abstractmethod
    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        """
        Validates the provided prompt request response
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

    def get_identifier(self):
        public_attributes = {}
        public_attributes["__type__"] = self.__class__.__name__
        public_attributes["__module__"] = self.__class__.__module__
        return public_attributes
