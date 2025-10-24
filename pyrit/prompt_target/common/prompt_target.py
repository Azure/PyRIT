# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import logging
from typing import Optional

from pyrit.memory import CentralMemory, MemoryInterface
from pyrit.models import Identifier, Message

logger = logging.getLogger(__name__)


class PromptTarget(abc.ABC, Identifier):
    _memory: MemoryInterface

    """
    A list of PromptConverters that are supported by the prompt target.
    An empty list implies that the prompt target supports all converters.
    """
    supported_converters: list

    def __init__(
        self,
        verbose: bool = False,
        max_requests_per_minute: Optional[int] = None,
        endpoint: str = "",
        model_name: str = "",
    ) -> None:
        self._memory = CentralMemory.get_memory_instance()
        self._verbose = verbose
        self._max_requests_per_minute = max_requests_per_minute
        self._endpoint = endpoint
        self._model_name = model_name

        if self._verbose:
            logging.basicConfig(level=logging.INFO)

    @abc.abstractmethod
    async def send_prompt_async(self, *, prompt_request: Message) -> Message:
        """
        Sends a normalized prompt async to the prompt target.
        """

    @abc.abstractmethod
    def _validate_request(self, *, prompt_request: Message) -> None:
        """
        Validates the provided message
        """

    def set_model_name(self, *, model_name: str) -> None:
        """
        Set the model name for this target.

        Args:
            model_name (str): The model name to set.
        """
        self._model_name = model_name

    def dispose_db_engine(self) -> None:
        """
        Dispose DuckDB database engine to release database connections and resources.
        """
        self._memory.dispose_engine()

    def get_identifier(self) -> dict:
        public_attributes = {}
        public_attributes["__type__"] = self.__class__.__name__
        public_attributes["__module__"] = self.__class__.__module__
        if self._endpoint:
            public_attributes["endpoint"] = self._endpoint
        if self._model_name:
            public_attributes["model_name"] = self._model_name
        return public_attributes
