# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import logging
from typing import Any, Dict, Optional

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
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._memory = CentralMemory.get_memory_instance()
        self._verbose = verbose
        self._max_requests_per_minute = max_requests_per_minute
        self._endpoint = endpoint
        self._model_name = model_name
        # Store any custom metadata provided for identifier purposes, including safety (safe vs. unsafe),
        # specific guardrails, fine-tuning information, version, etc.
        self._custom_metadata = custom_metadata

        if self._verbose:
            logging.basicConfig(level=logging.INFO)

    @abc.abstractmethod
    async def send_prompt_async(self, *, message: Message) -> Message:
        """
        Sends a normalized prompt async to the prompt target.
        """

    @abc.abstractmethod
    def _validate_request(self, *, message: Message) -> None:
        """
        Validates the provided message.
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
        Dispose database engine to release database connections and resources.
        """
        self._memory.dispose_engine()

    def get_identifier(self) -> Dict[str, Any]:
        public_attributes: Dict[str, Any] = {}
        public_attributes["__type__"] = self.__class__.__name__
        public_attributes["__module__"] = self.__class__.__module__
        if self._endpoint:
            public_attributes["endpoint"] = self._endpoint
        if self._model_name:
            public_attributes["model_name"] = self._model_name
        if self._custom_metadata:
            public_attributes["custom_metadata"] = self._custom_metadata
        return public_attributes

    def get_eval_identifier(self) -> Dict[str, Any]:
        """
        Get an identifier for scorer evaluation purposes.

        This method returns only the essential attributes needed for scorer evaluation
        and registry tracking.

        Returns:
            Dict[str, Any]: A dictionary containing identification attributes for scorer evaluation purposes.
        """
        eval_identifier = self.get_identifier()
        if "__module__" in eval_identifier:
            del eval_identifier["__module__"]
        if "endpoint" in eval_identifier:
            del eval_identifier["endpoint"]

        return eval_identifier
