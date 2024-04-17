# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import logging

from typing import Optional

from pyrit.memory import MemoryInterface, DuckDBMemory
from pyrit.models import PromptDataType, Identifier
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest, NormalizerRequestPiece

logger = logging.getLogger(__name__)


class Orchestrator(abc.ABC, Identifier):

    _memory: MemoryInterface

    def __init__(
        self,
        *,
        prompt_converters: Optional[list[PromptConverter]] = None,
        memory: Optional[MemoryInterface] = None,
        memory_labels: dict[str, str] = {},
        verbose: bool = False,
    ):
        self._prompt_converters = prompt_converters if prompt_converters else []
        self._memory = memory or DuckDBMemory()
        self._verbose = verbose

        self._global_memory_labels = memory_labels if memory_labels else {}

        if self._verbose:
            logging.basicConfig(level=logging.INFO)

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

    def _create_normalizer_request(
        self, prompt_text: str, prompt_type: PromptDataType = "text", converters=None, metadata=None
    ):

        if converters is None:
            converters = self._prompt_converters

        request_piece = NormalizerRequestPiece(
            prompt_converters=converters,
            prompt_text=prompt_text,
            prompt_data_type=prompt_type,
            metadata=metadata,
        )

        request = NormalizerRequest([request_piece])
        return request

    def get_memory(self):
        """
        Retrieves the memory associated with this orchestrator.
        """
        return self._memory.get_prompt_entries_by_orchestrator(orchestrator_id=self)

    def get_identifier(self):
        orchestrator_dict = {}
        orchestrator_dict["__type__"] = self.__class__.__name__
        orchestrator_dict["__module__"] = self.__class__.__module__
        orchestrator_dict["id"] = str(id(self))
        return orchestrator_dict
