# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import logging


from typing import Optional
from uuid import uuid4

from pyrit.memory import MemoryInterface, FileMemory
from pyrit.prompt_normalizer import Prompt, PromptNormalizer
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter, NoOpConverter

logger = logging.getLogger(__name__)

class Orchestrator:

    _memory: MemoryInterface

    def __init__(
        self,
        *,
        prompt_converters: Optional[list[PromptConverter]] = None,
        memory: Optional[MemoryInterface] = None,
        memory_labels: list[str] = [],
        verbose: bool = False,
    ):
        self._prompt_converters = prompt_converters if prompt_converters else [NoOpConverter()]
        self._memory = memory or FileMemory()
        self._global_memory_labels = memory_labels
        self._verbose = verbose
        self._logger = logging.getLogger(__name__)


        if self._verbose:
            logging.basicConfig(level=logging.INFO)

        if self.requires_one_to_one_converters():
            # Ensure that all converters return exactly one prompt.
            # Otherwise, there will be more than 1 conversation to manage.
            one_to_many_converters = []
            for converter in self._prompt_converters:
                if not converter.is_one_to_one_converter():
                    one_to_many_converters.append(str(converter))
            if one_to_many_converters:
                one_to_many_converters_str = ", ".join(one_to_many_converters)
                raise ValueError(f"The following converters create more than one prompt: {one_to_many_converters_str}")


    @abc.abstractmethod
    def requires_one_to_one_converters(self) -> bool:
        """Returns True if all prompt_converters must be 1:1, False otherwise."""
        return False