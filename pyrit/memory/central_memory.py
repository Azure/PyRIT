# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from pyrit.memory.memory_interface import MemoryInterface

logger = logging.getLogger(__name__)


class CentralMemory:
    """
    Provides a centralized memory instance across the framework. The provided memory
    instance will be reused for future calls.
    """

    _memory_instance: MemoryInterface = None

    @classmethod
    def set_memory_instance(cls, passed_memory: MemoryInterface) -> None:
        """
        Set a provided memory instance as the central instance for subsequent calls.

        Args:
            passed_memory (MemoryInterface): The memory instance to set as the central instance.
        """
        cls._memory_instance = passed_memory
        logger.info(f"Central memory instance set to: {type(cls._memory_instance).__name__}")

    @classmethod
    def get_memory_instance(cls) -> MemoryInterface:
        """
        Returns a centralized memory instance.
        """
        if cls._memory_instance:
            logger.info(f"Using existing memory instance: {type(cls._memory_instance).__name__}")
            return cls._memory_instance
        else:
            raise ValueError("Central memory instance has not been set. Use `set_memory_instance` to set it.")
