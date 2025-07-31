# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Memory interface package containing the MemoryInterface and its mixins."""

from pyrit.memory.memory_interface.interface import MemoryInterface
from pyrit.memory.memory_interface.protocol import MemoryInterfaceProtocol

__all__ = ["MemoryInterface", "MemoryInterfaceProtocol"]
