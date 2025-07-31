# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Backward compatibility module for MemoryInterface.

The MemoryInterface and its mixins have been moved to the memory_interface/ subdirectory
for better organization. This module re-exports MemoryInterface to maintain
backward compatibility for existing imports.
"""

# Re-export for backward compatibility
from pyrit.memory.memory_interface import MemoryInterface  # type: ignore

__all__ = ["MemoryInterface"]
