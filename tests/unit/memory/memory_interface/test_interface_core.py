# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from pyrit.memory import MemoryInterface


def test_memory(sqlite_instance: MemoryInterface):
    assert sqlite_instance
