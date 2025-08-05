# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.memory import MemoryInterface 


def test_memory(duckdb_instance: MemoryInterface):
    assert duckdb_instance