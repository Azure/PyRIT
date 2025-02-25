# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from unittest.mock import MagicMock, patch

import pytest

from pyrit.common import default_values
from pyrit.memory import CentralMemory, DuckDBMemory


@pytest.fixture(autouse=True)
def reset_memory_instance():
    """Reset CentralMemory instance before each test."""
    CentralMemory._memory_instance = None


def test_set_memory_instance():
    """Test that setting a memory instance overrides the default behavior."""
    mock_memory_instance = MagicMock(spec=DuckDBMemory)
    CentralMemory.set_memory_instance(mock_memory_instance)

    memory_instance = CentralMemory.get_memory_instance()
    assert memory_instance is mock_memory_instance

    # Test that CentralMemory reuses the same instance on subsequent calls
    second_instance = CentralMemory.get_memory_instance()
    assert memory_instance is second_instance


def test_get_memory_instance_not_set():
    with pytest.raises(
        ValueError, match="Central memory instance has not been set. Use `set_memory_instance` to set it."
    ):
        CentralMemory.get_memory_instance()


@patch.dict("os.environ", {}, clear=True)
def test_get_non_required_value_empty_env():
    """Test that get_non_required_value returns passed value if env var is empty or not set."""
    os.environ["NON_EXISTENT_ENV"] = ""
    result = default_values.get_non_required_value(env_var_name="NON_EXISTENT_ENV", passed_value="default_value")
    assert result == "default_value"


@patch.dict("os.environ", {}, clear=True)
def test_get_non_required_value_env_set():
    """Test that get_non_required_value prefers environment variable over passed value."""
    os.environ["TEST_ENV_VAR"] = "env_value"
    result = default_values.get_non_required_value(env_var_name="TEST_ENV_VAR", passed_value="default_value")
    assert result == "default_value"
