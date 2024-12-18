# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pytest
from unittest.mock import patch, MagicMock

from pyrit.common import default_values
from pyrit.memory import AzureSQLMemory, DuckDBMemory, CentralMemory


@pytest.fixture(autouse=True)
def reset_memory_instance():
    """Reset CentralMemory instance before each test."""
    CentralMemory._memory_instance = None


@patch("pyrit.common.default_values.get_non_required_value")
@patch("pyrit.memory.AzureSQLMemory.__init__", return_value=None)
def test_get_memory_instance_with_azure_sql(mock_azure_init, mock_get_value):
    """Test that CentralMemory initializes with AzureSQLMemory when Azure configuration is present."""
    mock_get_value.side_effect = lambda env_var_name, passed_value: (
        "mock_value"
        if env_var_name in ["AZURE_SQL_DB_CONNECTION_STRING", "AZURE_STORAGE_ACCOUNT_RESULTS_CONTAINER_URL"]
        else ""
    )

    memory_instance = CentralMemory.get_memory_instance()
    assert isinstance(memory_instance, AzureSQLMemory)


@patch("pyrit.common.default_values.get_non_required_value")
def test_get_memory_instance_with_duckdb(mock_get_value):
    """Test that CentralMemory initializes with DuckDBMemory when Azure configuration is missing."""
    mock_get_value.side_effect = lambda env_var_name, passed_value: ""

    memory_instance = CentralMemory.get_memory_instance()
    assert isinstance(memory_instance, DuckDBMemory)


def test_set_memory_instance():
    """Test that setting a memory instance overrides the default behavior."""
    mock_memory_instance = MagicMock(spec=DuckDBMemory)
    CentralMemory.set_memory_instance(mock_memory_instance)

    memory_instance = CentralMemory.get_memory_instance()
    assert memory_instance is mock_memory_instance


@patch("pyrit.common.default_values.get_non_required_value")
def test_memory_instance_reusability(mock_get_value):
    """Test that CentralMemory reuses the same instance on subsequent calls."""
    mock_get_value.side_effect = lambda env_var_name, passed_value: ""

    first_instance = CentralMemory.get_memory_instance()
    second_instance = CentralMemory.get_memory_instance()

    assert first_instance is second_instance


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
