# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile
from unittest import mock

import pytest

from pyrit.common.apply_defaults import reset_default_values
from pyrit.setup import IN_MEMORY, initialize_pyrit_async
from pyrit.setup.initialization import _load_initializers_from_scripts


class TestLoadInitializersFromScripts:
    """Tests for _load_initializers_from_scripts function."""

    def test_load_initializer_from_script(self):
        """Test loading an initializer from a Python script."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
from pyrit.setup.initializers import PyRITInitializer

class TestInitializer(PyRITInitializer):
    @property
    def name(self) -> str:
        return "Test Initializer"

    @property
    def description(self) -> str:
        return "Test description"

    async def initialize_async(self) -> None:
        pass
"""
            )
            script_path = f.name

        try:
            initializers = _load_initializers_from_scripts(script_paths=[script_path])
            assert len(initializers) == 1
            assert initializers[0].name == "Test Initializer"
        finally:
            os.unlink(script_path)

    def test_script_not_found_raises_error(self):
        """Test that FileNotFoundError is raised for non-existent script."""
        with pytest.raises(FileNotFoundError):
            _load_initializers_from_scripts(script_paths=["nonexistent_script.py"])


class TestInitializePyrit:
    """Tests for initialize_pyrit function - basic orchestration tests."""

    def setup_method(self) -> None:
        """Clear default values before each test."""
        reset_default_values()

    @pytest.mark.asyncio
    @mock.patch("pyrit.memory.central_memory.CentralMemory.set_memory_instance")
    @mock.patch("pyrit.setup.initialization._load_environment_files")
    async def test_initialize_basic(self, mock_load_env, mock_set_memory):
        """Test basic initialization."""
        await initialize_pyrit_async(memory_db_type=IN_MEMORY)

        mock_load_env.assert_called_once()
        mock_set_memory.assert_called_once()

    @pytest.mark.asyncio
    @mock.patch("pyrit.memory.central_memory.CentralMemory.set_memory_instance")
    @mock.patch("pyrit.setup.initialization._load_environment_files")
    async def test_initialize_with_script(self, mock_load_env, mock_set_memory):
        """Test initialization with a script."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
from pyrit.setup.initializers import PyRITInitializer

class ScriptInit(PyRITInitializer):
    @property
    def name(self) -> str:
        return "Script"

    @property
    def description(self) -> str:
        return "From script"

    async def initialize_async(self) -> None:
        pass
"""
            )
            script_path = f.name

        try:
            await initialize_pyrit_async(memory_db_type=IN_MEMORY, initialization_scripts=[script_path])
            mock_load_env.assert_called_once()
            mock_set_memory.assert_called_once()
        finally:
            os.unlink(script_path)

    @pytest.mark.asyncio
    async def test_invalid_memory_type_raises_error(self):
        """Test that invalid memory type raises ValueError."""
        with pytest.raises(ValueError, match="is not a supported type"):
            await initialize_pyrit_async(memory_db_type="InvalidType")  # type: ignore
