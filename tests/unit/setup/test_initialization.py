# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pathlib
import tempfile
from unittest import mock

import pytest

from pyrit.common.apply_defaults import reset_default_values
from pyrit.setup import IN_MEMORY, initialize_pyrit_async
from pyrit.setup.initialization import (
    _find_project_root,
    _load_initializers_from_scripts,
)


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
    """Tests for initialize_pyrit_async function - basic orchestration tests."""

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


class TestFindProjectRoot:
    """Tests for _find_project_root function."""

    def test_find_project_root_with_env_file(self):
        """Test finding project root when .env file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a directory structure: tmpdir/.env and tmpdir/subdir/
            tmpdir_path = pathlib.Path(tmpdir)
            env_file = tmpdir_path / ".env"
            env_file.touch()

            subdir = tmpdir_path / "subdir"
            subdir.mkdir()

            # Change to subdir and find root
            original_cwd = os.getcwd()
            try:
                os.chdir(subdir)
                root = _find_project_root()
                assert root == tmpdir_path
            finally:
                os.chdir(original_cwd)

    def test_find_project_root_with_pyproject_toml(self):
        """Test finding project root when pyproject.toml exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a directory structure with pyproject.toml
            tmpdir_path = pathlib.Path(tmpdir)
            pyproject = tmpdir_path / "pyproject.toml"
            pyproject.touch()

            subdir = tmpdir_path / "nested" / "dir"
            subdir.mkdir(parents=True)

            # Change to nested subdir and find root
            original_cwd = os.getcwd()
            try:
                os.chdir(subdir)
                root = _find_project_root()
                assert root == tmpdir_path
            finally:
                os.chdir(original_cwd)

    def test_find_project_root_no_indicators_returns_cwd(self):
        """Test that when no indicators are found, current working directory is returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a directory with no project indicators
            tmpdir_path = pathlib.Path(tmpdir)
            subdir = tmpdir_path / "subdir"
            subdir.mkdir()

            # Change to subdir - should return subdir itself as no indicators found
            original_cwd = os.getcwd()
            try:
                os.chdir(subdir)
                root = _find_project_root()
                assert root == subdir
            finally:
                os.chdir(original_cwd)

    def test_find_project_root_multiple_indicators(self):
        """Test finding project root when multiple indicators exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a directory structure with multiple indicators
            tmpdir_path = pathlib.Path(tmpdir)
            (tmpdir_path / ".env").touch()
            (tmpdir_path / ".env.local").touch()

            subdir = tmpdir_path / "src" / "module"
            subdir.mkdir(parents=True)

            # Change to nested subdir and find root
            original_cwd = os.getcwd()
            try:
                os.chdir(subdir)
                root = _find_project_root()
                assert root == tmpdir_path
            finally:
                os.chdir(original_cwd)


class TestLoadEnvironmentFiles:
    """Tests for _load_environment_files function."""

    @mock.patch("pyrit.setup.initialization._find_project_root")
    @mock.patch("dotenv.load_dotenv")
    def test_load_env_file_exists(self, mock_load_dotenv, mock_find_root):
        """Test loading .env file when it exists in project root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            mock_find_root.return_value = tmpdir_path

            # Create .env file
            env_file = tmpdir_path / ".env"
            env_file.write_text("TEST_VAR=value")

            from pyrit.setup.initialization import _load_environment_files

            _load_environment_files()

            # Verify load_dotenv was called with the correct path
            calls = mock_load_dotenv.call_args_list
            assert any(str(env_file) in str(call) for call in calls)

    @mock.patch("pyrit.setup.initialization._find_project_root")
    @mock.patch("dotenv.load_dotenv")
    def test_load_env_local_file_exists(self, mock_load_dotenv, mock_find_root):
        """Test loading .env.local file when it exists in project root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            mock_find_root.return_value = tmpdir_path

            # Create both .env and .env.local files
            env_file = tmpdir_path / ".env"
            env_file.write_text("TEST_VAR=value")
            env_local_file = tmpdir_path / ".env.local"
            env_local_file.write_text("TEST_VAR=local_value")

            from pyrit.setup.initialization import _load_environment_files

            _load_environment_files()

            # Verify both files were loaded
            calls = mock_load_dotenv.call_args_list
            assert any(str(env_file) in str(call) for call in calls)
            assert any(str(env_local_file) in str(call) for call in calls)

    @mock.patch("pyrit.setup.initialization._find_project_root")
    @mock.patch("dotenv.load_dotenv")
    def test_load_env_files_do_not_exist(self, mock_load_dotenv, mock_find_root):
        """Test behavior when no .env files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            mock_find_root.return_value = tmpdir_path

            from pyrit.setup.initialization import _load_environment_files

            _load_environment_files()

            # Should still call load_dotenv (with verbose=True fallback)
            assert mock_load_dotenv.called
