# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pathlib
import sys
import tempfile
from typing import List
from unittest import mock

import pytest
from mock_alchemy.mocking import UnifiedAlchemyMagicMock

from pyrit.setup import AZURE_SQL, IN_MEMORY, SQLITE, initialize_pyrit, reset_default_values
from pyrit.setup.initialization import (
    _execute_initializers,
    _load_environment_files,
    _load_initializers_from_scripts,
)
from pyrit.setup.initializers.base import PyRITInitializer


class TestLoadEnvironmentFiles:
    """Tests for _load_environment_files function."""

    @mock.patch("dotenv.load_dotenv")
    @mock.patch("pathlib.Path.exists")
    @mock.patch("logging.getLogger")
    def test_load_both_env_files(self, mock_logger, mock_exists, mock_load_dotenv):
        """Test loading both .env and .env.local files."""
        mock_exists.side_effect = [True, True]
        mock_logger.return_value = mock.Mock()

        _load_environment_files()

        assert mock_load_dotenv.call_count == 2

    @mock.patch("dotenv.load_dotenv")
    @mock.patch("pathlib.Path.exists")
    @mock.patch("logging.getLogger")
    def test_load_base_only(self, mock_logger, mock_exists, mock_load_dotenv):
        """Test loading only .env file when .env.local doesn't exist."""
        mock_exists.side_effect = [True, False]
        mock_logger.return_value = mock.Mock()

        _load_environment_files()

        assert mock_load_dotenv.call_count == 2

    @mock.patch("dotenv.load_dotenv")
    @mock.patch("pathlib.Path.exists")
    @mock.patch("logging.getLogger")
    def test_load_neither_file_exists(self, mock_logger, mock_exists, mock_load_dotenv):
        """Test when neither .env nor .env.local exists."""
        mock_exists.side_effect = [False, False]
        mock_logger.return_value = mock.Mock()

        _load_environment_files()

        # Should still call load_dotenv for verbose mode
        assert mock_load_dotenv.call_count == 2


class TestLoadInitializersFromScripts:
    """Tests for _load_initializers_from_scripts function."""

    def test_load_single_initializer_from_script(self):
        """Test loading a single initializer from a Python script."""
        # Create a temporary script with a simple initializer
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
from pyrit.setup.initializers.base import PyRITInitializer

class TestInitializer(PyRITInitializer):
    @property
    def name(self) -> str:
        return "Test Initializer"
    
    @property
    def description(self) -> str:
        return "Test description"
    
    def initialize(self) -> None:
        pass
""")
            script_path = f.name

        try:
            initializers = _load_initializers_from_scripts(script_paths=[script_path])
            assert len(initializers) == 1
            assert initializers[0].name == "Test Initializer"
        finally:
            os.unlink(script_path)

    def test_load_multiple_initializers_from_script(self):
        """Test loading multiple initializers from a single script."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
from pyrit.setup.initializers.base import PyRITInitializer

class TestInitializer1(PyRITInitializer):
    @property
    def name(self) -> str:
        return "Test 1"
    
    @property
    def description(self) -> str:
        return "First test"
    
    def initialize(self) -> None:
        pass

class TestInitializer2(PyRITInitializer):
    @property
    def name(self) -> str:
        return "Test 2"
    
    @property
    def description(self) -> str:
        return "Second test"
    
    def initialize(self) -> None:
        pass
""")
            script_path = f.name

        try:
            initializers = _load_initializers_from_scripts(script_paths=[script_path])
            assert len(initializers) == 2
            names = [init.name for init in initializers]
            assert "Test 1" in names
            assert "Test 2" in names
        finally:
            os.unlink(script_path)

    def test_load_from_multiple_scripts(self):
        """Test loading initializers from multiple script files."""
        script_paths = []
        try:
            # Create first script
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write("""
from pyrit.setup.initializers.base import PyRITInitializer

class ScriptOneInitializer(PyRITInitializer):
    @property
    def name(self) -> str:
        return "Script One"
    
    @property
    def description(self) -> str:
        return "From script 1"
    
    def initialize(self) -> None:
        pass
""")
                script_paths.append(f.name)

            # Create second script
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write("""
from pyrit.setup.initializers.base import PyRITInitializer

class ScriptTwoInitializer(PyRITInitializer):
    @property
    def name(self) -> str:
        return "Script Two"
    
    @property
    def description(self) -> str:
        return "From script 2"
    
    def initialize(self) -> None:
        pass
""")
                script_paths.append(f.name)

            initializers = _load_initializers_from_scripts(script_paths=script_paths)
            assert len(initializers) == 2
            names = [init.name for init in initializers]
            assert "Script One" in names
            assert "Script Two" in names
        finally:
            for path in script_paths:
                if os.path.exists(path):
                    os.unlink(path)

    def test_script_not_found_raises_error(self):
        """Test that FileNotFoundError is raised for non-existent script."""
        with pytest.raises(FileNotFoundError):
            _load_initializers_from_scripts(script_paths=["nonexistent_script.py"])

    def test_non_python_file_raises_error(self):
        """Test that ValueError is raised for non-Python files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Not a Python file")
            txt_path = f.name

        try:
            with pytest.raises(ValueError, match="must be a Python file"):
                _load_initializers_from_scripts(script_paths=[txt_path])
        finally:
            os.unlink(txt_path)

    def test_script_without_initializers_raises_error(self):
        """Test that ValueError is raised when script has no PyRITInitializer classes."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
# A script with no initializers
def some_function():
    pass
""")
            script_path = f.name

        try:
            with pytest.raises(ValueError, match="must contain at least one PyRITInitializer"):
                _load_initializers_from_scripts(script_paths=[script_path])
        finally:
            os.unlink(script_path)

    def test_script_with_invalid_initializer(self):
        """Test that invalid initializer classes are skipped with warning."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
from pyrit.setup.initializers.base import PyRITInitializer

class BrokenInitializer(PyRITInitializer):
    def __init__(self):
        raise RuntimeError("Cannot instantiate")
    
    @property
    def name(self) -> str:
        return "Broken"
    
    @property
    def description(self) -> str:
        return "Broken"
    
    def initialize(self) -> None:
        pass

class GoodInitializer(PyRITInitializer):
    @property
    def name(self) -> str:
        return "Good"
    
    @property
    def description(self) -> str:
        return "Good initializer"
    
    def initialize(self) -> None:
        pass
""")
            script_path = f.name

        try:
            # Should load the good initializer and skip the broken one
            initializers = _load_initializers_from_scripts(script_paths=[script_path])
            assert len(initializers) == 1
            assert initializers[0].name == "Good"
        finally:
            os.unlink(script_path)


class TestExecuteInitializers:
    """Tests for _execute_initializers function."""

    def setup_method(self) -> None:
        """Clear default values before each test."""
        reset_default_values()

    def test_execute_single_initializer(self):
        """Test executing a single initializer."""

        class MockInitializer(PyRITInitializer):
            def __init__(self):
                super().__init__()
                self.executed = False

            @property
            def name(self) -> str:
                return "Mock"

            @property
            def description(self) -> str:
                return "Mock initializer"

            def initialize(self) -> None:
                self.executed = True

        init = MockInitializer()
        _execute_initializers(initializers=[init])

        assert init.executed

    def test_execute_multiple_initializers(self):
        """Test executing multiple initializers."""
        executed = []

        class MockInitializer1(PyRITInitializer):
            @property
            def name(self) -> str:
                return "Mock 1"

            @property
            def description(self) -> str:
                return "First mock"

            def initialize(self) -> None:
                executed.append(1)

        class MockInitializer2(PyRITInitializer):
            @property
            def name(self) -> str:
                return "Mock 2"

            @property
            def description(self) -> str:
                return "Second mock"

            def initialize(self) -> None:
                executed.append(2)

        _execute_initializers(initializers=[MockInitializer1(), MockInitializer2()])

        assert 1 in executed
        assert 2 in executed

    def test_execute_in_order(self):
        """Test that initializers execute in execution_order."""
        execution_order = []

        class LateInitializer(PyRITInitializer):
            @property
            def name(self) -> str:
                return "Late"

            @property
            def description(self) -> str:
                return "Runs later"

            @property
            def execution_order(self) -> int:
                return 10

            def initialize(self) -> None:
                execution_order.append("late")

        class EarlyInitializer(PyRITInitializer):
            @property
            def name(self) -> str:
                return "Early"

            @property
            def description(self) -> str:
                return "Runs early"

            @property
            def execution_order(self) -> int:
                return 1

            def initialize(self) -> None:
                execution_order.append("early")

        # Pass in reverse order to test sorting
        _execute_initializers(initializers=[LateInitializer(), EarlyInitializer()])

        assert execution_order == ["early", "late"]

    def test_non_initializer_raises_error(self):
        """Test that passing non-PyRITInitializer raises ValueError."""

        class NotAnInitializer:
            pass

        with pytest.raises(ValueError, match="must be PyRITInitializer instances"):
            _execute_initializers(initializers=[NotAnInitializer()])  # type: ignore

    def test_validation_error_propagates(self):
        """Test that validation errors are propagated."""

        class FailingInitializer(PyRITInitializer):
            @property
            def name(self) -> str:
                return "Failing"

            @property
            def description(self) -> str:
                return "Will fail validation"

            def validate(self) -> None:
                raise ValueError("Validation failed")

            def initialize(self) -> None:
                pass

        with pytest.raises(ValueError, match="Validation failed"):
            _execute_initializers(initializers=[FailingInitializer()])

    def test_initialization_error_propagates(self):
        """Test that initialization errors are propagated."""

        class ErrorInitializer(PyRITInitializer):
            @property
            def name(self) -> str:
                return "Error"

            @property
            def description(self) -> str:
                return "Will error during init"

            def initialize(self) -> None:
                raise RuntimeError("Initialization failed")

        with pytest.raises(RuntimeError, match="Initialization failed"):
            _execute_initializers(initializers=[ErrorInitializer()])


class TestInitializePyrit:
    """Tests for initialize_pyrit function."""

    def setup_method(self) -> None:
        """Clear default values before each test."""
        reset_default_values()

    @pytest.mark.parametrize(
        "memory_db_type,memory_kwargs",
        [
            (IN_MEMORY, {}),
            (SQLITE, {"verbose": True}),
            (
                AZURE_SQL,
                {
                    "connection_string": "mssql+pyodbc://test:test@test/test?driver=ODBC+Driver+18+for+SQL+Server",
                    "results_container_url": "https://test.blob.core.windows.net/test",
                    "results_sas_token": "valid_sas_token",
                },
            ),
        ],
    )
    @mock.patch("pyrit.memory.central_memory.CentralMemory.set_memory_instance")
    @mock.patch("pyrit.setup.initialization._load_environment_files")
    def test_initialize_with_different_memory_types(
        self, mock_load_env, mock_set_memory, memory_db_type, memory_kwargs
    ):
        """Test initialization with different memory database types."""
        with (
            mock.patch("pyrit.memory.AzureSQLMemory.get_session") as get_session_mock,
            mock.patch("pyrit.memory.AzureSQLMemory._create_auth_token") as create_auth_token_mock,
            mock.patch("pyrit.memory.AzureSQLMemory._enable_azure_authorization") as enable_azure_authorization_mock,
        ):
            # Mock AzureSQL dependencies
            session_mock = UnifiedAlchemyMagicMock()
            session_mock.__enter__.return_value = session_mock
            session_mock.is_modified.return_value = True
            get_session_mock.return_value = session_mock
            create_auth_token_mock.return_value = "token"
            enable_azure_authorization_mock.return_value = None

            initialize_pyrit(memory_db_type=memory_db_type, **memory_kwargs)

            mock_load_env.assert_called_once()
            mock_set_memory.assert_called_once()

    @mock.patch("pyrit.memory.central_memory.CentralMemory.set_memory_instance")
    @mock.patch("pyrit.setup.initialization._load_environment_files")
    def test_initialize_with_initializers(self, mock_load_env, mock_set_memory):
        """Test initialization with initializer instances."""
        executed = []

        class TestInit(PyRITInitializer):
            @property
            def name(self) -> str:
                return "Test"

            @property
            def description(self) -> str:
                return "Test init"

            def initialize(self) -> None:
                executed.append("test")

        initialize_pyrit(memory_db_type=IN_MEMORY, initializers=[TestInit()])

        assert "test" in executed
        mock_load_env.assert_called_once()
        mock_set_memory.assert_called_once()

    @mock.patch("pyrit.memory.central_memory.CentralMemory.set_memory_instance")
    @mock.patch("pyrit.setup.initialization._load_environment_files")
    def test_initialize_with_scripts(self, mock_load_env, mock_set_memory):
        """Test initialization with script paths."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
from pyrit.setup.initializers.base import PyRITInitializer
from pyrit.common.apply_defaults import set_global_variable

class ScriptInit(PyRITInitializer):
    @property
    def name(self) -> str:
        return "Script"
    
    @property
    def description(self) -> str:
        return "From script"
    
    def initialize(self) -> None:
        set_global_variable(name="script_executed", value=True)
""")
            script_path = f.name

        try:
            initialize_pyrit(memory_db_type=IN_MEMORY, initialization_scripts=[script_path])

            mock_load_env.assert_called_once()
            mock_set_memory.assert_called_once()
        finally:
            os.unlink(script_path)

    @mock.patch("pyrit.memory.central_memory.CentralMemory.set_memory_instance")
    @mock.patch("pyrit.setup.initialization._load_environment_files")
    def test_initialize_with_both_initializers_and_scripts(self, mock_load_env, mock_set_memory):
        """Test initialization with both initializers and scripts."""
        executed = []

        class DirectInit(PyRITInitializer):
            @property
            def name(self) -> str:
                return "Direct"

            @property
            def description(self) -> str:
                return "Direct initializer"

            def initialize(self) -> None:
                executed.append("direct")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
from pyrit.setup.initializers.base import PyRITInitializer
from pyrit.common.apply_defaults import set_global_variable

class ScriptInit(PyRITInitializer):
    @property
    def name(self) -> str:
        return "Script"
    
    @property
    def description(self) -> str:
        return "From script"
    
    def initialize(self) -> None:
        import sys
        # Access the test's executed list through global var
        if hasattr(sys.modules['__main__'], 'test_executed'):
            sys.modules['__main__'].test_executed.append("script")
""")
            script_path = f.name

        try:
            # Make executed list accessible to script
            sys.modules["__main__"].test_executed = executed  # type: ignore

            initialize_pyrit(memory_db_type=IN_MEMORY, initializers=[DirectInit()], initialization_scripts=[script_path])

            assert "direct" in executed
            mock_load_env.assert_called_once()
            mock_set_memory.assert_called_once()
        finally:
            os.unlink(script_path)
            if hasattr(sys.modules["__main__"], "test_executed"):
                delattr(sys.modules["__main__"], "test_executed")

    def test_invalid_memory_type_raises_error(self):
        """Test that invalid memory type raises ValueError."""
        with pytest.raises(ValueError, match="is not a supported type"):
            initialize_pyrit(memory_db_type="InvalidType")  # type: ignore

    @mock.patch("pyrit.memory.central_memory.CentralMemory.set_memory_instance")
    @mock.patch("pyrit.setup.initialization._load_environment_files")
    @mock.patch("logging.getLogger")
    def test_duckdb_deprecated_warning(self, mock_logger, mock_load_env, mock_set_memory):
        """Test that DuckDB shows deprecation warning and uses SQLite."""
        mock_log = mock.Mock()
        mock_logger.return_value = mock_log

        initialize_pyrit(memory_db_type="DuckDB")  # type: ignore

        # Should log warning about DuckDB deprecation
        mock_log.warning.assert_called()
        warning_msg = str(mock_log.warning.call_args)
        assert "DuckDB is no longer supported" in warning_msg

    @mock.patch("pyrit.memory.central_memory.CentralMemory.set_memory_instance")
    @mock.patch("pyrit.setup.initialization._load_environment_files")
    def test_reset_defaults_called_before_init(self, mock_load_env, mock_set_memory):
        """Test that default values are reset before initialization."""
        from pyrit.common.apply_defaults import set_default_value

        # Set a default value
        class DummyClass:
            def __init__(self, *, value: str = "default") -> None:
                self.value = value

        set_default_value(class_type=DummyClass, parameter_name="value", value="before_init")

        # Initialize PyRIT - this should reset defaults
        initialize_pyrit(memory_db_type=IN_MEMORY)

        # Verify defaults were cleared
        from pyrit.common.apply_defaults import get_global_default_values

        registry = get_global_default_values()
        assert len(registry._default_values) == 0
