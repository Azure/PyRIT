# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
import tempfile
from typing import Optional, Sequence, Union, cast
from unittest.mock import MagicMock, patch

import pytest

from pyrit.setup import (
    IN_MEMORY,
    get_global_default_values,
    initialize_pyrit,
    set_default_value,
    set_global_variable,
)
from pyrit.setup.initialization import _execute_initialization_scripts


class TestExecuteInitializationScripts:
    """Tests for the _execute_initialization_scripts function."""

    def test_execute_single_script(self) -> None:
        """Test executing a single initialization script with explicit set_global_variable."""
        import sys

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                "from pyrit.setup import set_global_variable\n"
                "test_variable_single = 'initialized'\n"
                "set_global_variable(name='test_variable_single', value=test_variable_single)\n"
            )
            script_path = f.name

        try:
            # Execute the script
            _execute_initialization_scripts(script_paths=[script_path])

            # Verify the variable is accessible in __main__
            assert hasattr(sys.modules["__main__"], "test_variable_single")
            assert sys.modules["__main__"].test_variable_single == "initialized"  # type: ignore[attr-defined]
        finally:
            # Cleanup
            if hasattr(sys.modules["__main__"], "test_variable_single"):
                delattr(sys.modules["__main__"], "test_variable_single")
            pathlib.Path(script_path).unlink()

    def test_execute_multiple_scripts_in_order(self) -> None:
        """Test that multiple scripts are executed in the order provided."""
        import sys

        script_paths = []
        try:
            # Create first script
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(
                    "from pyrit.setup import set_global_variable\n"
                    "test_list_multi = [1]\n"
                    "set_global_variable(name='test_list_multi', value=test_list_multi)\n"
                )
                script_paths.append(f.name)

            # Create second script that depends on first
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(
                    "from pyrit.setup import set_global_variable\n"
                    "test_list_multi.append(2)\n"
                    "set_global_variable(name='test_list_multi', value=test_list_multi)\n"
                )
                script_paths.append(f.name)

            # Execute both scripts
            _execute_initialization_scripts(script_paths=cast(Sequence[Union[str, pathlib.Path]], script_paths))

            # Verify both scripts executed in order
            assert hasattr(sys.modules["__main__"], "test_list_multi")
            assert sys.modules["__main__"].test_list_multi == [1, 2]  # type: ignore[attr-defined]
        finally:
            # Cleanup
            if hasattr(sys.modules["__main__"], "test_list_multi"):
                delattr(sys.modules["__main__"], "test_list_multi")
            for script_path in script_paths:
                pathlib.Path(script_path).unlink()

    def test_execute_script_with_default_values(self) -> None:
        """Test executing a script that sets default values."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                "from pyrit.setup import set_default_value\n"
                "class TestClass:\n"
                "    pass\n"
                "set_default_value(class_type=TestClass, parameter_name='test_param', value='test_value')\n"
            )
            script_path = f.name

        try:
            _execute_initialization_scripts(script_paths=[script_path])
        finally:
            pathlib.Path(script_path).unlink()

    def test_nonexistent_script_raises_error(self) -> None:
        """Test that attempting to execute a nonexistent script raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Initialization script not found"):
            _execute_initialization_scripts(script_paths=["nonexistent_script.py"])

    def test_non_python_file_raises_error(self) -> None:
        """Test that attempting to execute a non-Python file raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("not a python file")
            script_path = f.name

        try:
            with pytest.raises(ValueError, match="must be a Python file"):
                _execute_initialization_scripts(script_paths=[script_path])
        finally:
            pathlib.Path(script_path).unlink()

    def test_script_with_syntax_error_raises_exception(self) -> None:
        """Test that a script with syntax errors raises an exception."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("this is not valid python syntax !!!")
            script_path = f.name

        try:
            with pytest.raises(SyntaxError):
                _execute_initialization_scripts(script_paths=[script_path])
        finally:
            pathlib.Path(script_path).unlink()

    def test_script_with_runtime_error_raises_exception(self) -> None:
        """Test that a script with runtime errors raises an exception."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("raise ValueError('Script error')\n")
            script_path = f.name

        try:
            with pytest.raises(ValueError, match="Script error"):
                _execute_initialization_scripts(script_paths=[script_path])
        finally:
            pathlib.Path(script_path).unlink()

    def test_accepts_pathlib_path(self) -> None:
        """Test that script paths can be provided as pathlib.Path objects and execute successfully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                "# This test just verifies that pathlib.Path objects work\n"
                "# We don't need to create global variables for this test\n"
                "temp_var = 'from_pathlib'\n"
            )
            script_path = pathlib.Path(f.name)

        try:
            # This should execute without errors
            _execute_initialization_scripts(script_paths=[script_path])
        finally:
            script_path.unlink()

    def test_accepts_string_path(self) -> None:
        """Test that script paths can be provided as strings and execute successfully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                "# This test just verifies that string paths work\n"
                "# We don't need to create global variables for this test\n"
                "temp_var = 'from_string'\n"
            )
            script_path = f.name

        try:
            # This should execute without errors
            _execute_initialization_scripts(script_paths=[script_path])
        finally:
            pathlib.Path(script_path).unlink()

    def test_variables_not_automatically_exposed(self) -> None:
        """Test that no variables are automatically exposed without explicit set_global_variable calls."""
        import sys

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                "_private_var = 'should_not_be_exposed'\n"
                "public_var = 'should_not_be_exposed'\n"
                "_helper_function = lambda x: x * 2\n"
            )
            script_path = f.name

        try:
            _execute_initialization_scripts(script_paths=[script_path])

            # No variables should be automatically accessible
            assert not hasattr(sys.modules["__main__"], "public_var")
            assert not hasattr(sys.modules["__main__"], "_private_var")
            assert not hasattr(sys.modules["__main__"], "_helper_function")
        finally:
            # Cleanup - these shouldn't exist but just in case
            if hasattr(sys.modules["__main__"], "public_var"):
                delattr(sys.modules["__main__"], "public_var")
            if hasattr(sys.modules["__main__"], "_private_var"):
                delattr(sys.modules["__main__"], "_private_var")
            if hasattr(sys.modules["__main__"], "_helper_function"):
                delattr(sys.modules["__main__"], "_helper_function")
            pathlib.Path(script_path).unlink()

    def test_new_explicit_behavior_comprehensive(self) -> None:
        """Test the comprehensive new behavior: explicit global variable setting only."""
        import sys

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                "from pyrit.setup import set_global_variable\n"
                "# Helper variables (not exposed)\n"
                "_private_helper = 'helper_value'\n"
                "_config_data = {'key': 'value'}\n"
                "\n"
                "# Variables that would have been auto-exposed before (not exposed now)\n"
                "auto_var = 'would_have_been_global'\n"
                "public_config = {'setting': 'value'}\n"
                "\n"
                "# Explicitly set only some variables as global\n"
                "set_global_variable(name='explicit_global', value='this_is_global')\n"
                "set_global_variable(name='computed_global', value=_private_helper.upper())\n"
                "# Note: auto_var and public_config are NOT set as global\n"
            )
            script_path = f.name

        try:
            _execute_initialization_scripts(script_paths=[script_path])

            # Only explicitly set variables should be accessible
            assert hasattr(sys.modules["__main__"], "explicit_global")
            assert sys.modules["__main__"].explicit_global == "this_is_global"  # type: ignore[attr-defined]
            
            assert hasattr(sys.modules["__main__"], "computed_global")
            assert sys.modules["__main__"].computed_global == "HELPER_VALUE"  # type: ignore[attr-defined]

            # Variables that would have been auto-exposed in the old system should NOT be accessible
            assert not hasattr(sys.modules["__main__"], "auto_var")
            assert not hasattr(sys.modules["__main__"], "public_config")
            assert not hasattr(sys.modules["__main__"], "_private_helper")
            assert not hasattr(sys.modules["__main__"], "_config_data")
        finally:
            # Cleanup
            for var_name in ["explicit_global", "computed_global", "auto_var", "public_config", "_private_helper", "_config_data"]:
                if hasattr(sys.modules["__main__"], var_name):
                    delattr(sys.modules["__main__"], var_name)
            pathlib.Path(script_path).unlink()

    def test_explicit_global_variables_work(self) -> None:
        """Test that explicit set_global_variable calls work as expected."""
        import sys

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                "from pyrit.setup import set_global_variable\n"
                "__custom_dunder__ = 'should_not_be_exposed'\n"
                "normal_var = 'should_be_exposed'\n"
                "set_global_variable(name='normal_var', value=normal_var)\n"
                "# Note: not setting __custom_dunder__ as global\n"
            )
            script_path = f.name

        try:
            _execute_initialization_scripts(script_paths=[script_path])

            # Normal variable should be accessible via explicit set_global_variable
            assert hasattr(sys.modules["__main__"], "normal_var")
            assert sys.modules["__main__"].normal_var == "should_be_exposed"  # type: ignore[attr-defined]

            # Dunder variable should NOT be accessible since it wasn't explicitly set
            assert not hasattr(sys.modules["__main__"], "__custom_dunder__")
        finally:
            if hasattr(sys.modules["__main__"], "normal_var"):
                delattr(sys.modules["__main__"], "normal_var")
            if hasattr(sys.modules["__main__"], "__custom_dunder__"):
                delattr(sys.modules["__main__"], "__custom_dunder__")
            pathlib.Path(script_path).unlink()

    def test_helper_variables_in_script_with_explicit_globals(self) -> None:
        """Test realistic scenario with helper variables used to compute explicitly set global variables."""
        import sys

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                "from pyrit.setup import set_global_variable\n"
                "# Helper variables for configuration\n"
                "_base_url = 'https://api.example.com'\n"
                "_api_version = 'v2'\n"
                "_endpoints = {'users': '/users', 'posts': '/posts'}\n"
                "\n"
                "# Public configuration variable\n"
                "api_config = {\n"
                "    'base_url': _base_url,\n"
                "    'version': _api_version,\n"
                "    'endpoints': _endpoints\n"
                "}\n"
                "\n"
                "# Helper function\n"
                "_calculate_timeout = lambda retries: retries * 2\n"
                "\n"
                "# Public variable using helper\n"
                "max_timeout = _calculate_timeout(5)\n"
                "\n"
                "# Explicitly set global variables\n"
                "set_global_variable(name='api_config', value=api_config)\n"
                "set_global_variable(name='max_timeout', value=max_timeout)\n"
            )
            script_path = f.name

        try:
            _execute_initialization_scripts(script_paths=[script_path])

            # Explicitly set global variables should be accessible and correctly computed
            assert hasattr(sys.modules["__main__"], "api_config")
            assert sys.modules["__main__"].api_config == {  # type: ignore[attr-defined]
                "base_url": "https://api.example.com",
                "version": "v2",
                "endpoints": {"users": "/users", "posts": "/posts"},
            }

            assert hasattr(sys.modules["__main__"], "max_timeout")
            assert sys.modules["__main__"].max_timeout == 10  # type: ignore[attr-defined]

            # Helper variables should NOT be accessible (no explicit set_global_variable)
            assert not hasattr(sys.modules["__main__"], "_base_url")
            assert not hasattr(sys.modules["__main__"], "_api_version")
            assert not hasattr(sys.modules["__main__"], "_endpoints")
            assert not hasattr(sys.modules["__main__"], "_calculate_timeout")
        finally:
            if hasattr(sys.modules["__main__"], "api_config"):
                delattr(sys.modules["__main__"], "api_config")
            if hasattr(sys.modules["__main__"], "max_timeout"):
                delattr(sys.modules["__main__"], "max_timeout")
            pathlib.Path(script_path).unlink()

    def test_multiple_scripts_with_explicit_globals(self) -> None:
        """Test that explicit global variables from multiple scripts work correctly."""
        import sys

        script_paths = []
        try:
            # First script with private helper
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(
                    "from pyrit.setup import set_global_variable\n"
                    "_helper1 = 'first'\n"
                    "result1 = _helper1.upper()\n"
                    "set_global_variable(name='result1', value=result1)\n"
                )
                script_paths.append(f.name)

            # Second script with private helper
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(
                    "from pyrit.setup import set_global_variable\n"
                    "_helper2 = 'second'\n"
                    "result2 = _helper2.upper()\n"
                    "set_global_variable(name='result2', value=result2)\n"
                )
                script_paths.append(f.name)

            _execute_initialization_scripts(script_paths=cast(Sequence[Union[str, pathlib.Path]], script_paths))

            # Explicitly set global results should be accessible
            assert hasattr(sys.modules["__main__"], "result1")
            assert sys.modules["__main__"].result1 == "FIRST"  # type: ignore[attr-defined]

            assert hasattr(sys.modules["__main__"], "result2")
            assert sys.modules["__main__"].result2 == "SECOND"  # type: ignore[attr-defined]

            # Private helpers should NOT be accessible (no explicit set_global_variable)
            assert not hasattr(sys.modules["__main__"], "_helper1")
            assert not hasattr(sys.modules["__main__"], "_helper2")
        finally:
            if hasattr(sys.modules["__main__"], "result1"):
                delattr(sys.modules["__main__"], "result1")
            if hasattr(sys.modules["__main__"], "result2"):
                delattr(sys.modules["__main__"], "result2")
            for script_path in script_paths:
                pathlib.Path(script_path).unlink()


class TestInitializePyritWithScripts:
    """Tests for initialize_pyrit function with initialization_scripts parameter."""

    @patch("pyrit.setup.initialization.CentralMemory")
    def test_initialize_pyrit_without_scripts(self, mock_central_memory: MagicMock) -> None:
        """Test that initialize_pyrit works without initialization scripts."""
        initialize_pyrit(memory_db_type=IN_MEMORY)
        mock_central_memory.set_memory_instance.assert_called_once()

    @patch("pyrit.setup.initialization.CentralMemory")
    def test_initialize_pyrit_with_valid_script(self, mock_central_memory: MagicMock) -> None:
        """Test that initialize_pyrit executes provided initialization scripts."""
        import sys

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                "from pyrit.setup import set_global_variable\n"
                "test_variable_init = 'initialized'\n"
                "set_global_variable(name='test_variable_init', value=test_variable_init)\n"
            )
            script_path = f.name

        try:
            initialize_pyrit(memory_db_type=IN_MEMORY, initialization_scripts=[script_path])
            mock_central_memory.set_memory_instance.assert_called_once()

            # Verify variable is accessible via explicit set_global_variable
            assert hasattr(sys.modules["__main__"], "test_variable_init")
            assert sys.modules["__main__"].test_variable_init == "initialized"  # type: ignore[attr-defined]
        finally:
            if hasattr(sys.modules["__main__"], "test_variable_init"):
                delattr(sys.modules["__main__"], "test_variable_init")
            pathlib.Path(script_path).unlink()

    @patch("pyrit.setup.initialization.CentralMemory")
    def test_initialize_pyrit_variables_accessible_after_call(self, mock_central_memory: MagicMock) -> None:
        """Test the exact use case: variables from scripts are accessible after initialize_pyrit with explicit set_global_variable."""
        import sys

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                'from pyrit.setup import set_global_variable\n'
                'myVar = "t"\n'
                'set_global_variable(name="myVar", value=myVar)\n'
            )
            script_path = f.name

        try:
            # This is the updated use case with explicit global variable setting
            initialize_pyrit(memory_db_type=IN_MEMORY, initialization_scripts=[script_path])

            # The variable should be accessible in __main__ via explicit set_global_variable
            assert hasattr(sys.modules["__main__"], "myVar")
            assert sys.modules["__main__"].myVar == "t"  # type: ignore[attr-defined]
        finally:
            if hasattr(sys.modules["__main__"], "myVar"):
                delattr(sys.modules["__main__"], "myVar")
            pathlib.Path(script_path).unlink()

    @patch("pyrit.setup.initialization.CentralMemory")
    def test_initialize_pyrit_with_multiple_scripts(self, mock_central_memory: MagicMock) -> None:
        """Test that initialize_pyrit executes multiple initialization scripts in order."""
        import sys

        script_paths = []
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(
                    "from pyrit.setup import set_global_variable\n"
                    "script_order_init = [1]\n"
                    "set_global_variable(name='script_order_init', value=script_order_init)\n"
                )
                script_paths.append(f.name)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(
                    "from pyrit.setup import set_global_variable\n"
                    "script_order_init.append(2)\n"
                    "set_global_variable(name='script_order_init', value=script_order_init)\n"
                )
                script_paths.append(f.name)

            initialize_pyrit(
                memory_db_type=IN_MEMORY, initialization_scripts=cast(Sequence[Union[str, pathlib.Path]], script_paths)
            )
            mock_central_memory.set_memory_instance.assert_called_once()

            # Verify execution order
            assert hasattr(sys.modules["__main__"], "script_order_init")
            assert sys.modules["__main__"].script_order_init == [1, 2]  # type: ignore[attr-defined]
        finally:
            if hasattr(sys.modules["__main__"], "script_order_init"):
                delattr(sys.modules["__main__"], "script_order_init")
            for script_path in script_paths:
                pathlib.Path(script_path).unlink()

    @patch("pyrit.setup.initialization.CentralMemory")
    def test_initialize_pyrit_user_case_separate_file(self, mock_central_memory: MagicMock) -> None:
        """
        Test the updated user case: initialize_pyrit with a script in a separate file,
        then access variables from that script using explicit set_global_variable.

        Updated user case:
            # In myscript.py:
            from pyrit.setup import set_global_variable
            myVar = "t"
            set_global_variable(name="myVar", value=myVar)

            # In main file:
            initialize_pyrit(memory_db_type="InMemory", initialization_scripts=['myscript.py'])
            assert myVar == "t"
        """
        import sys

        # Create the myscript.py file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=".") as f:
            f.write(
                'from pyrit.setup import set_global_variable\n'
                'myVar = "t"\n'
                'set_global_variable(name="myVar", value=myVar)\n'
            )
            script_name = pathlib.Path(f.name).name

        try:
            # User's updated use case
            initialize_pyrit(memory_db_type="InMemory", initialization_scripts=[script_name])

            # This is what the user wants to do - access myVar after initialization
            assert hasattr(sys.modules["__main__"], "myVar")
            assert sys.modules["__main__"].myVar == "t"  # type: ignore[attr-defined]

            mock_central_memory.set_memory_instance.assert_called_once()
        finally:
            if hasattr(sys.modules["__main__"], "myVar"):
                delattr(sys.modules["__main__"], "myVar")
            pathlib.Path(script_name).unlink(missing_ok=True)

    @patch("pyrit.setup.initialization.CentralMemory")
    def test_initialize_pyrit_with_invalid_script_raises_error(self, mock_central_memory: MagicMock) -> None:
        """Test that initialize_pyrit raises an error for invalid scripts."""
        with pytest.raises(FileNotFoundError):
            initialize_pyrit(memory_db_type=IN_MEMORY, initialization_scripts=["nonexistent.py"])

    @patch("pyrit.setup.initialization.SQLiteMemory")
    @patch("pyrit.setup.initialization.CentralMemory")
    def test_initialize_pyrit_memory_setup_before_scripts(
        self, mock_central_memory: MagicMock, mock_sqlite_memory: MagicMock
    ) -> None:
        """Test that memory setup occurs before initialization scripts are executed.

        This is critical because initialization scripts may instantiate objects
        (like prompt targets) that require central memory to be initialized.
        """
        import sys

        # Track the order of operations
        execution_order = []

        # Mock SQLiteMemory to track when it's created
        def track_memory_creation(*args, **kwargs):  # type: ignore[no-untyped-def]
            execution_order.append("memory_created")
            return MagicMock()

        mock_sqlite_memory.side_effect = track_memory_creation

        # Mock set_memory_instance to track when it's called
        def track_memory_set(*args, **kwargs):  # type: ignore[no-untyped-def]
            execution_order.append("memory_set")

        mock_central_memory.set_memory_instance.side_effect = track_memory_set

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Script appends to the execution order
            f.write(
                "import sys\n"
                "# Access the execution_order from the test's local scope via test module\n"
                "test_module = sys.modules[__name__]\n"
                "if hasattr(test_module, 'execution_order_tracker'):\n"
                "    test_module.execution_order_tracker.append('script_executed')\n"
            )
            script_path = f.name

        try:
            # Make execution_order accessible to the script
            sys.modules["__main__"].execution_order_tracker = execution_order  # type: ignore[attr-defined]

            initialize_pyrit(memory_db_type=IN_MEMORY, initialization_scripts=[script_path])

            # Memory should be created and set before scripts execute
            assert execution_order == ["memory_created", "memory_set", "script_executed"]
        finally:
            if hasattr(sys.modules["__main__"], "execution_order_tracker"):
                delattr(sys.modules["__main__"], "execution_order_tracker")
            pathlib.Path(script_path).unlink()

    @patch("pyrit.setup.initialization.CentralMemory")
    def test_initialize_pyrit_with_empty_script_list(self, mock_central_memory: MagicMock) -> None:
        """Test that initialize_pyrit handles empty script list gracefully."""
        initialize_pyrit(memory_db_type=IN_MEMORY, initialization_scripts=[])
        mock_central_memory.set_memory_instance.assert_called_once()

    @patch("pyrit.setup.initialization.CentralMemory")
    def test_initialize_pyrit_explicit_global_variables_only(self, mock_central_memory: MagicMock) -> None:
        """Test that initialize_pyrit only exposes variables via explicit set_global_variable calls."""
        import sys

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                "from pyrit.setup import set_global_variable\n"
                "_private_helper = 'internal'\n"
                "public_config = 'external'\n"
                "set_global_variable(name='public_config', value=public_config)\n"
                "# Note: _private_helper is not explicitly set as global\n"
            )
            script_path = f.name

        try:
            initialize_pyrit(memory_db_type=IN_MEMORY, initialization_scripts=[script_path])
            mock_central_memory.set_memory_instance.assert_called_once()

            # Explicitly set variable should be accessible
            assert hasattr(sys.modules["__main__"], "public_config")
            assert sys.modules["__main__"].public_config == "external"  # type: ignore[attr-defined]

            # Private variable should NOT be accessible (not explicitly set as global)
            assert not hasattr(sys.modules["__main__"], "_private_helper")
        finally:
            if hasattr(sys.modules["__main__"], "public_config"):
                delattr(sys.modules["__main__"], "public_config")
            if hasattr(sys.modules["__main__"], "_private_helper"):
                delattr(sys.modules["__main__"], "_private_helper")
            pathlib.Path(script_path).unlink()

    @patch("pyrit.setup.initialization.CentralMemory")
    def test_initialize_pyrit_realistic_config_with_helpers(self, mock_central_memory: MagicMock) -> None:
        """Test realistic initialization script with helper variables and default value configuration."""
        import sys

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                "# Helper configuration values\n"
                "_default_temp = 0.7\n"
                "_default_max_tokens = 1000\n"
                "_model_name = 'gpt-4'\n"
                "\n"
                "# Public configuration\n"
                "model_config = {\n"
                "    'temperature': _default_temp,\n"
                "    'max_tokens': _default_max_tokens,\n"
                "    'model': _model_name\n"
                "}\n"
                "\n"
                "# Explicitly set global variables\n"
                "from pyrit.setup import set_default_value, set_global_variable\n"
                "set_global_variable(name='model_config', value=model_config)\n"
                "\n"
                "# Set default values using PyRIT API with a lightweight test class\n"
                "class TestTargetClass:\n"
                "    pass\n"
                "set_default_value(class_type=TestTargetClass, parameter_name='temperature', value=_default_temp)\n"
            )
            script_path = f.name

        try:
            initialize_pyrit(memory_db_type=IN_MEMORY, initialization_scripts=[script_path])
            mock_central_memory.set_memory_instance.assert_called_once()

            # Explicitly set global config should be accessible
            assert hasattr(sys.modules["__main__"], "model_config")
            assert sys.modules["__main__"].model_config == {  # type: ignore[attr-defined]
                "temperature": 0.7,
                "max_tokens": 1000,
                "model": "gpt-4",
            }

            # Helper variables should NOT be accessible (not explicitly set as global)
            assert not hasattr(sys.modules["__main__"], "_default_temp")
            assert not hasattr(sys.modules["__main__"], "_default_max_tokens")
            assert not hasattr(sys.modules["__main__"], "_model_name")

            # Default values should still be set correctly
            from pyrit.setup import get_global_default_values

            defaults = get_global_default_values()._default_values

            # Verify the default was set (using the class name defined in the script)
            assert any(
                "TestTargetClass" in str(scope.class_type) and scope.parameter_name == "temperature"
                for scope in defaults.keys()
            )
        finally:
            if hasattr(sys.modules["__main__"], "model_config"):
                delattr(sys.modules["__main__"], "model_config")
            from pyrit.setup import get_global_default_values

            get_global_default_values()._default_values.clear()
            pathlib.Path(script_path).unlink()


class TestResetDefaultValuesInInitialization:
    """Tests for reset_default_values behavior during initialization."""

    @patch("pyrit.setup.initialization.CentralMemory")
    def test_initialize_pyrit_resets_defaults_before_scripts(self, mock_central_memory: MagicMock) -> None:
        """Test that initialize_pyrit resets default values before executing initialization scripts."""

        class TestClass:
            def __init__(self, *, param: Optional[str] = None) -> None:
                self.param = param

        try:
            # Set some defaults before initialization
            set_default_value(class_type=TestClass, parameter_name="param", value="old_default")
            assert len(get_global_default_values()._default_values) > 0

            # Create a script that sets new defaults
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(
                    "from pyrit.setup import set_default_value\n"
                    "class NewTestClass:\n"
                    "    pass\n"
                    "set_default_value(class_type=NewTestClass, parameter_name='new_param', value='new_value')\n"
                )
                script_path = f.name

            # Initialize PyRIT with the script
            initialize_pyrit(memory_db_type=IN_MEMORY, initialization_scripts=[script_path])

            # Verify old defaults were cleared
            # The global defaults should only contain what was set by the script
            defaults = get_global_default_values()._default_values
            # Old default should be gone
            old_scope_exists = any(
                scope.class_type == TestClass and scope.parameter_name == "param" for scope in defaults.keys()
            )
            assert not old_scope_exists, "Old defaults should be cleared before script execution"

            pathlib.Path(script_path).unlink()
        finally:
            get_global_default_values()._default_values.clear()

    @patch("pyrit.setup.initialization.CentralMemory")
    def test_initialize_pyrit_multiple_calls_reset_each_time(self, mock_central_memory: MagicMock) -> None:
        """Test that multiple calls to initialize_pyrit reset defaults each time."""

        class TestClass1:
            pass

        class TestClass2:
            pass

        try:
            # First initialization with a script
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(
                    "from pyrit.setup import set_default_value\n"
                    "class FirstClass:\n"
                    "    pass\n"
                    "set_default_value(class_type=FirstClass, parameter_name='first_param', value='first_value')\n"
                )
                first_script = f.name

            initialize_pyrit(memory_db_type=IN_MEMORY, initialization_scripts=[first_script])

            # Verify first script's defaults are set
            first_defaults_count = len(get_global_default_values()._default_values)
            assert first_defaults_count > 0

            # Second initialization with a different script
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(
                    "from pyrit.setup import set_default_value\n"
                    "class SecondClass:\n"
                    "    pass\n"
                    "set_default_value(class_type=SecondClass, parameter_name='second_param', value='second_value')\n"
                )
                second_script = f.name

            initialize_pyrit(memory_db_type=IN_MEMORY, initialization_scripts=[second_script])

            # Verify that we only have defaults from the second script
            # (first script's defaults should have been cleared)
            second_defaults_count = len(get_global_default_values()._default_values)
            assert second_defaults_count > 0

            # Check that first script's defaults are not present
            defaults = get_global_default_values()._default_values
            first_class_exists = any("FirstClass" in str(scope.class_type) for scope in defaults.keys())
            assert not first_class_exists, "First script's defaults should be cleared on second initialization"

            pathlib.Path(first_script).unlink()
            pathlib.Path(second_script).unlink()
        finally:
            get_global_default_values()._default_values.clear()

    @patch("pyrit.setup.initialization.CentralMemory")
    def test_initialize_pyrit_without_scripts_still_resets_defaults(self, mock_central_memory: MagicMock) -> None:
        """Test that initialize_pyrit resets defaults even when no scripts are provided."""

        class TestClass:
            pass

        try:
            # Set some defaults manually
            set_default_value(class_type=TestClass, parameter_name="param", value="manual_value")
            assert len(get_global_default_values()._default_values) > 0

            # Initialize without scripts
            initialize_pyrit(memory_db_type=IN_MEMORY, initialization_scripts=None)

            # Verify defaults were cleared
            assert len(get_global_default_values()._default_values) == 0
        finally:
            get_global_default_values()._default_values.clear()

    @patch("pyrit.setup.initialization.CentralMemory")
    def test_initialize_pyrit_reset_happens_before_script_execution(self, mock_central_memory: MagicMock) -> None:
        """Test that reset happens before scripts run, allowing scripts to set fresh defaults."""
        import sys

        try:
            # Set a default before initialization
            class OldClass:
                pass

            set_default_value(class_type=OldClass, parameter_name="old_param", value="old_value")

            # Create a script that checks if old defaults exist and sets new ones
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(
                    "from pyrit.setup import get_global_default_values, set_default_value, set_global_variable\n"
                    "# Store the count of defaults at script execution time\n"
                    "defaults_count_at_script_time = len(get_global_default_values()._default_values)\n"
                    "set_global_variable(name='defaults_count_at_script_time', value=defaults_count_at_script_time)\n"
                    "class ScriptClass:\n"
                    "    pass\n"
                    "set_default_value(class_type=ScriptClass, parameter_name='script_param', value='script_value')\n"
                )
                script_path = f.name

            initialize_pyrit(memory_db_type=IN_MEMORY, initialization_scripts=[script_path])

            # The script should have seen 0 defaults (because reset happened first)
            assert hasattr(sys.modules["__main__"], "defaults_count_at_script_time")
            assert sys.modules["__main__"].defaults_count_at_script_time == 0  # type: ignore[attr-defined]

            # And now we should only have the script's defaults
            defaults = get_global_default_values()._default_values
            assert len(defaults) == 1
            assert any("ScriptClass" in str(scope.class_type) for scope in defaults.keys())

            pathlib.Path(script_path).unlink()
            if hasattr(sys.modules["__main__"], "defaults_count_at_script_time"):
                delattr(sys.modules["__main__"], "defaults_count_at_script_time")
        finally:
            get_global_default_values()._default_values.clear()
