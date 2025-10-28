# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Unit tests for the InitializerRegistry module.
"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from pyrit.cli.initializer_registry import InitializerInfo, InitializerRegistry
from pyrit.setup.initializers.pyrit_initializer import PyRITInitializer


class MockInitializer(PyRITInitializer):
    """Mock initializer for testing."""

    def __init__(
        self,
        *,
        mock_name: str = "test_initializer",
        mock_description: str = "Test description",
        mock_required_env_vars: list[str] | None = None,
        mock_execution_order: int = 100,
    ) -> None:
        """Initialize mock initializer."""
        super().__init__()
        self._mock_name = mock_name
        self._mock_description = mock_description
        self._mock_required_env_vars = mock_required_env_vars or []
        self._mock_execution_order = mock_execution_order

    @property
    def name(self) -> str:
        """Get the name."""
        return self._mock_name

    @property
    def description(self) -> str:
        """Get the description."""
        return self._mock_description

    @property
    def required_env_vars(self) -> list[str]:
        """Get required environment variables."""
        return self._mock_required_env_vars

    @property
    def execution_order(self) -> int:
        """Get execution order."""
        return self._mock_execution_order

    def initialize(self) -> None:
        """Mock initialization."""
        pass


class TestInitializerRegistry:
    """Tests for InitializerRegistry class."""

    @patch("pyrit.cli.initializer_registry.Path")
    def test_init_with_nonexistent_directory(self, mock_path_class):
        """Test initialization when scenarios directory doesn't exist."""
        # Create a mock path that represents a non-existent directory
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_path.is_file.return_value = False
        mock_path.is_dir.return_value = False
        
        # Make Path() constructor and division operations return the mock path
        mock_path_class.return_value = mock_path
        mock_path.__truediv__ = MagicMock(return_value=mock_path)

        registry = InitializerRegistry()

        assert registry._initializers == {}

    @patch("pyrit.cli.initializer_registry.Path")
    def test_init_discovers_initializers(self, mock_path_class):
        """Test initialization discovers initializers."""
        # Mock the directory structure
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.is_file.return_value = False
        mock_path.is_dir.return_value = True

        # Create a mock file
        mock_file = MagicMock()
        mock_file.is_file.return_value = True
        mock_file.is_dir.return_value = False
        mock_file.suffix = ".py"
        mock_file.stem = "test_init"

        mock_path.iterdir.return_value = [mock_file]
        mock_path_class.return_value = mock_path

        with patch.object(InitializerRegistry, "_process_file"):
            registry = InitializerRegistry()
            # Just verify it attempted to process files
            assert registry is not None

    def test_get_initializer_existing(self):
        """Test getting an existing initializer."""
        registry = InitializerRegistry()
        test_info: InitializerInfo = {
            "name": "test",
            "class_name": "TestInitializer",
            "initializer_name": "test_init",
            "description": "Test",
            "required_env_vars": [],
            "execution_order": 100,
        }
        registry._initializers["test"] = test_info

        result = registry.get_initializer("test")
        assert result == test_info

    def test_get_initializer_nonexistent(self):
        """Test getting a non-existent initializer returns None."""
        registry = InitializerRegistry()
        result = registry.get_initializer("nonexistent")
        assert result is None

    def test_get_initializer_names_empty(self):
        """Test get_initializer_names with no initializers."""
        registry = InitializerRegistry()
        registry._initializers = {}
        names = registry.get_initializer_names()
        assert names == []

    def test_get_initializer_names_sorted(self):
        """Test get_initializer_names returns sorted list."""
        registry = InitializerRegistry()
        test_info: InitializerInfo = {
            "name": "test",
            "class_name": "Test",
            "initializer_name": "test",
            "description": "Test",
            "required_env_vars": [],
            "execution_order": 100,
        }
        registry._initializers = {
            "zebra": test_info,
            "apple": test_info,
            "middle": test_info,
        }

        names = registry.get_initializer_names()
        assert names == ["apple", "middle", "zebra"]

    def test_list_initializers_sorted_by_execution_order(self):
        """Test list_initializers returns sorted list by execution order."""
        registry = InitializerRegistry()
        registry._initializers = {
            "first": {
                "name": "first",
                "class_name": "First",
                "initializer_name": "first",
                "description": "First",
                "required_env_vars": [],
                "execution_order": 10,
            },
            "third": {
                "name": "third",
                "class_name": "Third",
                "initializer_name": "third",
                "description": "Third",
                "required_env_vars": [],
                "execution_order": 30,
            },
            "second": {
                "name": "second",
                "class_name": "Second",
                "initializer_name": "second",
                "description": "Second",
                "required_env_vars": [],
                "execution_order": 20,
            },
        }

        initializers = registry.list_initializers()

        assert len(initializers) == 3
        assert initializers[0]["name"] == "first"
        assert initializers[1]["name"] == "second"
        assert initializers[2]["name"] == "third"

    def test_list_initializers_sorted_by_name_when_same_order(self):
        """Test list_initializers sorts by name when execution order is same."""
        registry = InitializerRegistry()
        registry._initializers = {
            "zebra": {
                "name": "zebra",
                "class_name": "Zebra",
                "initializer_name": "zebra",
                "description": "Zebra",
                "required_env_vars": [],
                "execution_order": 10,
            },
            "apple": {
                "name": "apple",
                "class_name": "Apple",
                "initializer_name": "apple",
                "description": "Apple",
                "required_env_vars": [],
                "execution_order": 10,
            },
        }

        initializers = registry.list_initializers()

        assert len(initializers) == 2
        assert initializers[0]["name"] == "apple"
        assert initializers[1]["name"] == "zebra"

    def test_list_initializers_empty(self):
        """Test list_initializers with no initializers."""
        registry = InitializerRegistry()
        registry._initializers = {}

        initializers = registry.list_initializers()
        assert initializers == []

    def test_discover_in_directory_skips_init_files(self):
        """Test that __init__.py files are skipped."""
        registry = InitializerRegistry()

        mock_init_file = MagicMock()
        mock_init_file.is_file.return_value = True
        mock_init_file.is_dir.return_value = False
        mock_init_file.suffix = ".py"
        mock_init_file.stem = "__init__"

        mock_directory = MagicMock()
        mock_directory.iterdir.return_value = [mock_init_file]

        with patch.object(registry, "_process_file") as mock_process:
            registry._discover_in_directory(directory=mock_directory)
            # Should not process __init__.py
            mock_process.assert_not_called()

    def test_discover_in_directory_skips_pycache(self):
        """Test that __pycache__ directories are skipped."""
        registry = InitializerRegistry()

        mock_pycache = MagicMock()
        mock_pycache.is_file.return_value = False
        mock_pycache.is_dir.return_value = True
        mock_pycache.name = "__pycache__"

        mock_directory = MagicMock()
        mock_directory.iterdir.return_value = [mock_pycache]

        with patch.object(registry, "_discover_in_directory") as mock_discover:
            registry._discover_in_directory(directory=mock_directory)
            # Should only be called once (the initial call), not recursively for __pycache__
            assert mock_discover.call_count == 1

    def test_discover_in_directory_processes_valid_files(self):
        """Test that valid Python files are processed."""
        registry = InitializerRegistry()

        mock_file = MagicMock()
        mock_file.is_file.return_value = True
        mock_file.is_dir.return_value = False
        mock_file.suffix = ".py"
        mock_file.stem = "valid_init"

        mock_directory = MagicMock()
        mock_directory.iterdir.return_value = [mock_file]

        with patch.object(registry, "_process_file") as mock_process:
            registry._discover_in_directory(directory=mock_directory)
            mock_process.assert_called_once_with(file_path=mock_file)

    def test_discover_in_directory_recurses_subdirectories(self):
        """Test that subdirectories are recursively processed."""
        registry = InitializerRegistry()

        mock_subdir = MagicMock()
        mock_subdir.is_file.return_value = False
        mock_subdir.is_dir.return_value = True
        mock_subdir.name = "subdir"

        mock_directory = MagicMock()
        mock_directory.iterdir.return_value = [mock_subdir]

        with patch.object(registry, "_discover_in_directory", wraps=registry._discover_in_directory) as mock_discover:
            # Call once to start
            registry._discover_in_directory(directory=mock_directory)
            # Should be called twice: initial + recursive call for subdir
            assert mock_discover.call_count == 2
            # Second call should have the subdirectory
            second_call_kwargs = mock_discover.call_args_list[1][1]
            assert second_call_kwargs["directory"] == mock_subdir

    @patch("pyrit.cli.initializer_registry.importlib.util.spec_from_file_location")
    @patch("pyrit.cli.initializer_registry.PYRIT_PATH", "/fake/pyrit")
    def test_process_file_handles_import_errors(self, mock_spec):
        """Test that import errors are handled gracefully."""
        registry = InitializerRegistry()
        registry._initializers.clear()
        registry._initializer_paths.clear()

        mock_spec.side_effect = Exception("Import error")

        # Create a proper Path object that can handle relative_to
        mock_file = Path("/fake/pyrit/setup/initializers/scenarios/broken.py")

        # Should not raise exception
        registry._process_file(file_path=mock_file)

        assert "broken" not in registry._initializers

    @patch("pyrit.cli.initializer_registry.importlib.util.spec_from_file_location")
    @patch("pyrit.cli.initializer_registry.PYRIT_PATH", "/fake/pyrit")
    def test_process_file_handles_no_spec(self, mock_spec):
        """Test handling when spec_from_file_location returns None."""
        registry = InitializerRegistry()
        registry._initializers.clear()
        registry._initializer_paths.clear()

        mock_spec.return_value = None

        # Create a proper Path object
        mock_file = Path("/fake/pyrit/setup/initializers/scenarios/no_spec.py")

        registry._process_file(file_path=mock_file)

        assert "no_spec" not in registry._initializers

    @patch("pyrit.cli.initializer_registry.importlib.util.spec_from_file_location")
    @patch("pyrit.cli.initializer_registry.PYRIT_PATH", "/fake/pyrit")
    def test_process_file_discovers_initializer_class(self, mock_spec):
        """Test that PyRITInitializer subclasses are discovered."""
        registry = InitializerRegistry()
        registry._initializers.clear()
        registry._initializer_paths.clear()

        # Create a mock module with our test initializer
        mock_module = MagicMock()
        mock_module.MockInitializer = MockInitializer

        mock_spec_obj = MagicMock()
        mock_spec_obj.loader = MagicMock()
        mock_spec.return_value = mock_spec_obj

        # Create a proper Path object
        mock_file = Path("/fake/pyrit/setup/initializers/scenarios/test_init.py")

        with patch("pyrit.cli.initializer_registry.importlib.util.module_from_spec", return_value=mock_module):
            with patch("pyrit.cli.initializer_registry.dir", return_value=["MockInitializer"]):
                with patch("pyrit.cli.initializer_registry.getattr", return_value=MockInitializer):
                    with patch("pyrit.cli.initializer_registry.inspect.isclass", return_value=True):
                        registry._process_file(file_path=mock_file)

        # Verify the initializer was registered
        assert "test_init" in registry._initializers

    def test_try_register_initializer_success(self):
        """Test successful registration of an initializer."""
        registry = InitializerRegistry()
        mock_path = Path("/fake/path/test.py")

        # Clear any auto-discovered initializers to ensure clean test
        registry._initializers.clear()
        registry._initializer_paths.clear()

        registry._try_register_initializer(
            initializer_class=MockInitializer, short_name="test", file_path=mock_path
        )

        assert "test" in registry._initializers
        info = registry._initializers["test"]
        assert info["name"] == "test"
        assert info["class_name"] == "MockInitializer"
        assert info["initializer_name"] == "test_initializer"
        assert info["description"] == "Test description"
        assert info["required_env_vars"] == []
        assert info["execution_order"] == 100
        assert registry._initializer_paths["test"] == mock_path

    def test_try_register_initializer_with_env_vars(self):
        """Test registration with required environment variables."""
        registry = InitializerRegistry()
        mock_path = Path("/fake/path/env_test.py")

        class EnvVarInitializer(PyRITInitializer):
            @property
            def name(self) -> str:
                return "env_test"

            @property
            def description(self) -> str:
                return "Test with env vars"

            @property
            def required_env_vars(self) -> list[str]:
                return ["API_KEY", "ENDPOINT"]

            @property
            def execution_order(self) -> int:
                return 50

            def initialize(self) -> None:
                pass

        # Clear any auto-discovered initializers to ensure clean test
        registry._initializers.clear()
        registry._initializer_paths.clear()

        registry._try_register_initializer(
            initializer_class=EnvVarInitializer, short_name="env_test", file_path=mock_path
        )

        assert "env_test" in registry._initializers
        info = registry._initializers["env_test"]
        assert info["required_env_vars"] == ["API_KEY", "ENDPOINT"]
        assert info["execution_order"] == 50

    def test_try_register_initializer_handles_instantiation_error(self):
        """Test that instantiation errors are handled gracefully."""
        registry = InitializerRegistry()
        mock_path = Path("/fake/path/broken.py")

        class BrokenInitializer(PyRITInitializer):
            def __init__(self) -> None:
                raise ValueError("Cannot instantiate")

            @property
            def name(self) -> str:
                return "broken"

            def initialize(self) -> None:
                pass

        # Should not raise exception
        registry._try_register_initializer(
            initializer_class=BrokenInitializer, short_name="broken", file_path=mock_path
        )

        # Should not be registered
        assert "broken" not in registry._initializers

    def test_initializer_info_typed_dict_structure(self):
        """Test that InitializerInfo TypedDict has correct structure."""
        info: InitializerInfo = {
            "name": "test",
            "class_name": "TestClass",
            "initializer_name": "test_init",
            "description": "Description",
            "required_env_vars": ["VAR1"],
            "execution_order": 10,
        }

        assert info["name"] == "test"
        assert info["class_name"] == "TestClass"
        assert info["initializer_name"] == "test_init"
        assert info["description"] == "Description"
        assert info["required_env_vars"] == ["VAR1"]
        assert info["execution_order"] == 10


class TestResolveInitializerPaths:
    """Tests for resolve_initializer_paths method."""

    def test_resolve_single_initializer(self):
        """Test resolving a single valid initializer name."""
        registry = InitializerRegistry()
        registry._initializers.clear()
        registry._initializer_paths.clear()

        test_path = Path("/fake/simple.py")
        registry._initializers["simple"] = {
            "name": "simple",
            "class_name": "SimpleInitializer",
            "initializer_name": "simple_init",
            "description": "Test",
            "required_env_vars": [],
            "execution_order": 100,
        }
        registry._initializer_paths["simple"] = test_path

        result = registry.resolve_initializer_paths(initializer_names=["simple"])

        assert len(result) == 1
        assert result[0] == test_path

    def test_resolve_multiple_initializers(self):
        """Test resolving multiple initializer names."""
        registry = InitializerRegistry()
        registry._initializers.clear()
        registry._initializer_paths.clear()

        path1 = Path("/fake/simple.py")
        path2 = Path("/fake/objective_target.py")

        registry._initializers["simple"] = {
            "name": "simple",
            "class_name": "SimpleInitializer",
            "initializer_name": "simple_init",
            "description": "Test",
            "required_env_vars": [],
            "execution_order": 100,
        }
        registry._initializer_paths["simple"] = path1

        registry._initializers["objective_target"] = {
            "name": "objective_target",
            "class_name": "ObjectiveTargetInitializer",
            "initializer_name": "obj_target_init",
            "description": "Test",
            "required_env_vars": [],
            "execution_order": 200,
        }
        registry._initializer_paths["objective_target"] = path2

        result = registry.resolve_initializer_paths(initializer_names=["simple", "objective_target"])

        assert len(result) == 2
        assert path1 in result
        assert path2 in result

    def test_resolve_invalid_initializer_name(self):
        """Test resolving an invalid initializer name raises ValueError."""
        registry = InitializerRegistry()
        registry._initializers.clear()
        registry._initializer_paths.clear()

        with pytest.raises(ValueError, match="Built-in initializer 'invalid' not found"):
            registry.resolve_initializer_paths(initializer_names=["invalid"])

    def test_resolve_initializer_without_file_path(self):
        """Test resolving initializer without file path raises ValueError."""
        registry = InitializerRegistry()
        registry._initializers.clear()
        registry._initializer_paths.clear()

        registry._initializers["simple"] = {
            "name": "simple",
            "class_name": "SimpleInitializer",
            "initializer_name": "simple_init",
            "description": "Test",
            "required_env_vars": [],
            "execution_order": 100,
        }
        # Intentionally not adding to _initializer_paths

        with pytest.raises(ValueError, match="Could not locate file for initializer 'simple'"):
            registry.resolve_initializer_paths(initializer_names=["simple"])


class TestResolveScriptPaths:
    """Tests for resolve_script_paths static method."""

    @patch("pyrit.cli.initializer_registry.Path")
    def test_resolve_absolute_path_exists(self, mock_path_class):
        """Test resolving absolute path that exists."""
        mock_path = MagicMock()
        mock_path.is_absolute.return_value = True
        mock_path.exists.return_value = True
        mock_path_class.return_value = mock_path

        result = InitializerRegistry.resolve_script_paths(script_paths=["/absolute/script.py"])

        assert len(result) == 1

    @patch("pyrit.cli.initializer_registry.Path")
    def test_resolve_relative_path_exists(self, mock_path_class):
        """Test resolving relative path that exists."""
        mock_path = MagicMock()
        mock_path.is_absolute.return_value = False
        mock_path.exists.return_value = True

        mock_cwd = MagicMock()
        mock_resolved_path = MagicMock()
        mock_resolved_path.exists.return_value = True

        mock_cwd.__truediv__ = lambda self, other: mock_resolved_path
        mock_path_class.return_value = mock_path
        mock_path_class.cwd.return_value = mock_cwd

        result = InitializerRegistry.resolve_script_paths(script_paths=["script.py"])

        assert len(result) == 1

    @patch("pyrit.cli.initializer_registry.Path")
    def test_resolve_path_not_exists(self, mock_path_class):
        """Test resolving path that doesn't exist raises FileNotFoundError."""
        mock_path = MagicMock()
        mock_path.is_absolute.return_value = True
        mock_path.exists.return_value = False
        mock_path.absolute.return_value = "/fake/missing.py"
        mock_path_class.return_value = mock_path

        with pytest.raises(FileNotFoundError, match="Initialization script not found"):
            InitializerRegistry.resolve_script_paths(script_paths=["/fake/missing.py"])

    @patch("pyrit.cli.initializer_registry.Path")
    def test_resolve_multiple_paths(self, mock_path_class):
        """Test resolving multiple script paths."""
        mock_path1 = MagicMock()
        mock_path1.is_absolute.return_value = True
        mock_path1.exists.return_value = True

        mock_path2 = MagicMock()
        mock_path2.is_absolute.return_value = True
        mock_path2.exists.return_value = True

        # Make Path() return different mocks for different inputs
        def path_side_effect(path_str):
            if "script1" in str(path_str):
                return mock_path1
            return mock_path2

        mock_path_class.side_effect = path_side_effect

        result = InitializerRegistry.resolve_script_paths(script_paths=["/fake/script1.py", "/fake/script2.py"])

        assert len(result) == 2
