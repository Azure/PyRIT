# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib

from pyrit.setup.initialization_paths import InitializationPaths, initialization_paths


class TestInitializationPaths:
    """Tests for the InitializationPaths class and its path accessors."""

    def test_converter_initialization_path_exists(self) -> None:
        """Test that the converter_initialization path is correctly defined."""
        path = initialization_paths.converter_initialization
        assert isinstance(path, pathlib.Path)
        assert path.name == "converter_initialization.py"
        assert "config" in str(path)
        assert path.exists(), f"Expected converter_initialization path to exist: {path}"

    def test_scorer_initialization_path_exists(self) -> None:
        """Test that the scorer_initialization path is correctly defined."""
        path = initialization_paths.scorer_initialization
        assert isinstance(path, pathlib.Path)
        assert path.name == "scorer_initialization.py"
        assert "config" in str(path)
        assert path.exists(), f"Expected scorer_initialization path to exist: {path}"

    def test_target_initialization_path_exists(self) -> None:
        """Test that the target_initialization path is correctly defined."""
        path = initialization_paths.target_initialization
        assert isinstance(path, pathlib.Path)
        assert path.name == "target_initialization.py"
        assert "config" in str(path)
        assert path.exists(), f"Expected target_initialization path to exist: {path}"

    def test_list_all_paths_returns_all_paths(self) -> None:
        """Test that list_all_paths() returns all available paths."""
        from pyrit.setup.initialization_paths import InitializationPaths

        all_paths = InitializationPaths.list_all_paths()

        assert isinstance(all_paths, list)
        assert len(all_paths) == 3, f"Expected 3 paths, got {len(all_paths)}"

        # Verify all paths are Path objects
        for path in all_paths:
            assert isinstance(path, pathlib.Path)

        # Verify specific paths are included
        path_names = [p.name for p in all_paths]
        assert "converter_initialization.py" in path_names
        assert "scorer_initialization.py" in path_names
        assert "target_initialization.py" in path_names

    def test_paths_point_to_config_directory(self) -> None:
        """Test that all paths point to the config directory."""
        from pyrit.setup.initialization_paths import InitializationPaths

        # Get the expected config directory
        module_dir = pathlib.Path(__file__).parent.parent.parent.parent / "pyrit" / "setup" / "config"

        all_paths = InitializationPaths.list_all_paths()
        for path in all_paths:
            assert module_dir in path.parents, f"Path {path} should be under config directory {module_dir}"

    def test_singleton_instance_consistency(self) -> None:
        """Test that the singleton instance provides consistent paths."""
        # Access paths multiple times
        path1 = initialization_paths.converter_initialization
        path2 = initialization_paths.converter_initialization

        # Verify they are the same
        assert path1 == path2

        # Verify other paths are also consistent
        scorer_path1 = initialization_paths.scorer_initialization
        scorer_path2 = initialization_paths.scorer_initialization

        assert scorer_path1 == scorer_path2

    def test_all_returned_paths_exist(self) -> None:
        """Test that all paths returned by list_all_paths() exist in the filesystem."""
        all_paths = InitializationPaths.list_all_paths()

        for path in all_paths:
            assert path.exists(), f"Path does not exist: {path}"
            assert path.is_file(), f"Path is not a file: {path}"
            assert path.suffix == ".py", f"Path is not a Python file: {path}"

    def test_class_has_expected_attributes(self) -> None:
        """Test that InitializationPaths instance has all expected attributes."""
        assert hasattr(initialization_paths, "converter_initialization")
        assert hasattr(initialization_paths, "scorer_initialization")
        assert hasattr(initialization_paths, "target_initialization")
        assert hasattr(initialization_paths, "_CONFIG_PATH")

    def test_config_path_is_correct(self) -> None:
        """Test that the config path points to the correct directory."""
        config_path = initialization_paths._CONFIG_PATH

        assert isinstance(config_path, pathlib.Path)
        assert config_path.name == "config"
        assert config_path.exists()
        assert config_path.is_dir()
