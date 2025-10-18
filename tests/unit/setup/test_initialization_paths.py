# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib

from pyrit.setup.initialization_paths import InitializationPaths, initialization_paths


class TestInitializationPaths:
    """Tests for the InitializationPaths class and its path accessors."""

    def test_airt_converter_initialization_path_exists(self) -> None:
        """Test that the airt_converter_initialization path is correctly defined."""
        path = initialization_paths.airt_converter_initialization
        assert isinstance(path, pathlib.Path)
        assert path.name == "converter_initialization.py"
        assert "initialization_scripts" in str(path)
        assert "airt" in str(path)
        assert path.exists(), f"Expected airt_converter_initialization path to exist: {path}"

    def test_airt_scorer_initialization_path_exists(self) -> None:
        """Test that the airt_scorer_initialization path is correctly defined."""
        path = initialization_paths.airt_scorer_initialization
        assert isinstance(path, pathlib.Path)
        assert path.name == "scorer_initialization.py"
        assert "initialization_scripts" in str(path)
        assert "airt" in str(path)
        assert path.exists(), f"Expected airt_scorer_initialization path to exist: {path}"

    def test_airt_target_initialization_path_exists(self) -> None:
        """Test that the airt_target_initialization path is correctly defined."""
        path = initialization_paths.airt_target_initialization
        assert isinstance(path, pathlib.Path)
        assert path.name == "target_initialization.py"
        assert "initialization_scripts" in str(path)
        assert "airt" in str(path)
        assert path.exists(), f"Expected airt_target_initialization path to exist: {path}"

    def test_simple_converter_initialization_path_exists(self) -> None:
        """Test that the simple_converter_initialization path is correctly defined."""
        path = initialization_paths.simple_converter_initialization
        assert isinstance(path, pathlib.Path)
        assert path.name == "converter_initialization.py"
        assert "initialization_scripts" in str(path)
        assert "simple" in str(path)
        assert path.exists(), f"Expected simple_converter_initialization path to exist: {path}"

    def test_simple_scorer_initialization_path_exists(self) -> None:
        """Test that the simple_scorer_initialization path is correctly defined."""
        path = initialization_paths.simple_scorer_initialization
        assert isinstance(path, pathlib.Path)
        assert path.name == "scorer_initialization.py"
        assert "initialization_scripts" in str(path)
        assert "simple" in str(path)
        assert path.exists(), f"Expected simple_scorer_initialization path to exist: {path}"

    def test_simple_target_initialization_path_exists(self) -> None:
        """Test that the simple_target_initialization path is correctly defined."""
        path = initialization_paths.simple_target_initialization
        assert isinstance(path, pathlib.Path)
        assert path.name == "target_initialization.py"
        assert "initialization_scripts" in str(path)
        assert "simple" in str(path)
        assert path.exists(), f"Expected simple_target_initialization path to exist: {path}"

    def test_list_all_airt_paths_returns_all_airt_paths(self) -> None:
        """Test that list_all_airt_paths() returns all AIRT paths."""
        from pyrit.setup.initialization_paths import InitializationPaths

        airt_paths = InitializationPaths.list_all_airt_paths()

        assert isinstance(airt_paths, list)
        assert len(airt_paths) == 3, f"Expected 3 AIRT paths, got {len(airt_paths)}"

        # Verify all paths are Path objects
        for path in airt_paths:
            assert isinstance(path, pathlib.Path)
            assert "airt" in str(path)

        # Verify specific paths are included
        path_names = [p.name for p in airt_paths]
        assert "converter_initialization.py" in path_names
        assert "scorer_initialization.py" in path_names
        assert "target_initialization.py" in path_names

    def test_list_all_simple_paths_returns_all_simple_paths(self) -> None:
        """Test that list_all_simple_paths() returns all simple paths."""
        from pyrit.setup.initialization_paths import InitializationPaths

        simple_paths = InitializationPaths.list_all_simple_paths()

        assert isinstance(simple_paths, list)
        assert len(simple_paths) == 3, f"Expected 3 simple paths, got {len(simple_paths)}"

        # Verify all paths are Path objects
        for path in simple_paths:
            assert isinstance(path, pathlib.Path)
            assert "simple" in str(path)

        # Verify specific paths are included
        path_names = [p.name for p in simple_paths]
        assert "converter_initialization.py" in path_names
        assert "scorer_initialization.py" in path_names
        assert "target_initialization.py" in path_names

    def test_airt_paths_point_to_config_airt_directory(self) -> None:
        """Test that all AIRT paths point to the config/airt directory."""
        from pyrit.setup.initialization_paths import InitializationPaths

        # Get the expected config directory
        module_dir = pathlib.Path(__file__).parent.parent.parent.parent / "pyrit" / "setup" / "initialization_scripts" / "airt"

        airt_paths = InitializationPaths.list_all_airt_paths()
        for path in airt_paths:
            assert (
                module_dir in path.parents or path.parent == module_dir
            ), f"Path {path} should be under config/airt directory {module_dir}"

    def test_simple_paths_point_to_config_simple_directory(self) -> None:
        """Test that all simple paths point to the config/simple directory."""
        from pyrit.setup.initialization_paths import InitializationPaths

        # Get the expected config directory
        module_dir = pathlib.Path(__file__).parent.parent.parent.parent / "pyrit" / "setup" / "initialization_scripts" / "simple"

        simple_paths = InitializationPaths.list_all_simple_paths()
        for path in simple_paths:
            assert (
                module_dir in path.parents or path.parent == module_dir
            ), f"Path {path} should be under config/simple directory {module_dir}"

    def test_singleton_instance_consistency(self) -> None:
        """Test that the singleton instance provides consistent paths."""
        # Access paths multiple times
        path1 = initialization_paths.airt_converter_initialization
        path2 = initialization_paths.airt_converter_initialization

        # Verify they are the same
        assert path1 == path2

        # Verify other paths are also consistent
        scorer_path1 = initialization_paths.simple_scorer_initialization
        scorer_path2 = initialization_paths.simple_scorer_initialization

        assert scorer_path1 == scorer_path2

    def test_all_airt_returned_paths_exist(self) -> None:
        """Test that all AIRT paths returned by list_all_airt_paths() exist in the filesystem."""
        airt_paths = InitializationPaths.list_all_airt_paths()

        for path in airt_paths:
            assert path.exists(), f"Path does not exist: {path}"
            assert path.is_file(), f"Path is not a file: {path}"
            assert path.suffix == ".py", f"Path is not a Python file: {path}"

    def test_all_simple_returned_paths_exist(self) -> None:
        """Test that all simple paths returned by list_all_simple_paths() exist in the filesystem."""
        simple_paths = InitializationPaths.list_all_simple_paths()

        for path in simple_paths:
            assert path.exists(), f"Path does not exist: {path}"
            assert path.is_file(), f"Path is not a file: {path}"
            assert path.suffix == ".py", f"Path is not a Python file: {path}"

    def test_class_has_expected_attributes(self) -> None:
        """Test that InitializationPaths instance has all expected attributes."""
        assert hasattr(initialization_paths, "airt_converter_initialization")
        assert hasattr(initialization_paths, "airt_scorer_initialization")
        assert hasattr(initialization_paths, "airt_target_initialization")
        assert hasattr(initialization_paths, "simple_converter_initialization")
        assert hasattr(initialization_paths, "simple_scorer_initialization")
        assert hasattr(initialization_paths, "simple_target_initialization")
        assert hasattr(initialization_paths, "_CONFIG_PATH")

    def test_config_path_is_correct(self) -> None:
        """Test that the config path points to the correct directory."""
        config_path = initialization_paths._CONFIG_PATH

        assert isinstance(config_path, pathlib.Path)
        assert config_path.name == "initialization_scripts"
        assert config_path.exists()
        assert config_path.is_dir()
