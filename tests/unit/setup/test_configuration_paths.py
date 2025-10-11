# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib

import pytest

from pyrit.setup import ConfigurationPaths


class TestConfigurationPaths:
    """Tests for the ConfigurationPaths class and its path accessors."""

    def test_attack_foundry_ansi_attack_path_exists(self) -> None:
        """Test that the attack.foundry.ansi_attack path is correctly defined."""
        path = ConfigurationPaths.attack.foundry.ansi_attack
        assert isinstance(path, pathlib.Path)
        assert path.name == "ansi_attack.py"
        assert "attack" in str(path)
        assert "foundry" in str(path)
        assert path.exists(), f"Expected attack foundry ansi_attack path to exist: {path}"

    def test_initialization_defaults_converter_initialization_path_exists(self) -> None:
        """Test that the initialization.defaults.converter_initialization path is correctly defined."""
        path = ConfigurationPaths.initialization.defaults.converter_initialization
        assert isinstance(path, pathlib.Path)
        assert path.name == "converter_initialization.py"
        assert "initialization" in str(path)
        assert "defaults" in str(path)
        assert path.exists(), f"Expected initialization defaults converter_initialization path to exist: {path}"

    def test_initialization_defaults_scorer_initialization_path_exists(self) -> None:
        """Test that the initialization.defaults.scorer_initialization path is correctly defined."""
        path = ConfigurationPaths.initialization.defaults.scorer_initialization
        assert isinstance(path, pathlib.Path)
        assert path.name == "scorer_initialization.py"
        assert "initialization" in str(path)
        assert "defaults" in str(path)
        assert path.exists(), f"Expected initialization defaults scorer_initialization path to exist: {path}"

    def test_initialization_defaults_target_initialization_path_exists(self) -> None:
        """Test that the initialization.defaults.target_initialization path is correctly defined."""
        path = ConfigurationPaths.initialization.defaults.target_initialization
        assert isinstance(path, pathlib.Path)
        assert path.name == "target_initialization.py"
        assert "initialization" in str(path)
        assert "defaults" in str(path)
        assert path.exists(), f"Expected initialization defaults target_initialization path to exist: {path}"

    def test_list_all_paths_returns_all_paths(self) -> None:
        """Test that list_all_paths() returns all available paths."""
        all_paths = ConfigurationPaths.list_all_paths()

        assert isinstance(all_paths, list)
        assert len(all_paths) == 7, f"Expected 7 paths, got {len(all_paths)}"

        # Verify all paths are Path objects
        for path in all_paths:
            assert isinstance(path, pathlib.Path)

        # Verify specific paths are included
        path_names = [p.name for p in all_paths]
        assert "ansi_attack.py" in path_names
        assert "ascii_art.py" in path_names
        assert "crescendo.py" in path_names
        assert "tense.py" in path_names
        assert "converter_initialization.py" in path_names
        assert "scorer_initialization.py" in path_names
        assert "target_initialization.py" in path_names

    def test_list_all_paths_with_initialization_defaults_subdirectory(self) -> None:
        """Test that list_all_paths('initialization.defaults') returns only initialization defaults paths."""
        defaults_paths = ConfigurationPaths.list_all_paths("initialization.defaults")

        assert isinstance(defaults_paths, list)
        assert len(defaults_paths) == 3, f"Expected 3 initialization defaults paths, got {len(defaults_paths)}"

        # Verify all paths are in the initialization defaults directory
        for path in defaults_paths:
            assert isinstance(path, pathlib.Path)
            assert "initialization" in str(path)
            assert "defaults" in str(path)

        # Verify specific defaults paths are included
        path_names = [p.name for p in defaults_paths]
        assert "converter_initialization.py" in path_names
        assert "scorer_initialization.py" in path_names
        assert "target_initialization.py" in path_names

        # Verify attack foundry paths are NOT included
        assert "ansi_attack.py" not in path_names

    def test_list_all_paths_with_attack_foundry_subdirectory(self) -> None:
        """Test that list_all_paths('attack.foundry') returns only attack foundry paths."""
        foundry_paths = ConfigurationPaths.list_all_paths("attack.foundry")

        assert isinstance(foundry_paths, list)
        assert len(foundry_paths) == 4, f"Expected 4 attack foundry paths, got {len(foundry_paths)}"

        # Verify all paths are in the attack foundry directory
        for path in foundry_paths:
            assert isinstance(path, pathlib.Path)
            assert "attack" in str(path)
            assert "foundry" in str(path)

        # Verify specific foundry paths are included
        path_names = [p.name for p in foundry_paths]
        assert "ansi_attack.py" in path_names
        assert "ascii_art.py" in path_names
        assert "crescendo.py" in path_names
        assert "tense.py" in path_names

        # Verify initialization defaults paths are NOT included
        assert "converter_initialization.py" not in path_names
        assert "scorer_initialization.py" not in path_names
        assert "target_initialization.py" not in path_names

    def test_list_all_paths_with_none_subdirectory_returns_all(self) -> None:
        """Test that list_all_paths(None) returns all paths (same as no argument)."""
        all_paths_no_arg = ConfigurationPaths.list_all_paths()
        all_paths_none_arg = ConfigurationPaths.list_all_paths(None)

        assert len(all_paths_no_arg) == len(all_paths_none_arg)

        # Convert to sets of path names for comparison
        paths_no_arg_names = {p.name for p in all_paths_no_arg}
        paths_none_arg_names = {p.name for p in all_paths_none_arg}

        assert paths_no_arg_names == paths_none_arg_names

    def test_paths_point_to_config_directory(self) -> None:
        """Test that all paths point to the config directory."""
        # Get the expected config directory
        module_dir = pathlib.Path(__file__).parent.parent.parent.parent / "pyrit" / "setup" / "config"

        all_paths = ConfigurationPaths.list_all_paths()
        for path in all_paths:
            assert module_dir in path.parents, f"Path {path} should be under config directory {module_dir}"

    def test_singleton_instance_consistency(self) -> None:
        """Test that the singleton instance provides consistent paths."""
        # Access paths multiple times
        path1 = ConfigurationPaths.attack.foundry.ansi_attack
        path2 = ConfigurationPaths.attack.foundry.ansi_attack

        # Verify they are the same
        assert path1 == path2

        # Verify initialization defaults paths are also consistent
        defaults_path1 = ConfigurationPaths.initialization.defaults.scorer_initialization
        defaults_path2 = ConfigurationPaths.initialization.defaults.scorer_initialization

        assert defaults_path1 == defaults_path2

    def test_all_returned_paths_exist(self) -> None:
        """Test that all paths returned by list_all_paths() exist in the filesystem."""
        all_paths = ConfigurationPaths.list_all_paths()

        for path in all_paths:
            assert path.exists(), f"Path does not exist: {path}"
            assert path.is_file(), f"Path is not a file: {path}"
            assert path.suffix == ".py", f"Path is not a Python file: {path}"


class TestAttackFoundryPaths:
    """Tests specifically for the attack foundry paths."""

    def test_attack_foundry_paths_has_expected_attributes(self) -> None:
        """Test that attack foundry paths has all expected attributes."""
        foundry = ConfigurationPaths.attack.foundry

        assert hasattr(foundry, "ansi_attack")
        assert hasattr(foundry, "ascii_art")
        assert hasattr(foundry, "crescendo")
        assert hasattr(foundry, "tense")
        assert hasattr(foundry, "_base_path")


class TestInitializationDefaultsPaths:
    """Tests specifically for the initialization defaults paths."""

    def test_initialization_defaults_paths_has_expected_attributes(self) -> None:
        """Test that initialization defaults paths has all expected attributes."""
        defaults = ConfigurationPaths.initialization.defaults

        assert hasattr(defaults, "converter_initialization")
        assert hasattr(defaults, "scorer_initialization")
        assert hasattr(defaults, "target_initialization")
        assert hasattr(defaults, "_base_path")

    def test_initialization_defaults_base_path_is_correct(self) -> None:
        """Test that the initialization defaults base path points to the correct directory."""
        defaults = ConfigurationPaths.initialization.defaults
        base_path = defaults._base_path

        assert isinstance(base_path, pathlib.Path)
        assert base_path.name == "defaults"
        assert base_path.exists()
        assert base_path.is_dir()
