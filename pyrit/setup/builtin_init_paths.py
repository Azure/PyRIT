# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Provides convenient access to built-in initialization script paths.

This module defines a hierarchical path structure for accessing PyRIT's built-in
initialization scripts that can be used with initialize_pyrit().

Example:
    from pyrit.setup import BuiltInInitPath, initialize_pyrit

    # Use a built-in initialization script
    initialize_pyrit(
        memory_db_type="InMemory",
        initialization_scripts=[BuiltInInitPath.foundry.ansi_attack]
    )
"""

import pathlib
from typing import List, Literal, Union


class _FoundryPaths:
    """Paths to foundry-related initialization scripts."""

    def __init__(self, base_path: pathlib.Path) -> None:
        self._base_path = base_path

    @property
    def ansi_attack(self) -> pathlib.Path:
        """Path to the ANSI attack initialization script."""
        return self._base_path / "ansi_attack.py"


class _DefaultsPaths:
    """Paths to default initialization scripts."""

    def __init__(self, base_path: pathlib.Path) -> None:
        self._base_path = base_path

    @property
    def converter_initialization(self) -> pathlib.Path:
        """Path to the converter initialization script."""
        return self._base_path / "converter_initialization.py"

    @property
    def scorer_initialization(self) -> pathlib.Path:
        """Path to the scorer initialization script."""
        return self._base_path / "scorer_initialization.py"

    @property
    def target_initialization(self) -> pathlib.Path:
        """Path to the target initialization script."""
        return self._base_path / "target_initialization.py"


class _BuiltInInitPath:
    """
    Provides hierarchical access to built-in initialization script paths.

    This class organizes PyRIT's built-in initialization scripts by category,
    making them easily discoverable and accessible.

    Attributes:
        foundry: Access to foundry-related initialization scripts.
        defaults: Access to default initialization scripts.

    Example:
        # Access a foundry script
        path = BuiltInInitPath.foundry.ansi_attack

        # Access a defaults script
        path = BuiltInInitPath.defaults.scorer_initialization

        # Use with initialize_pyrit
        initialize_pyrit(
            memory_db_type="InMemory",
            initialization_scripts=[BuiltInInitPath.foundry.ansi_attack]
        )

        # List all paths in a specific subdirectory
        default_paths = BuiltInInitPath.list_all_paths("defaults")

        # List all paths
        all_paths = BuiltInInitPath.list_all_paths()
    """

    def __init__(self) -> None:
        self._CONFIG_PATH = pathlib.Path(__file__).parent / "config"
        self._FOUNDRY_PATH = self._CONFIG_PATH / "foundry"
        self._DEFAULTS_PATH = self._CONFIG_PATH / "defaults"
        self.foundry = _FoundryPaths(self._FOUNDRY_PATH)
        self.defaults = _DefaultsPaths(self._DEFAULTS_PATH)

    @classmethod
    def list_all_paths(
        cls, subdirectory: Literal["foundry", "defaults"] | None = None
    ) -> List[Union[str, pathlib.Path]]:
        """
        Get a list of all available built-in initialization script paths.

        Args:
            subdirectory: Optional subdirectory to filter paths. Can be "foundry" or "defaults".
                         If None, returns all paths.

        Returns:
            List[Union[str, pathlib.Path]]: List of built-in initialization script paths.

        Example:
            # Get all paths
            all_paths = BuiltInInitPath.list_all_paths()

            # Get only defaults paths
            defaults_paths = BuiltInInitPath.list_all_paths("defaults")

            # Get only foundry paths
            foundry_paths = BuiltInInitPath.list_all_paths("foundry")
        """
        instance = cls()
        paths: List[Union[str, pathlib.Path]] = []

        if subdirectory is None or subdirectory == "foundry":
            # Add foundry paths
            paths.append(instance.foundry.ansi_attack)

        if subdirectory is None or subdirectory == "defaults":
            # Add defaults paths
            paths.append(instance.defaults.converter_initialization)
            paths.append(instance.defaults.scorer_initialization)
            paths.append(instance.defaults.target_initialization)

        return paths


# Create a singleton instance for easy access
BuiltInInitPath = _BuiltInInitPath()
