# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Provides convenient access to configuration script paths.

This module defines a hierarchical path structure for accessing PyRIT's
configuration scripts that can be used with initialize_pyrit().

Example:
    from pyrit.setup import ConfigurationPaths, initialize_pyrit

    # Use a configuration script
    initialize_pyrit(
        memory_db_type="InMemory",
        initialization_scripts=[ConfigurationPaths.attack.foundry.ansi_attack]
    )
"""

import pathlib
from typing import List, Literal, Union


class _FoundryPaths:
    """Paths to attack foundry initialization scripts."""

    def __init__(self, base_path: pathlib.Path) -> None:
        self._base_path = base_path

    @property
    def ansi_attack(self) -> pathlib.Path:
        """Path to the ANSI attack initialization script."""
        return self._base_path / "ansi_attack.py"

    @property
    def ascii_art(self) -> pathlib.Path:
        """Path to the ASCII art initialization script."""
        return self._base_path / "ascii_art.py"

    @property
    def crescendo(self) -> pathlib.Path:
        """Path to the Crescendo initialization script."""
        return self._base_path / "crescendo.py"
    
    @property
    def tense(self) -> pathlib.Path:
        """Path to the Tense initialization script."""
        return self._base_path / "tense.py"


class _AttackPaths:
    """Paths to attack-related initialization scripts."""

    def __init__(self, base_path: pathlib.Path) -> None:
        self._base_path = base_path
        self.foundry = _FoundryPaths(self._base_path / "foundry")


class _DatasetPaths:
    """Paths to dataset configuration scripts."""

    def __init__(self, base_path: pathlib.Path) -> None:
        self._base_path = base_path

    @property
    def harm_bench(self) -> pathlib.Path:
        """Path to the HarmBench dataset configuration script."""
        return self._base_path / "harm_bench.py"


class _DefaultsPaths:
    """Paths to initialization default scripts."""

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


class _InitializationPaths:
    """Paths to initialization-related scripts."""

    def __init__(self, base_path: pathlib.Path) -> None:
        self._base_path = base_path
        self.defaults = _DefaultsPaths(self._base_path / "defaults")


class _ConfigurationPaths:
    """
    Provides hierarchical access to configuration script paths.

    This class organizes PyRIT's configuration scripts by category,
    making them easily discoverable and accessible using dot notation.

    Attributes:
        attack: Access to attack-related configuration scripts.
        dataset: Access to dataset configuration scripts.
        initialization: Access to initialization configuration scripts.

    Example:
        # Access an attack foundry script
        path = ConfigurationPaths.attack.foundry.ansi_attack

        # Access a dataset configuration script
        path = ConfigurationPaths.dataset.harm_bench

        # Access an initialization defaults script
        path = ConfigurationPaths.initialization.defaults.scorer_initialization

        # Use with initialize_pyrit
        initialize_pyrit(
            memory_db_type="InMemory",
            initialization_scripts=[ConfigurationPaths.attack.foundry.ansi_attack]
        )

        # List all paths in a specific subdirectory
        attack_paths = ConfigurationPaths.list_all_paths("attack.foundry")

        # List all paths
        all_paths = ConfigurationPaths.list_all_paths()
    """

    def __init__(self) -> None:
        self._CONFIG_PATH = pathlib.Path(__file__).parent / "config"
        self.attack = _AttackPaths(self._CONFIG_PATH / "attack")
        self.dataset = _DatasetPaths(self._CONFIG_PATH / "datasets")
        self.initialization = _InitializationPaths(self._CONFIG_PATH / "initialization")

    @classmethod
    def list_all_paths(
        cls, subdirectory: Literal["attack.foundry", "dataset", "initialization.defaults"] | None = None
    ) -> List[pathlib.Path]:
        """
        Get a list of all available configuration script paths.

        Args:
            subdirectory: Optional subdirectory to filter paths. Can be "attack.foundry", "dataset", or
                         "initialization.defaults". If None, returns all paths.

        Returns:
            List[pathlib.Path]: List of configuration script paths.

        Example:
            # Get all paths
            all_paths = ConfigurationPaths.list_all_paths()

            # Get only initialization.defaults paths
            defaults_paths = ConfigurationPaths.list_all_paths("initialization.defaults")

            # Get only attack.foundry paths
            foundry_paths = ConfigurationPaths.list_all_paths("attack.foundry")

            # Get only dataset paths
            dataset_paths = ConfigurationPaths.list_all_paths("dataset")
        """
        instance = cls()
        paths: List[pathlib.Path] = []

        if subdirectory is None or subdirectory == "attack.foundry":
            # Add attack foundry paths
            paths.append(instance.attack.foundry.ansi_attack)
            paths.append(instance.attack.foundry.ascii_art)
            paths.append(instance.attack.foundry.crescendo)
            paths.append(instance.attack.foundry.tense)

        if subdirectory is None or subdirectory == "dataset":
            # Add dataset paths
            paths.append(instance.dataset.harm_bench)

        if subdirectory is None or subdirectory == "initialization.defaults":
            # Add initialization defaults paths
            paths.append(instance.initialization.defaults.converter_initialization)
            paths.append(instance.initialization.defaults.scorer_initialization)
            paths.append(instance.initialization.defaults.target_initialization)

        return paths


# Create a singleton instance for easy access
ConfigurationPaths = _ConfigurationPaths()
