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
from typing import List, Literal


class _AttackPaths:
    """Paths to attack-related configuration scripts."""

    def __init__(self, base_path: pathlib.Path) -> None:
        self._base_path = base_path

    @property
    def crescendo(self) -> pathlib.Path:
        """Path to the Crescendo attack configuration script."""
        return self._base_path / "crescendo.py"

    @property
    def prompt_sending(self) -> pathlib.Path:
        """Path to the Prompt attack configuration script."""
        return self._base_path / "prompt_sending.py"


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


class _ScenarioPaths:
    """Paths to scenario configuration scripts."""

    def __init__(self, base_path: pathlib.Path) -> None:
        self._base_path = base_path

    @property
    def foundry(self) -> pathlib.Path:
        """Path to the Foundry comprehensive test scenario."""
        return self._base_path / "foundry.py"


class _ConverterPaths:
    """Paths to converter configuration scripts."""

    def __init__(self, base_path: pathlib.Path) -> None:
        self._base_path = base_path

    @property
    def ansi_attack(self) -> pathlib.Path:
        """Path to the ANSI attack converter configuration script."""
        return self._base_path / "ansi_attack.py"

    @property
    def ascii_art(self) -> pathlib.Path:
        """Path to the ASCII art converter configuration script."""
        return self._base_path / "ascii_art.py"

    @property
    def base64(self) -> pathlib.Path:
        """Path to the Base64 converter configuration script."""
        return self._base_path / "base64.py"

    @property
    def leetspeak(self) -> pathlib.Path:
        """Path to the Leetspeak converter configuration script."""
        return self._base_path / "leetspeak.py"

    @property
    def rot13(self) -> pathlib.Path:
        """Path to the ROT13 converter configuration script."""
        return self._base_path / "rot13.py"


class _ConfigurationPaths:
    """
    Provides hierarchical access to configuration script paths.

    This class organizes PyRIT's configuration scripts by category,
    making them easily discoverable and accessible using dot notation.

    Attributes:
        attack: Access to attack-related configuration scripts.
        converter: Access to converter configuration scripts.
        dataset: Access to dataset configuration scripts.
        initialization: Access to initialization configuration scripts.
        scenario: Access to scenario configuration scripts.

    Example:
        # Access an attack script
        path = ConfigurationPaths.attack.crescendo

        # Access a converter configuration script
        path = ConfigurationPaths.converter.base64

        # Access a dataset configuration script
        path = ConfigurationPaths.dataset.harm_bench

        # Access an initialization defaults script
        path = ConfigurationPaths.initialization.defaults.scorer_initialization

        # Access a scenario configuration script
        path = ConfigurationPaths.scenario.foundry

        # Use with initialize_pyrit
        initialize_pyrit(
            memory_db_type="InMemory",
            initialization_scripts=[ConfigurationPaths.attack.crescendo]
        )

        # List all paths in a specific subdirectory
        attack_paths = ConfigurationPaths.list_all_paths("attack")

        # List all paths
        all_paths = ConfigurationPaths.list_all_paths()
    """

    def __init__(self) -> None:
        self._CONFIG_PATH = pathlib.Path(__file__).parent / "config"
        self.attack = _AttackPaths(self._CONFIG_PATH / "attack")
        self.converter = _ConverterPaths(self._CONFIG_PATH / "converters")
        self.dataset = _DatasetPaths(self._CONFIG_PATH / "datasets")
        self.initialization = _InitializationPaths(self._CONFIG_PATH / "initialization")
        self.scenario = _ScenarioPaths(self._CONFIG_PATH / "scenarios")

    @classmethod
    def list_all_paths(
        cls, subdirectory: Literal["attack", "converter", "dataset", "initialization.defaults"] | None = None
    ) -> List[pathlib.Path]:
        """
        Get a list of all available configuration script paths.

        Args:
            subdirectory: Optional subdirectory to filter paths. Can be "attack", "converter", "dataset", or
                         "initialization.defaults". If None, returns all paths.

        Returns:
            List[pathlib.Path]: List of configuration script paths.

        Example:
            # Get all paths
            all_paths = ConfigurationPaths.list_all_paths()

            # Get only initialization.defaults paths
            defaults_paths = ConfigurationPaths.list_all_paths("initialization.defaults")

            # Get only attack paths
            attack_paths = ConfigurationPaths.list_all_paths("attack")

            # Get only converter paths
            converter_paths = ConfigurationPaths.list_all_paths("converter")

            # Get only dataset paths
            dataset_paths = ConfigurationPaths.list_all_paths("dataset")
        """
        instance = cls()
        paths: List[pathlib.Path] = []

        if subdirectory is None or subdirectory == "attack":
            # Add attack paths
            paths.append(instance.attack.crescendo)
            paths.append(instance.attack.prompt_sending)

        if subdirectory is None or subdirectory == "converter":
            # Add converter paths
            paths.append(instance.converter.ansi_attack)
            paths.append(instance.converter.ascii_art)
            paths.append(instance.converter.base64)
            paths.append(instance.converter.leetspeak)
            paths.append(instance.converter.rot13)

        if subdirectory is None or subdirectory == "dataset":
            # Add dataset paths
            paths.append(instance.dataset.harm_bench)

        if subdirectory is None:
            # Add scenario paths
            paths.append(instance.scenario.foundry)

        if subdirectory is None or subdirectory == "initialization.defaults":
            # Add initialization defaults paths
            paths.append(instance.initialization.defaults.converter_initialization)
            paths.append(instance.initialization.defaults.scorer_initialization)
            paths.append(instance.initialization.defaults.target_initialization)

        return paths


# Create a singleton instance for easy access
ConfigurationPaths = _ConfigurationPaths()
