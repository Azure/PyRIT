# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Provides convenient access to initialization script paths.

This module defines a path structure for accessing PyRIT's
initialization scripts that can be used with initialize_pyrit().

Example:
    from pyrit.setup import initialization_paths, initialize_pyrit

    # Use an initialization script
    initialize_pyrit(
        memory_db_type="InMemory",
        initialization_scripts=[initialization_paths.converter_initialization]
    )
"""

import pathlib
from typing import List


class InitializationPaths:
    """
    Provides access to initialization script paths.

    This class provides direct access to PyRIT's initialization scripts.

    Attributes:
        converter_initialization: Path to the converter initialization script.
        scorer_initialization: Path to the scorer initialization script.
        target_initialization: Path to the target initialization script.

    Example:
        # Access an initialization script
        path = InitializationPaths.converter_initialization

        # Use with initialize_pyrit
        initialize_pyrit(
            memory_db_type="InMemory",
            initialization_scripts=[InitializationPaths.scorer_initialization]
        )

        # List all paths
        all_paths = InitializationPaths.list_all_paths()
    """

    def __init__(self) -> None:
        self._CONFIG_PATH = pathlib.Path(__file__).parent / "config"

    @property
    def converter_initialization(self) -> pathlib.Path:
        """Path to the converter initialization script."""
        return self._CONFIG_PATH / "converter_initialization.py"

    @property
    def scorer_initialization(self) -> pathlib.Path:
        """Path to the scorer initialization script."""
        return self._CONFIG_PATH / "scorer_initialization.py"

    @property
    def target_initialization(self) -> pathlib.Path:
        """Path to the target initialization script."""
        return self._CONFIG_PATH / "target_initialization.py"

    @classmethod
    def list_all_paths(cls) -> List[pathlib.Path]:
        """
        Get a list of all available configuration script paths.

        Returns:
            List[pathlib.Path]: List of configuration script paths.

        Example:
            # Get all paths
            all_paths = InitializationPaths.list_all_paths()
        """
        instance = cls()
        paths: List[pathlib.Path] = []

        paths.append(instance.converter_initialization)
        paths.append(instance.scorer_initialization)
        paths.append(instance.target_initialization)

        return paths


# Create a singleton instance for easy access
initialization_paths = InitializationPaths()
