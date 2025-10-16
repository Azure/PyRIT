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

    This class provides direct access to PyRIT's initialization scripts,
    organized into 'airt' (AI Red Team) and 'simple' categories.

    Attributes:
        airt_converter_initialization: Path to the AIRT converter initialization script.
        airt_scorer_initialization: Path to the AIRT scorer initialization script.
        airt_target_initialization: Path to the AIRT target initialization script.
        simple_converter_initialization: Path to the simple converter initialization script.
        simple_scorer_initialization: Path to the simple scorer initialization script.
        simple_target_initialization: Path to the simple target initialization script.

    Example:
        # Access an initialization script
        path = InitializationPaths.airt_scorer_initialization

        # Use with initialize_pyrit
        initialize_pyrit(
            memory_db_type="InMemory",
            initialization_scripts=[InitializationPaths.airt_scorer_initialization]
        )

        # List all AIRT paths
        airt_paths = InitializationPaths.list_all_airt_paths()

        # List all simple paths
        simple_paths = InitializationPaths.list_all_simple_paths()
    """

    def __init__(self) -> None:
        self._CONFIG_PATH = pathlib.Path(__file__).parent / "config"

    @property
    def airt_converter_initialization(self) -> pathlib.Path:
        """Path to the AIRT converter initialization script."""
        return self._CONFIG_PATH / "airt" / "converter_initialization.py"

    @property
    def airt_scorer_initialization(self) -> pathlib.Path:
        """Path to the AIRT scorer initialization script."""
        return self._CONFIG_PATH / "airt" / "scorer_initialization.py"

    @property
    def airt_target_initialization(self) -> pathlib.Path:
        """Path to the AIRT target initialization script."""
        return self._CONFIG_PATH / "airt" / "target_initialization.py"

    @property
    def simple_converter_initialization(self) -> pathlib.Path:
        """Path to the simple converter initialization script."""
        return self._CONFIG_PATH / "simple" / "converter_initialization.py"

    @property
    def simple_scorer_initialization(self) -> pathlib.Path:
        """Path to the simple scorer initialization script."""
        return self._CONFIG_PATH / "simple" / "scorer_initialization.py"

    @property
    def simple_target_initialization(self) -> pathlib.Path:
        """Path to the simple target initialization script."""
        return self._CONFIG_PATH / "simple" / "target_initialization.py"

    @classmethod
    def list_all_airt_paths(cls) -> List[pathlib.Path]:
        """
        Get a list of all AIRT configuration script paths.

        Returns:
            List[pathlib.Path]: List of AIRT configuration script paths.

        Example:
            # Get all AIRT paths
            airt_paths = InitializationPaths.list_all_airt_paths()
        """
        instance = cls()
        paths: List[pathlib.Path] = []

        paths.append(instance.airt_converter_initialization)
        paths.append(instance.airt_scorer_initialization)
        paths.append(instance.airt_target_initialization)

        return paths

    @classmethod
    def list_all_simple_paths(cls) -> List[pathlib.Path]:
        """
        Get a list of all simple configuration script paths.

        Returns:
            List[pathlib.Path]: List of simple configuration script paths.

        Example:
            # Get all simple paths
            simple_paths = InitializationPaths.list_all_simple_paths()
        """
        instance = cls()
        paths: List[pathlib.Path] = []

        paths.append(instance.simple_converter_initialization)
        paths.append(instance.simple_scorer_initialization)
        paths.append(instance.simple_target_initialization)

        return paths


# Create a singleton instance for easy access
initialization_paths = InitializationPaths()
