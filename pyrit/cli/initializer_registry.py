# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

"""
Initializer registry for discovering and cataloging PyRIT initializers.

This module provides functionality to discover all available PyRITInitializer subclasses.

PERFORMANCE OPTIMIZATION:
This module uses lazy imports and direct path computation to minimize import overhead:

1. Lazy Imports via TYPE_CHECKING: PyRITInitializer is only imported for type checking,
   not at runtime. Runtime imports happen inside methods when actually needed.

2. Direct Path Computation: Computes PYRIT_PATH directly using __file__ instead of importing
   from pyrit.common.path, avoiding loading of the pyrit package.
"""

import importlib.util
import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, TypedDict

# Compute PYRIT_PATH directly to avoid importing pyrit package
# (which triggers heavy imports from __init__.py)
PYRIT_PATH = Path(__file__).parent.parent.resolve()

if TYPE_CHECKING:
    from pyrit.setup.initializers.pyrit_initializer import PyRITInitializer

logger = logging.getLogger(__name__)


class InitializerInfo(TypedDict):
    """Type definition for initializer information dictionary."""

    name: str
    class_name: str
    initializer_name: str
    description: str
    required_env_vars: list[str]
    execution_order: int


class InitializerRegistry:
    """
    Registry for discovering and managing available initializers.

    This class discovers all PyRITInitializer subclasses from the
    pyrit/setup/initializers directory structure.

    Initializers are identified by their filename (e.g., "objective_target", "simple").
    The directory structure is used for organization but not exposed to users.
    """

    def __init__(self, *, discovery_path: Path | None = None) -> None:
        """
        Initialize the initializer registry.

        Args:
            discovery_path (Path | None): The path to discover initializers from.
                If None, defaults to pyrit/setup/initializers (discovers all).
                To discover only scenarios, pass pyrit/setup/initializers/scenarios.
        """
        self._initializers: Dict[str, InitializerInfo] = {}
        self._initializer_paths: Dict[str, Path] = {}  # Track file paths for collision detection
        self._initializer_metadata: Optional[List[InitializerInfo]] = None

        if discovery_path is None:
            discovery_path = Path(PYRIT_PATH) / "setup" / "initializers"

        self._discovery_path = discovery_path

        self._discover_initializers()

    def _discover_initializers(self) -> None:
        """
        Discover all initializers from the specified discovery path.

        This method recursively walks the directory tree and registers
        any PyRITInitializer subclasses found. Initializers are registered
        by filename only for simpler user experience.
        """
        if not self._discovery_path.exists():
            logger.warning(f"Initializers directory not found: {self._discovery_path}")
            return

        # Check if discovery path is a file or directory
        if self._discovery_path.is_file():
            self._process_file(file_path=self._discovery_path)
        elif self._discovery_path.is_dir():
            # Discover from the specified directory and its subdirectories
            self._discover_in_directory(directory=self._discovery_path)

    def _discover_in_directory(self, *, directory: Path) -> None:
        """
        Recursively discover PyRIT initializers in a directory.

        Args:
            directory (Path): The directory to search for initializer modules.
        """
        for item in directory.iterdir():
            if item.is_file() and item.suffix == ".py" and item.stem != "__init__":
                self._process_file(file_path=item)
            elif item.is_dir() and item.name != "__pycache__":
                self._discover_in_directory(directory=item)

    def _process_file(self, *, file_path: Path) -> None:
        """
        Process a Python file to extract PyRITInitializer subclasses.

        Args:
            file_path (Path): Path to the Python file to process.
        """
        # Runtime import to avoid loading heavy modules at module level
        from pyrit.setup.initializers.pyrit_initializer import PyRITInitializer

        # Calculate module name for import (still needs full path for Python import)
        # Convert file path to module path relative to initializers directory
        initializers_base = Path(PYRIT_PATH) / "setup" / "initializers"
        relative_path = file_path.relative_to(initializers_base)
        module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
        module_name = ".".join(module_parts)

        # Use just the filename as the name (e.g., "load_default_datasets")
        short_name = file_path.stem

        # Check for name collision
        if short_name in self._initializer_paths:
            existing_path = self._initializer_paths[short_name]
            logger.error(
                f"Initializer name collision: '{short_name}' found in both "
                f"'{file_path}' and '{existing_path}'. "
                f"Initializer filenames must be unique across all directories."
            )
            return

        try:
            spec = importlib.util.spec_from_file_location(f"pyrit.setup.initializers.{module_name}", file_path)
            if not spec or not spec.loader:
                return

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find all PyRITInitializer subclasses in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if inspect.isclass(attr) and issubclass(attr, PyRITInitializer) and attr != PyRITInitializer:
                    self._try_register_initializer(initializer_class=attr, short_name=short_name, file_path=file_path)

        except Exception as e:
            logger.warning(f"Failed to load initializer module {short_name}: {e}")

    def _try_register_initializer(
        self, *, initializer_class: type[PyRITInitializer], short_name: str, file_path: Path
    ) -> None:
        """
        Try to instantiate an initializer and register it.

        Args:
            initializer_class (type[PyRITInitializer]): The initializer class to instantiate.
            short_name (str): The short name for the initializer (filename without extension).
            file_path (Path): The path to the file containing the initializer.
        """
        try:
            instance = initializer_class()
            initializer_info: InitializerInfo = {
                "name": short_name,
                "class_name": initializer_class.__name__,
                "initializer_name": instance.name,
                "description": instance.description,
                "required_env_vars": instance.required_env_vars,
                "execution_order": instance.execution_order,
            }
            self._initializers[short_name] = initializer_info
            self._initializer_paths[short_name] = file_path
            logger.debug(f"Registered initializer: {short_name} ({initializer_class.__name__})")

        except Exception as e:
            logger.warning(f"Failed to instantiate initializer {initializer_class.__name__}: {e}")

    def get_initializer(self, name: str) -> InitializerInfo | None:
        """
        Get an initializer by name.

        Args:
            name (str): Initializer identifier (e.g., "objective_target", "simple")

        Returns:
            InitializerInfo | None: The initializer information, or None if not found.
        """
        return self._initializers.get(name)

    def list_initializers(self) -> List[InitializerInfo]:
        """
        List all available initializers with their metadata.

        Returns:
            List[InitializerInfo]: List of initializer information dictionaries, sorted by
                execution order and then by name.
        """
        # Return cached metadata if available
        if self._initializer_metadata is not None:
            return self._initializer_metadata

        # Build from discovered initializers
        initializers_list = list(self._initializers.values())
        initializers_list.sort(key=lambda x: (x["execution_order"], x["name"]))

        # Cache for subsequent calls
        self._initializer_metadata = initializers_list

        return initializers_list

    def get_initializer_names(self) -> List[str]:
        """
        Get a list of all available initializer names.

        Returns:
            List[str]: Sorted list of initializer identifiers.
        """
        return sorted(self._initializers.keys())

    def resolve_initializer_paths(self, *, initializer_names: list[str]) -> list[Path]:
        """
        Resolve initializer names to their file paths.

        Args:
            initializer_names (list[str]): List of initializer names to resolve.

        Returns:
            list[Path]: List of resolved file paths.

        Raises:
            ValueError: If any initializer name is not found or has no file path.
        """
        resolved_paths = []

        for initializer_name in initializer_names:
            initializer_info = self.get_initializer(initializer_name)

            if initializer_info is None:
                available = ", ".join(sorted(self.get_initializer_names()))
                raise ValueError(
                    f"Built-in initializer '{initializer_name}' not found.\n"
                    f"Available initializers: {available}\n"
                    f"Use 'pyrit_scan --list-initializers' to see detailed information."
                )

            initializer_file = self._initializer_paths.get(initializer_name)
            if initializer_file is None:
                raise ValueError(f"Could not locate file for initializer '{initializer_name}'.")

            resolved_paths.append(initializer_file)

        return resolved_paths

    def get_initializer_class(self, *, name: str) -> type["PyRITInitializer"]:
        """
        Get the initializer class by name.

        Args:
            name: The initializer name.

        Returns:
            The initializer class.

        Raises:
            ValueError: If initializer not found.
        """
        import importlib.util

        initializer_info = self.get_initializer(name)
        if initializer_info is None:
            available = ", ".join(sorted(self.get_initializer_names()))
            raise ValueError(
                f"Initializer '{name}' not found.\n"
                f"Available initializers: {available}\n"
                f"Use 'pyrit_scan --list-initializers' to see detailed information."
            )

        initializer_file = self._initializer_paths.get(name)
        if initializer_file is None:
            raise ValueError(f"Could not locate file for initializer '{name}'.")

        # Load the module
        spec = importlib.util.spec_from_file_location("initializer_module", initializer_file)
        if spec is None or spec.loader is None:
            raise ValueError(f"Failed to load initializer from {initializer_file}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the initializer class
        initializer_class: type[PyRITInitializer] = getattr(module, initializer_info["class_name"])
        return initializer_class

    @staticmethod
    def resolve_script_paths(*, script_paths: list[str]) -> list[Path]:
        """
        Resolve and validate custom script paths.

        Args:
            script_paths (list[str]): List of script path strings to resolve.

        Returns:
            list[Path]: List of resolved Path objects.

        Raises:
            FileNotFoundError: If any script path does not exist.
        """
        resolved_paths = []

        for script in script_paths:
            script_path = Path(script)
            if not script_path.is_absolute():
                script_path = Path.cwd() / script_path

            if not script_path.exists():
                raise FileNotFoundError(
                    f"Initialization script not found: {script_path}\n  Looked in: {script_path.absolute()}"
                )

            resolved_paths.append(script_path)

        return resolved_paths
