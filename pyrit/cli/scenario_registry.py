# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

"""
Scenario registry for discovering and instantiating PyRIT scenarios.

This module provides functionality to discover all available Scenario subclasses
from the pyrit.scenario.scenarios module and from user-defined initialization scripts.

PERFORMANCE OPTIMIZATION:
This module uses lazy imports to minimize overhead during CLI operations:

1. Lazy Imports via TYPE_CHECKING: Heavy dependencies (like Scenario base class) are only
   imported for type checking, not at runtime. Runtime imports happen inside methods only
   when actually needed.

2. Direct Path Computation: Computes PYRIT_PATH directly using __file__ instead of importing
   from pyrit.common.path, which would trigger loading the entire pyrit package (including
   heavy dependencies like transformers).
"""

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Type

# Compute PYRIT_PATH directly to avoid importing pyrit package
# (which triggers heavy imports from __init__.py)
PYRIT_PATH = Path(__file__).parent.parent.resolve()

# Lazy import to avoid loading heavy scenario modules when just listing scenarios
if TYPE_CHECKING:
    from pyrit.cli.frontend_core import ScenarioInfo
    from pyrit.scenario.core import Scenario

logger = logging.getLogger(__name__)


class ScenarioRegistry:
    """
    Registry for discovering and managing available scenarios.

    This class discovers all Scenario subclasses from:
    1. Built-in scenarios in pyrit.scenario.scenarios module
    2. User-defined scenarios from initialization scripts (set via globals)

    Scenarios are identified by their simple name (e.g., "encoding_scenario", "foundry_scenario").
    """

    def __init__(self) -> None:
        """Initialize the scenario registry with lazy discovery."""
        self._scenarios: Dict[str, Type[Scenario]] = {}
        self._scenario_metadata: Optional[List[ScenarioInfo]] = None
        self._discovered = False

    def _ensure_discovered(self) -> None:
        """Ensure scenarios have been discovered. Discovers on first call only."""
        if not self._discovered:
            self._discover_builtin_scenarios()
            self._discovered = True

    def _discover_builtin_scenarios(self) -> None:
        """
        Discover all built-in scenarios from pyrit.scenario.scenarios module.

        This method dynamically imports all modules in the scenarios package
        and registers any Scenario subclasses found.
        """
        from pyrit.scenario.core import Scenario

        try:
            import pyrit.scenario.scenarios as scenarios_package

            # Get the path to the scenarios package
            package_file = scenarios_package.__file__
            if package_file is None:
                # Try using __path__ instead
                if hasattr(scenarios_package, "__path__"):
                    package_path = Path(scenarios_package.__path__[0])
                else:
                    logger.error("Cannot determine scenarios package location")
                    return
            else:
                package_path = Path(package_file).parent

            # Iterate through all Python files in the scenarios directory
            for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
                if module_name.startswith("_"):
                    continue

                try:
                    # Import the module
                    full_module_name = f"pyrit.scenario.scenarios.{module_name}"
                    module = importlib.import_module(full_module_name)

                    # Find all Scenario subclasses in the module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        # Check if it's a Scenario subclass (but not Scenario itself)
                        if issubclass(obj, Scenario) and obj is not Scenario:
                            # Use the module name as the scenario identifier
                            scenario_name = module_name
                            self._scenarios[scenario_name] = obj
                            logger.debug(f"Registered built-in scenario: {scenario_name} ({obj.__name__})")

                except Exception as e:
                    logger.warning(f"Failed to load scenario module {module_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to discover built-in scenarios: {e}")

    def discover_user_scenarios(self) -> None:
        """
        Discover user-defined scenarios from global variables.

        After initialization scripts are executed, they may define Scenario subclasses
        and store them in globals. This method searches for such classes.

        User scenarios will override built-in scenarios with the same name.
        """
        from pyrit.scenario.core import Scenario

        try:
            # Check the global namespace for Scenario subclasses
            import sys

            # Create a snapshot of modules to avoid dictionary changed size during iteration
            modules_snapshot = list(sys.modules.items())

            # Look through all loaded modules for scenario classes
            for module_name, module in modules_snapshot:
                if module is None or not hasattr(module, "__dict__"):
                    continue

                # Skip built-in and standard library modules
                if module_name.startswith(("builtins", "_", "sys", "os", "importlib")):
                    continue

                # Look for Scenario subclasses in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, Scenario) and obj is not Scenario:
                        # Check if this is a user-defined class (not from pyrit.scenario.scenarios)
                        if not obj.__module__.startswith("pyrit.scenario.scenarios"):
                            # Convert class name to snake_case for scenario name
                            scenario_name = self._class_name_to_scenario_name(obj.__name__)
                            self._scenarios[scenario_name] = obj
                            logger.info(f"Registered user-defined scenario: {scenario_name} ({obj.__name__})")

        except Exception as e:
            # Silently ignore errors during user scenario discovery
            # User scenarios are optional and errors here are not critical
            logger.debug(f"Failed to discover user scenarios: {e}")

    def _class_name_to_scenario_name(self, class_name: str) -> str:
        """
        Convert a class name to a scenario identifier.

        Args:
            class_name (str): Class name (e.g., "EncodingScenario", "MyCustomScenario")

        Returns:
            str: Scenario identifier (e.g., "encoding_scenario", "my_custom_scenario")
        """
        # Remove "Scenario" suffix if present
        if class_name.endswith("Scenario"):
            class_name = class_name[:-8]

        # Convert CamelCase to snake_case
        import re

        name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", class_name)
        name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

        return name

    def get_scenario(self, name: str) -> Optional[Type[Scenario]]:
        """
        Get a scenario class by name.

        Args:
            name (str): Scenario identifier (e.g., "encoding_scenario", "foundry_scenario")

        Returns:
            Optional[Type[Scenario]]: The scenario class, or None if not found.
        """
        self._ensure_discovered()
        return self._scenarios.get(name)

    def list_scenarios(self) -> List[ScenarioInfo]:
        """
        List all available scenarios with their metadata.

        Returns:
            List[ScenarioInfo]: List of scenario information dictionaries containing:
                - name: Scenario identifier
                - class_name: Class name
                - description: Full class docstring
                - default_strategy: The default strategy used when none specified
                - all_strategies: All available strategy values
                - aggregate_strategies: Aggregate strategy values
        """
        # If we already have metadata, return it
        if self._scenario_metadata is not None:
            return self._scenario_metadata

        # Discover scenarios and build metadata
        self._ensure_discovered()
        scenarios_info: List[ScenarioInfo] = []

        for name, scenario_class in sorted(self._scenarios.items()):
            # Extract full docstring as description, clean up whitespace
            doc = scenario_class.__doc__ or ""
            if doc:
                # Normalize whitespace: remove leading/trailing, collapse multiple spaces/newlines
                description = " ".join(doc.split())
            else:
                description = "No description available"

            # Get the strategy class for this scenario
            strategy_class = scenario_class.get_strategy_class()

            scenarios_info.append(
                {
                    "name": name,
                    "class_name": scenario_class.__name__,
                    "description": description,
                    "default_strategy": scenario_class.get_default_strategy().value,
                    "all_strategies": [s.value for s in strategy_class.get_all_strategies()],
                    "aggregate_strategies": [s.value for s in strategy_class.get_aggregate_strategies()],
                }
            )

        # Cache metadata for subsequent calls
        self._scenario_metadata = scenarios_info

        return scenarios_info

    def get_scenario_names(self) -> List[str]:
        """
        Get a list of all available scenario names.

        Returns:
            List[str]: Sorted list of scenario identifiers.
        """
        self._ensure_discovered()
        return sorted(self._scenarios.keys())
