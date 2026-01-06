# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Scenario registry for discovering and managing PyRIT scenarios.

This module provides a unified registry for discovering all available Scenario subclasses
from the pyrit.scenario.scenarios module and from user-defined initialization scripts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from pyrit.registry.base import RegistryItemMetadata
from pyrit.registry.class_registries.base_class_registry import BaseClassRegistry, ClassEntry
from pyrit.registry.discovery import discover_in_package, discover_subclasses_in_loaded_modules
from pyrit.registry.name_utils import class_name_to_registry_name

if TYPE_CHECKING:
    from pyrit.scenario.core import Scenario

logger = logging.getLogger(__name__)


class ScenarioMetadata(RegistryItemMetadata):
    """
    Metadata describing a registered Scenario class.

    This TypedDict provides descriptive information about a Scenario class,
    NOT the class itself. Use get_class() to get the actual class.
    """

    default_strategy: str
    all_strategies: list[str]
    aggregate_strategies: list[str]
    default_datasets: list[str]
    max_dataset_size: Optional[int]


class ScenarioRegistry(BaseClassRegistry["Scenario", ScenarioMetadata]):
    """
    Registry for discovering and managing available scenario classes.

    This class discovers all Scenario subclasses from:
    1. Built-in scenarios in pyrit.scenario.scenarios module
    2. User-defined scenarios from initialization scripts (set via globals)

    Scenarios are identified by their simple name (e.g., "encoding", "foundry").
    """

    @classmethod
    def get_instance(cls) -> "ScenarioRegistry":
        """Get the singleton instance of the ScenarioRegistry."""
        return super().get_instance()  # type: ignore[return-value]

    def __init__(self, *, lazy_discovery: bool = True) -> None:
        """
        Initialize the scenario registry.

        Args:
            lazy_discovery: If True, discovery is deferred until first access.
                Defaults to True for performance.
        """
        super().__init__(lazy_discovery=lazy_discovery)

    def _discover(self) -> None:
        """Discover all built-in scenarios from pyrit.scenario.scenarios module."""
        self._discover_builtin_scenarios()

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
                if hasattr(scenarios_package, "__path__"):
                    package_path = Path(scenarios_package.__path__[0])
                else:
                    logger.error("Cannot determine scenarios package location")
                    return
            else:
                package_path = Path(package_file).parent

            # Discover scenarios using the shared discovery utility
            for module_name, scenario_class in discover_in_package(
                package_path=package_path,
                package_name="pyrit.scenario.scenarios",
                base_class=Scenario,
                recursive=True,
            ):
                entry = ClassEntry(registered_class=scenario_class)
                self._class_entries[module_name] = entry
                logger.debug(f"Registered built-in scenario: {module_name} ({scenario_class.__name__})")

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
            for module_name, scenario_class in discover_subclasses_in_loaded_modules(base_class=Scenario):
                # Check if this is a user-defined class (not from pyrit.scenario.scenarios)
                if not scenario_class.__module__.startswith("pyrit.scenario.scenarios"):
                    # Convert class name to snake_case for scenario name
                    registry_name = class_name_to_registry_name(scenario_class.__name__, suffix="Scenario")
                    entry = ClassEntry(registered_class=scenario_class)
                    self._class_entries[registry_name] = entry
                    logger.info(f"Registered user-defined scenario: {registry_name} ({scenario_class.__name__})")

        except Exception as e:
            logger.debug(f"Failed to discover user scenarios: {e}")

    def _build_metadata(self, name: str, entry: ClassEntry["Scenario"]) -> ScenarioMetadata:
        """
        Build metadata for a Scenario class.

        Args:
            name: The registry name of the scenario.
            entry: The ClassEntry containing the scenario class.

        Returns:
            ScenarioMetadata dictionary describing the scenario class.
        """
        # Get base metadata (name, class_name, description)
        base_metadata = self._build_base_metadata(name, entry)
        scenario_class = entry.registered_class

        # Get the strategy class for this scenario
        strategy_class = scenario_class.get_strategy_class()

        dataset_config = scenario_class.default_dataset_config()
        default_datasets = dataset_config.get_default_dataset_names()
        max_dataset_size = dataset_config.max_dataset_size

        return ScenarioMetadata(
            **base_metadata,
            default_strategy=scenario_class.get_default_strategy().value,
            all_strategies=[s.value for s in strategy_class.get_all_strategies()],
            aggregate_strategies=[s.value for s in strategy_class.get_aggregate_strategies()],
            default_datasets=default_datasets,
            max_dataset_size=max_dataset_size,
        )
