# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Registry module for PyRIT class and instance registries."""

from pyrit.registry.base import RegistryItemMetadata, RegistryProtocol
from pyrit.registry.class_registries import (
    BaseClassRegistry,
    ClassEntry,
    InitializerMetadata,
    InitializerRegistry,
    ScenarioMetadata,
    ScenarioRegistry,
)
from pyrit.registry.discovery import (
    discover_in_directory,
    discover_in_package,
    discover_subclasses_in_loaded_modules,
)
from pyrit.registry.instance_registries import (
    BaseInstanceRegistry,
    ScorerMetadata,
    ScorerRegistry,
    TargetMetadata,
    TargetRegistry,
)
from pyrit.registry.name_utils import class_name_to_registry_name, registry_name_to_class_name

__all__ = [
    "BaseClassRegistry",
    "BaseInstanceRegistry",
    "ClassEntry",
    "class_name_to_registry_name",
    "discover_in_directory",
    "discover_in_package",
    "discover_subclasses_in_loaded_modules",
    "InitializerMetadata",
    "InitializerRegistry",
    "RegistryItemMetadata",
    "RegistryProtocol",
    "registry_name_to_class_name",
    "ScenarioMetadata",
    "ScenarioRegistry",
    "ScorerMetadata",
    "ScorerRegistry",
    "TargetMetadata",
    "TargetRegistry",
]
