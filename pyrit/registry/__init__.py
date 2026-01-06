# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from pyrit.registry.base import RegistryItemMetadata, RegistryProtocol

from pyrit.registry.class_registries import (
    BaseClassRegistry,
    ClassEntry,
    InitializerMetadata,
    InitializerRegistry,
    ScenarioMetadata,
    ScenarioRegistry,
)

from pyrit.registry.instance_registries import (
    BaseInstanceRegistry,
    ScorerMetadata,
    ScorerRegistry,
)

from pyrit.registry.discovery import (
    discover_in_directory,
    discover_in_package,
    discover_subclasses_in_loaded_modules,
)
from pyrit.registry.name_utils import class_name_to_registry_name, registry_name_to_class_name


__all__ = [
    # Base classes and protocols
    "BaseClassRegistry",
    "ClassEntry",
    "RegistryItemMetadata",
    "RegistryProtocol",
    # Instance registry base
    "BaseInstanceRegistry",
    # Name utilities
    "class_name_to_registry_name",
    "registry_name_to_class_name",
    # Discovery utilities
    "discover_in_directory",
    "discover_in_package",
    "discover_subclasses_in_loaded_modules",
    # Concrete registries and their metadata types
    "ScenarioRegistry",
    "ScenarioMetadata",
    "InitializerRegistry",
    "InitializerMetadata",
    "ScorerRegistry",
    "ScorerMetadata",
]
