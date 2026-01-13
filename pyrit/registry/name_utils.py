# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Name conversion utilities for PyRIT registries.

This module provides functions for converting between different naming conventions
used in registries (e.g., CamelCase class names to snake_case registry names).
"""

from __future__ import annotations

import re


def class_name_to_registry_name(class_name: str, *, suffix: str = "") -> str:
    """
    Convert a CamelCase class name to a snake_case registry name.

    Args:
        class_name: The class name to convert (e.g., "MyCustomScenario").
        suffix: Optional suffix to strip from the class name before conversion
            (e.g., "Scenario" would convert "MyCustomScenario" to "my_custom").

    Returns:
        The snake_case registry name (e.g., "my_custom_scenario" or "my_custom").
    """
    # Remove suffix if present
    if suffix and class_name.endswith(suffix):
        class_name = class_name[: -len(suffix)]

    # Convert CamelCase to snake_case
    # First, handle transitions like "XMLParser" -> "XML_Parser"
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", class_name)
    # Then handle transitions like "getHTTPResponse" -> "get_HTTP_Response"
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

    return name


def registry_name_to_class_name(registry_name: str, *, suffix: str = "") -> str:
    """
    Convert a snake_case registry name to a PascalCase class name.

    Args:
        registry_name: The registry name to convert (e.g., "my_custom").
        suffix: Optional suffix to append to the class name
            (e.g., "Scenario" would convert "my_custom" to "MyCustomScenario").

    Returns:
        The PascalCase class name (e.g., "MyCustomScenario").
    """
    # Split on underscores and capitalize each part
    parts = registry_name.split("_")
    pascal_case = "".join(part.capitalize() for part in parts)

    # Append suffix if provided
    if suffix:
        pascal_case += suffix

    return pascal_case
