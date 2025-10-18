# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""This module contains initialization PyRIT."""

from pyrit.setup.initialization_paths import InitializationPaths, initialization_paths
from pyrit.setup.initialization import (
    initialize_pyrit,
    AZURE_SQL,
    SQLITE,
    IN_MEMORY,
)
# Import from common for backward compatibility
from pyrit.common.apply_defaults import (
    apply_defaults,
    apply_defaults_to_method,
    set_default_value,
    reset_default_values,
    get_global_default_values,
    set_global_variable,
)


__all__ = [
    "AZURE_SQL",
    "SQLITE",
    "IN_MEMORY",
    "InitializationPaths",
    "initialization_paths",
    "initialize_pyrit",
    "apply_defaults",
    "apply_defaults_to_method",
    "set_default_value",
    "set_global_variable",
    "get_global_default_values",
    "reset_default_values",
]
