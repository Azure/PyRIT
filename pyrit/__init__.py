# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .common import turn_off_transformers_warning  # noqa: F401
from .show_versions import show_versions  # noqa: F401

__name__ = "pyrit"
# Remove dev suffix when releasing and keep in sync with pyproject.toml
__version__ = "0.11.0"
