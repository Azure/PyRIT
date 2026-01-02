# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Backward compatibility module - imports from pyrit.models.seeds.seed_prompt

.. deprecated::
    Import from pyrit.models.seeds instead: `from pyrit.models.seeds import SeedPrompt`
"""

import warnings

# Re-export from new location
from pyrit.models.seeds.seed_prompt import SeedPrompt

warnings.warn(
    "Importing from pyrit.models.seed_prompt is deprecated and will be removed in  0.13.0. "
    "Use 'from pyrit.models.seeds import SeedPrompt' or 'from pyrit.models import SeedPrompt' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SeedPrompt"]
