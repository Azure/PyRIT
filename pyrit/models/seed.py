# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Backward compatibility module - imports from pyrit.models.seeds.seed

.. deprecated::
    Import from pyrit.models.seeds instead: `from pyrit.models.seeds import Seed`
"""

import warnings

# Re-export from new location
from pyrit.models.seeds.seed import Seed, PartialUndefined, T

warnings.warn(
    "Importing from pyrit.models.seed is deprecated and will be removed in 0.13.0."
    "Use 'from pyrit.models.seeds import Seed' or 'from pyrit.models import Seed' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Seed", "PartialUndefined", "T"]
