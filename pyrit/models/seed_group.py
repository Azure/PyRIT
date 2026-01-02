# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Backward compatibility module - imports from pyrit.models.seeds.seed_group

.. deprecated::
    Import from pyrit.models.seeds instead: `from pyrit.models.seeds import SeedGroup, SeedAttackGroup`
"""

import warnings

# Re-export from new location
from pyrit.models.seeds.seed_group import SeedGroup, SeedAttackGroup

warnings.warn(
    "Importing from pyrit.models.seed_group is deprecated and will be removed in 0.13.0."
    "Use 'from pyrit.models.seeds import SeedGroup, SeedAttackGroup' or 'from pyrit.models import SeedGroup' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SeedGroup", "SeedAttackGroup"]
