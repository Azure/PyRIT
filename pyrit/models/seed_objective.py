# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Backward compatibility module - imports from pyrit.models.seeds.seed_objective

.. deprecated::
    Import from pyrit.models.seeds instead: `from pyrit.models.seeds import SeedObjective`
"""

import warnings

# Re-export from new location
from pyrit.models.seeds.seed_objective import SeedObjective

warnings.warn(
    "Importing from pyrit.models.seed_objective is deprecated and will be removed in 0.13.0. "
    "Use 'from pyrit.models.seeds import SeedObjective' or 'from pyrit.models import SeedObjective' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SeedObjective"]
