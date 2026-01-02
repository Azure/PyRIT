# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Backward compatibility shim for SeedDataset.

.. deprecated::
    Import from `pyrit.models.seeds` instead. This module will be removed in a future version.
"""

import warnings

from pyrit.models.seeds.seed_dataset import SeedDataset

# Issue deprecation warning on import
warnings.warn(
    "Importing SeedDataset from pyrit.models.seed_dataset is deprecated and will be removed in 0.13.0. "
    "Import from pyrit.models.seeds instead: `from pyrit.models.seeds import SeedDataset`",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SeedDataset"]

