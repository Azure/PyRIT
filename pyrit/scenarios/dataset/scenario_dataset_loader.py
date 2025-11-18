# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from pyrit.models import Seed, SeedDataset, SeedGroup, SeedObjective
from typing import final

logger = logging.getLogger(__name__)

@final
class ScenarioDatasetLoader:
    """
    The ScenarioDatasetLoader is a lazy and stateful dataset handling component for Scenarios.
    It contains several methods for data wrangling, parsing, storage, and retrieval to keep them
    out of Scenario.
    """
    def __init__(self, lazy_loading: bool = True) -> None:
        """
        Inintialize the ScenarioDatasetLoader.
        
        Args:
            lazy_loading (bool): Whether or not to only fetch dataset contents in Scenario.initialize.async.
        """
        self.lazy_loading = lazy_loading