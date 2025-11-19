# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from pyrit.models import Seed, SeedDataset, SeedGroup, SeedObjective
from typing import final, Union

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = [
    
]

@final
class ScenarioDatasetLoader:
    """
    The ScenarioDatasetLoader is a lazy and stateful dataset handling component for Scenarios.
    It contains several methods for data wrangling, parsing, storage, and retrieval to keep them
    out of Scenario. Note that this class does NOT contain the dataset itself; that is Scenario's
    responsibility.
    """
    def __init__(self, dataset: Union[Path, str]) -> None:
        """
        Inintialize the ScenarioDatasetLoader.
        
        Args:
            dataset (Union[Path, str]): Dataset reference.
        """
        self._path = self._resolve_and_validate_path(dataset)
        self._loaded = False

    def load(self) -> SeedDataset:
        """
        Called to pass a SeedDataset to Scenario, which owns it for its entire lifecycle. This uses a functional
        pattern where the SeedDataset is ephemeral, but the DatasetLoader is not.
        
        This is called in Scenario.__init__ if lazy loading is enabled and in Scenario.initialize_async otherwise.
        """
        # TODO: Add logging hierarchy with namespace discovery at runtime and standardized format
        # Format like (timestamp) (pyrit@<host>) [pyrit.component] message
        # (01-01-2000;00:00:00.0000) (pyrit@devcontainer) [pyrit.scenarios.dataset.scenario_dataset_loader.load]
        #   loading dataset from datasets/seed_prompts...
        
        # To avoid filling scenario with more stateful attributes, we use a flag here to see if the dataset has
        # already been loaded. This is the signal that informs the lazy loading check in initialize_async.
        self._loaded = True
        logging.info(f"Loading dataset from {self._path}")
        
        return SeedDataset.from_yaml_file(self._path)

        
    def _resolve_and_validate_path(self, path: Union[Path, str]) -> Path:
        """
        Resolve and validate path provided to the dataset loader.
        """
        try:
            if isinstance(path, Path):
                path = path.resolve()
            elif isinstance(path, str):
                path = Path(path).resolve()
        except:
            raise FileNotFoundError(f"Error: path provided at {path} not found!")
        return path
    