# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from typing import List
from pyrit.models import SeedDataset
from pyrit.common.path import DATASETS_PATH, SCORER_CONFIG_PATH
from pyrit.datasets.harmbench_dataset import fetch_harmbench_dataset


class ScenarioDatasetUtils:
    """
    Set of dataset loading utilities for Scenario class.
    """
    @classmethod
    def seed_dataset_to_list_str(cls, dataset: Path) -> List[str]:
        seed_prompts: List[str] = []
        seed_prompts.extend(SeedDataset.from_yaml_file(dataset).get_values())
        return seed_prompts

    @classmethod
    def get_seed_dataset(cls, which: str) -> SeedDataset:
        """
        Get SeedDataset from shorthand string.
        Args:
            which (str): Which SeedDataset.
        Returns:
            SeedDataset: Desired dataset.
        Raises:
            ValueError: If dataset not found.
        """
        match which:
            case "harmbench":
                return fetch_harmbench_dataset()
            case _:
                raise ValueError(f"Error: unknown dataset `{which}` provided.")