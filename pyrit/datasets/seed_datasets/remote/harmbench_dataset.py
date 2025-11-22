# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from typing import Literal, Optional

from pyrit.datasets.seed_datasets.remote.remote_dataset_loader import RemoteDatasetLoader
from pyrit.models import SeedDataset, SeedPrompt


class HarmBenchDataset(RemoteDatasetLoader):
    """
    Loader for the HarmBench dataset.

    HarmBench is a standardized evaluation framework for automated red teaming.
    It consists of a dataset of harmful behaviors across multiple categories.

    Reference: https://github.com/centerforaisafety/HarmBench
    """

    def __init__(
        self,
        *,
        source: str = (
            "https://raw.githubusercontent.com/centerforaisafety/HarmBench/c0423b9/data/behavior_datasets/"
            "harmbench_behaviors_text_all.csv"
        ),
        source_type: Literal["public_url", "file"] = "public_url",
        cache: bool = True,
        data_home: Optional[Path] = None,
    ):
        """
        Initialize the HarmBench dataset loader.

        Args:
            source: URL to the HarmBench CSV file. Defaults to the official repository.
            source_type: The type of source ('public_url' or 'file').
            cache: Whether to cache the fetched examples. Defaults to True.
            data_home: Directory to store cached data. Defaults to None.
        """
        self.source = source
        self.source_type: Literal["public_url", "file"] = source_type
        self.cache = cache
        self.data_home = data_home

    @property
    def dataset_name(self) -> str:
        return "harmbench"

    async def fetch_dataset(self) -> SeedDataset:
        """
        Fetch HarmBench dataset and return as SeedDataset.

        Returns:
            SeedDataset: A SeedDataset containing the HarmBench examples.

        Raises:
            ValueError: If any example is missing required keys.
        """
        # Required keys to validate each example
        required_keys = {"Behavior", "SemanticCategory"}

        # Fetch the examples using the inherited method
        examples = self._fetch_from_url(
            source=self.source,
            source_type=self.source_type,
            cache=self.cache,
            data_home=self.data_home,
        )

        # Validate and process examples
        seed_prompts = []
        for example in examples:
            # Check for missing keys in the example
            missing_keys = required_keys - example.keys()
            if missing_keys:
                raise ValueError(f"Missing keys in example: {', '.join(missing_keys)}")

            # Extract data
            category = example["SemanticCategory"]

            # Create SeedPrompt
            seed_prompt = SeedPrompt(
                value=example["Behavior"],
                data_type="text",
                name="HarmBench Examples",
                dataset_name=self.dataset_name,
                harm_categories=[category],
                description=(
                    "A dataset of HarmBench examples containing various categories such as chemical, "
                    "biological, illegal activities, etc."
                ),
                source="https://github.com/centerforaisafety/HarmBench",
                authors=["Mantas Mazeika", "Long Phan", "Xuwang Yin", "Andy Zou", "Zifan Wang", "Norman Mu"],
            )
            seed_prompts.append(seed_prompt)

        # Create and return SeedDataset
        return SeedDataset(seeds=seed_prompts)
