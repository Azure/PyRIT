# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from pathlib import Path
from typing import Optional

from pyrit.datasets.seed_datasets.remote.remote_dataset_loader import RemoteDatasetLoader
from pyrit.models import SeedDataset, SeedPrompt

logger = logging.getLogger(__name__)


class LLMLatentAdversarialTrainingDataset(RemoteDatasetLoader):
    """
    Loader for the LLM-LAT harmful dataset.

    This dataset contains prompts used to assess and analyze harmful behaviors
    in large language models.

    Reference: https://huggingface.co/datasets/LLM-LAT/harmful-dataset
    """

    def __init__(
        self,
        *,
        source: str = "LLM-LAT/harmful-dataset",
        cache: bool = True,
        data_home: Optional[Path] = None,
    ):
        """
        Initialize the LLM-LAT harmful dataset loader.

        Args:
            source: HuggingFace dataset identifier. Defaults to "LLM-LAT/harmful-dataset".
            cache: Whether to cache the fetched examples. Defaults to True.
            data_home: Directory to store cached data. Defaults to None.
        """
        self.source = source
        self.cache = cache
        self.data_home = data_home

    @property
    def dataset_name(self) -> str:
        return "llm_lat_harmful"

    async def fetch_dataset(self) -> SeedDataset:
        """
        Fetch LLM-LAT harmful dataset and return as SeedDataset.

        Returns:
            SeedDataset: A SeedDataset containing the harmful prompts.
        """
        logger.info(f"Loading LLM-LAT harmful dataset from {self.source}")

        data = await self._fetch_from_huggingface(
            dataset_name=self.source,
            config="default",
            split="train",
        )

        seed_prompts = [
            SeedPrompt(
                value=item["prompt"],
                data_type="text",
                name=self.dataset_name,
                dataset_name=self.dataset_name,
                description="This dataset contains prompts used to assess and analyze harmful behaviors in llm",
                source=f"https://huggingface.co/datasets/{self.source}",
            )
            for item in data
        ]

        logger.info(f"Successfully loaded {len(seed_prompts)} prompts from LLM-LAT harmful dataset")

        return SeedDataset(seeds=seed_prompts)
