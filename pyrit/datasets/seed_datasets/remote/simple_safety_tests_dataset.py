# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from pyrit.datasets.seed_datasets.remote.remote_dataset_loader import (
    _RemoteDatasetLoader,
)
from pyrit.models import SeedDataset, SeedPrompt

logger = logging.getLogger(__name__)


class _SimpleSafetyTestsDataset(_RemoteDatasetLoader):
    """
    Loader for the SimpleSafetyTests dataset from HuggingFace.

    SimpleSafetyTests contains 100 critical safety test prompts designed as a lightweight
    diagnostic set for quickly evaluating the most basic safety properties of LLMs.

    References:
        - https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests
        - https://arxiv.org/abs/2311.08370
    """

    def __init__(
        self,
        *,
        dataset_name: str = "Bertievidgen/SimpleSafetyTests",
        split: str = "test",
    ):
        """
        Initialize the SimpleSafetyTests dataset loader.

        Args:
            dataset_name: HuggingFace dataset identifier. Defaults to "Bertievidgen/SimpleSafetyTests".
            split: Dataset split to load. Defaults to "test".
        """
        self.hf_dataset_name = dataset_name
        self.split = split

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "simple_safety_tests"

    async def fetch_dataset(self, *, cache: bool = True) -> SeedDataset:
        """
        Fetch SimpleSafetyTests dataset from HuggingFace and return as SeedDataset.

        Args:
            cache: Whether to cache the fetched dataset. Defaults to True.

        Returns:
            SeedDataset: A SeedDataset containing the SimpleSafetyTests prompts.
        """
        logger.info(f"Loading SimpleSafetyTests dataset from {self.hf_dataset_name}")

        data = await self._fetch_from_huggingface(
            dataset_name=self.hf_dataset_name,
            split=self.split,
            cache=cache,
        )

        authors = [
            "Bertie Vidgen",
            "Nino Scherrer",
            "Hannah Rose Kirk",
            "Rebecca Qian",
            "Anand Kannappan",
            "Scott Niekum",
            "Paul Röttger",
        ]
        description = (
            "SimpleSafetyTests contains 100 critical safety test prompts designed as a lightweight "
            "diagnostic set for quickly evaluating the most basic safety properties of LLMs."
        )

        seed_prompts = [
            SeedPrompt(
                value=item["prompt"],
                data_type="text",
                dataset_name=self.dataset_name,
                harm_categories=[item["harm_area"]],
                description=description,
                source=f"https://huggingface.co/datasets/{self.hf_dataset_name}",
                authors=authors,
                groups=["Patronus AI", "University of Oxford", "Bocconi University"],
                metadata={"category": category} if (category := item.get("category")) else {},
            )
            for item in data
        ]

        logger.info(f"Successfully loaded {len(seed_prompts)} prompts from SimpleSafetyTests dataset")

        return SeedDataset(seeds=seed_prompts, dataset_name=self.dataset_name)
