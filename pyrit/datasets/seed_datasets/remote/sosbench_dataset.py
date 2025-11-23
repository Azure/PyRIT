# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from pathlib import Path
from typing import Optional

from pyrit.datasets.seed_datasets.remote.remote_dataset_loader import RemoteDatasetLoader
from pyrit.models import SeedDataset, SeedPrompt

logger = logging.getLogger(__name__)


class SOSBenchDataset(RemoteDatasetLoader):
    """
    Loader for the SOSBench dataset.

    SOSBench is a regulation-grounded, hazard-focused benchmark encompassing six high-risk
    scientific domains: chemistry, biology, medicine, pharmacology, physics, and psychology.
    The benchmark comprises 3,000 prompts derived from real-world regulations and laws.

    Reference: https://arxiv.org/abs/2505.21605
    """

    def __init__(
        self,
        *,
        source: str = "SOSBench/SOSBench",
        cache: bool = True,
        data_home: Optional[Path] = None,
    ):
        """
        Initialize the SOSBench dataset loader.

        Args:
            source: HuggingFace dataset identifier. Defaults to "SOSBench/SOSBench".
            cache: Whether to cache the fetched examples. Defaults to True.
            data_home: Directory to store cached data. Defaults to None.
        """
        self.source = source
        self.cache = cache
        self.data_home = data_home

    @property
    def dataset_name(self) -> str:
        return "sosbench"

    async def fetch_dataset(self) -> SeedDataset:
        """
        Fetch SOSBench dataset and return as SeedDataset.

        Returns:
            SeedDataset: A SeedDataset containing the SOSBench prompts.
        """
        logger.info(f"Loading SOSBench dataset from {self.source}")

        data = await self._fetch_from_huggingface(
            dataset_name=self.source,
            config="default",
            split="train",
        )

        seed_prompts = [
            SeedPrompt(
                value=item["goal"],
                data_type="text",
                name=self.dataset_name,
                dataset_name=self.dataset_name,
                harm_categories=[item["subject"]],
                description=(
                    "SOSBench is a regulation-grounded, hazard-focused benchmark encompassing "
                    "six high-risk scientific domains: chemistry, biology, medicine, pharmacology, "
                    "physics, and psychology. The benchmark comprises 3,000 prompts derived from "
                    "real-world regulations and laws, systematically expanded via an LLM-assisted "
                    "evolutionary pipeline that introduces diverse, realistic misuse scenarios"
                    " (e.g., detailed explosive synthesis instructions involving advanced"
                    " chemical formulas)."
                ),
                source=f"https://huggingface.co/datasets/{self.source}",
                authors=[
                    "Fengqing Jiang",
                    "Fengbo Ma",
                    "Zhangchen Xu",
                    "Yuetai Li",
                    "Bhaskar Ramasubramanian",
                    "Luyao Niu",
                    "Bo Li",
                    "Xianyan Chen",
                    "Zhen Xiang",
                    "Radha Poovendran",
                ],
            )
            for item in data
        ]

        logger.info(f"Successfully loaded {len(seed_prompts)} prompts from SOSBench dataset")

        return SeedDataset(seeds=seed_prompts, dataset_name=self.dataset_name)
