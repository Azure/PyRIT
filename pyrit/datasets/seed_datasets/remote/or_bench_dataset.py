# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from pyrit.datasets.seed_datasets.remote.remote_dataset_loader import (
    _RemoteDatasetLoader,
)
from pyrit.models import SeedDataset, SeedPrompt

logger = logging.getLogger(__name__)


class _ORBenchDataset(_RemoteDatasetLoader):
    """
    Loader for the OR-Bench dataset from HuggingFace.

    OR-Bench is an over-refusal benchmark that tests whether language models wrongly refuse
    safe prompts. It provides prompts across multiple harm categories that models should
    be able to answer without refusing.

    Available configs:
        - or-bench-hard-1k: ~1k challenging prompts that trigger over-refusal
        - or-bench-toxic: Toxic prompts that models should refuse (for contrast)

    References:
        - https://huggingface.co/datasets/bench-llm/OR-Bench
        - https://arxiv.org/abs/2405.20947
    """

    HF_DATASET_NAME: str = "bench-llm/OR-Bench"

    def __init__(
        self,
        *,
        config: str = "or-bench-hard-1k",
        split: str = "train",
    ):
        """
        Initialize the OR-Bench dataset loader.

        Args:
            config: Dataset configuration. One of "or-bench-hard-1k" or "or-bench-toxic".
                Defaults to "or-bench-hard-1k".
            split: Dataset split to load. Defaults to "train".
        """
        self.config = config
        self.split = split

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "or_bench"

    async def fetch_dataset(self, *, cache: bool = True) -> SeedDataset:
        """
        Fetch OR-Bench dataset from HuggingFace and return as SeedDataset.

        Args:
            cache: Whether to cache the fetched dataset. Defaults to True.

        Returns:
            SeedDataset: A SeedDataset containing the OR-Bench prompts.
        """
        logger.info(f"Loading OR-Bench dataset from {self.HF_DATASET_NAME} (config={self.config})")

        data = await self._fetch_from_huggingface(
            dataset_name=self.HF_DATASET_NAME,
            config=self.config,
            split=self.split,
            cache=cache,
        )

        authors = [
            "Justin Cui",
            "Wei-Lin Chiang",
            "Ion Stoica",
            "Cho-Jui Hsieh",
        ]
        description = (
            "OR-Bench is an over-refusal benchmark that tests whether language models wrongly "
            "refuse safe prompts. It provides prompts that models should handle without refusing."
        )

        source_url = f"https://huggingface.co/datasets/{self.HF_DATASET_NAME}"
        groups = ["UCLA", "UC Berkeley"]

        seed_prompts = [
            SeedPrompt(
                value=f"{{% raw %}}{item['prompt']}{{% endraw %}}",
                data_type="text",
                dataset_name=self.dataset_name,
                harm_categories=[item["category"]] if item.get("category") else [],
                description=description,
                source=source_url,
                authors=authors,
                groups=groups,
                metadata={"or_bench_config": self.config},
            )
            for item in data
        ]

        logger.info(f"Successfully loaded {len(seed_prompts)} prompts from OR-Bench dataset")

        return SeedDataset(seeds=seed_prompts, dataset_name=self.dataset_name)
