# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from pathlib import Path
from typing import Optional

from pyrit.datasets.seed_datasets.remote.remote_dataset_loader import RemoteDatasetLoader
from pyrit.models import SeedDataset, SeedPrompt

logger = logging.getLogger(__name__)


class LibrAIDoNotAnswerDataset(RemoteDatasetLoader):
    """
    Loader for the LibrAI 'Do Not Answer' dataset.

    This dataset contains questions across multiple risk areas and harm types
    to test LLM safety and refusal behaviors.

    Reference: https://arxiv.org/abs/2308.13387
    GitHub: https://github.com/libr-ai/do-not-answer
    """

    def __init__(
        self,
        *,
        source: str = "LibrAI/do-not-answer",
        cache: bool = True,
        data_home: Optional[Path] = None,
    ):
        """
        Initialize the LibrAI Do Not Answer dataset loader.

        Args:
            source: HuggingFace dataset identifier. Defaults to "LibrAI/do-not-answer".
            cache: Whether to cache the fetched examples. Defaults to True.
            data_home: Directory to store cached data. Defaults to None.
        """
        self.source = source
        self.cache = cache
        self.data_home = data_home

    @property
    def dataset_name(self) -> str:
        return "librai_do_not_answer"

    async def fetch_dataset(self) -> SeedDataset:
        """
        Fetch LibrAI Do Not Answer dataset and return as SeedDataset.

        Returns:
            SeedDataset: A SeedDataset containing the Do Not Answer prompts.
        """
        logger.info(f"Loading LibrAI Do Not Answer dataset from {self.source}")

        data = await self._fetch_from_huggingface(
            dataset_name=self.source,
            split="train",
        )

        seed_prompts = [
            SeedPrompt(
                value=entry["question"],
                data_type="text",
                name=self.dataset_name,
                dataset_name=self.dataset_name,
                harm_categories=[entry["risk_area"], entry["types_of_harm"], entry["specific_harms"]],
                description=(
                    f"This is a prompt from the 'Do Not Answer' dataset under the risk area: {entry['risk_area']}, "
                    f"harm type: {entry['types_of_harm']}, and specific harm: {entry['specific_harms']}."
                ),
                source=f"https://huggingface.co/datasets/{self.source}",
            )
            for entry in data
        ]

        logger.info(f"Successfully loaded {len(seed_prompts)} prompts from LibrAI Do Not Answer dataset")

        return SeedDataset(prompts=seed_prompts)
