# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from pyrit.datasets.seed_datasets.remote.remote_dataset_loader import (
    _RemoteDatasetLoader,
)
from pyrit.models import SeedDataset, SeedPrompt

logger = logging.getLogger(__name__)


class _HarmfulQADataset(_RemoteDatasetLoader):
    """
    Loader for the HarmfulQA dataset from HuggingFace.

    HarmfulQA contains approximately 2k harmful questions organized by academic topic
    and subtopic, designed to test LLM susceptibility to harm-inducing question-answering.

    References:
        - https://huggingface.co/datasets/declare-lab/HarmfulQA
        - https://arxiv.org/abs/2310.18469
    """

    def __init__(
        self,
        *,
        dataset_name: str = "declare-lab/HarmfulQA",
        split: str = "train",
    ):
        """
        Initialize the HarmfulQA dataset loader.

        Args:
            dataset_name: HuggingFace dataset identifier. Defaults to "declare-lab/HarmfulQA".
            split: Dataset split to load. Defaults to "train".
        """
        self.hf_dataset_name = dataset_name
        self.split = split

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "harmful_qa"

    async def fetch_dataset(self, *, cache: bool = True) -> SeedDataset:
        """
        Fetch HarmfulQA dataset from HuggingFace and return as SeedDataset.

        Args:
            cache: Whether to cache the fetched dataset. Defaults to True.

        Returns:
            SeedDataset: A SeedDataset containing the HarmfulQA questions.
        """
        logger.info(f"Loading HarmfulQA dataset from {self.hf_dataset_name}")

        data = await self._fetch_from_huggingface(
            dataset_name=self.hf_dataset_name,
            split=self.split,
            cache=cache,
        )

        authors = ["Rishabh Bhardwaj", "Soujanya Poria"]
        description = (
            "HarmfulQA contains ~2k harmful questions organized by academic topic and subtopic, "
            "designed to test LLM susceptibility to harm-inducing question-answering. Introduced "
            "in 'Red-Teaming Large Language Models using Chain of Utterances for Safety Alignment' (2023)."
        )

        seed_prompts = [
            SeedPrompt(
                value=item["question"],
                data_type="text",
                dataset_name=self.dataset_name,
                harm_categories=[item["topic"]],
                description=description,
                source=f"https://huggingface.co/datasets/{self.hf_dataset_name}",
                authors=authors,
                groups=["DeCLaRe Lab, Singapore University of Technology and Design"],
                metadata={"subtopic": subtopic} if (subtopic := item.get("subtopic")) else {},
            )
            for item in data
        ]

        logger.info(f"Successfully loaded {len(seed_prompts)} questions from HarmfulQA dataset")

        return SeedDataset(seeds=seed_prompts, dataset_name=self.dataset_name)
