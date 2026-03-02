# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from pyrit.datasets.seed_datasets.remote.remote_dataset_loader import (
    _RemoteDatasetLoader,
)
from pyrit.models import SeedDataset, SeedPrompt

logger = logging.getLogger(__name__)


class _ToxicChatDataset(_RemoteDatasetLoader):
    """
    Loader for the ToxicChat dataset from HuggingFace.

    ToxicChat contains approximately 10k real user-chatbot conversations from the Chatbot Arena,
    annotated for toxicity and jailbreaking attempts. It provides real-world examples of
    how users interact with LLMs in adversarial ways.

    References:
        - https://huggingface.co/datasets/lmsys/toxic-chat
        - https://arxiv.org/abs/2310.17389
    License: CC BY-NC 4.0

    Warning: This dataset contains toxic, offensive, and jailbreaking content from real user
    conversations. Consult your legal department before using these prompts for testing.
    """

    HF_DATASET_NAME: str = "lmsys/toxic-chat"

    def __init__(
        self,
        *,
        config: str = "toxicchat0124",
        split: str = "train",
    ):
        """
        Initialize the ToxicChat dataset loader.

        Args:
            config: Dataset configuration. Defaults to "toxicchat0124".
            split: Dataset split to load. Defaults to "train".
        """
        self.config = config
        self.split = split

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "toxic_chat"

    async def fetch_dataset(self, *, cache: bool = True) -> SeedDataset:
        """
        Fetch ToxicChat dataset from HuggingFace and return as SeedDataset.

        Args:
            cache: Whether to cache the fetched dataset. Defaults to True.

        Returns:
            SeedDataset: A SeedDataset containing the ToxicChat user inputs.
        """
        logger.info(f"Loading ToxicChat dataset from {self.HF_DATASET_NAME}")

        data = await self._fetch_from_huggingface(
            dataset_name=self.HF_DATASET_NAME,
            config=self.config,
            split=self.split,
            cache=cache,
        )

        authors = [
            "Zi Lin",
            "Zihan Wang",
            "Yongqi Tong",
            "Yangkun Wang",
            "Yuxin Guo",
            "Yujia Wang",
            "Jingbo Shang",
        ]
        description = (
            "ToxicChat contains ~10k real user-chatbot conversations from the Chatbot Arena, "
            "annotated for toxicity and jailbreaking attempts. It provides real-world examples "
            "of adversarial user interactions with LLMs."
        )

        source_url = f"https://huggingface.co/datasets/{self.HF_DATASET_NAME}"
        groups = ["UC San Diego"]

        seed_prompts = [
            SeedPrompt(
                value=f"{{% raw %}}{item['user_input']}{{% endraw %}}",
                data_type="text",
                dataset_name=self.dataset_name,
                description=description,
                source=source_url,
                authors=authors,
                groups=groups,
                metadata={
                    "toxicity": str(item.get("toxicity", "")),
                    "jailbreaking": str(item.get("jailbreaking", "")),
                    "human_annotation": str(item.get("human_annotation", "")),
                },
            )
            for item in data
        ]

        logger.info(f"Successfully loaded {len(seed_prompts)} prompts from ToxicChat dataset")

        return SeedDataset(seeds=seed_prompts, dataset_name=self.dataset_name)
