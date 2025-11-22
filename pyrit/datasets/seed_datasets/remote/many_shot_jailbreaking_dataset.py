# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from typing import Literal, Optional

from pyrit.datasets.seed_datasets.remote.remote_dataset_loader import RemoteDatasetLoader
from pyrit.models import SeedDataset, SeedPrompt


class ManyShotJailbreakingDataset(RemoteDatasetLoader):
    """
    Loader for the Many-Shot Jailbreaking dataset.

    This dataset contains many-shot jailbreaking examples.

    Reference: https://github.com/KutalVolkan/many-shot-jailbreaking-dataset
    """

    def __init__(
        self,
        *,
        source: str = (
            "https://raw.githubusercontent.com/KutalVolkan/many-shot-jailbreaking-dataset/5eac855/examples.json"
        ),
        source_type: Literal["public_url", "file"] = "public_url",
        cache: bool = True,
        data_home: Optional[Path] = None,
    ):
        """
        Initialize the Many-Shot Jailbreaking dataset loader.

        Args:
            source: URL to the JSON file. Defaults to the official repository.
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
        return "many_shot_jailbreaking"

    async def fetch_dataset(self) -> SeedDataset:
        """
        Fetch Many-Shot Jailbreaking dataset and return as SeedDataset.

        Returns:
            SeedDataset: A SeedDataset containing the many-shot jailbreaking examples.
        """
        examples = self._fetch_from_url(
            source=self.source,
            source_type=self.source_type,
            cache=self.cache,
            data_home=self.data_home,
        )

        seed_prompts = [
            SeedPrompt(
                value=example.get("Prompt", ""),
                data_type="text",
                name=self.dataset_name,
                dataset_name=self.dataset_name,
            )
            for example in examples
        ]

        return SeedDataset(prompts=seed_prompts)
