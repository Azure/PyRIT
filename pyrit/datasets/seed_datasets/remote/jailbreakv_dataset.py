# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Literal, Optional

from pyrit.datasets.seed_datasets.remote.remote_dataset_loader import (
    _RemoteDatasetLoader,
)
from pyrit.models import SeedDataset, SeedPrompt

logger = logging.getLogger(__name__)


class _JailbreakVDataset(_RemoteDatasetLoader):
    """
    Loader for the JailBreakV-28K dataset.

    This dataset contains 28,000+ jailbreak prompts across 16 safety policy categories,
    designed to evaluate the robustness of multimodal large language models against
    jailbreak attacks.

    Reference: https://arxiv.org/abs/2404.03027
    HuggingFace: https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k
    License: MIT

    Content Warning: This dataset contains prompts aimed at provoking harmful responses
    and may contain offensive content. Users should check with their legal department
    before using these prompts against production LLMs.
    """

    POLICY_CATEGORIES = [
        "Animal Abuse",
        "Bias",
        "Economic Harm",
        "Fraud",
        "Government Decision",
        "Hate Speech",
        "Health Consultation",
        "Illegal Activity",
        "Malware",
        "Physical Harm",
        "Political Sensitivity",
        "Privacy Violation",
        "Tailored Unlicensed Advice",
        "Unethical Behavior",
        "Violence",
    ]

    def __init__(
        self,
        *,
        source: str = "JailbreakV-28K/JailBreakV-28k",
        config: Literal["JailBreakV_28K", "RedTeam_2K"] = "JailBreakV_28K",
        split: Optional[str] = None,
    ):
        """
        Initialize the JailBreakV-28K dataset loader.

        Args:
            source: HuggingFace dataset identifier.
                Defaults to "JailbreakV-28K/JailBreakV-28k".
            config: Dataset configuration to load.
                "JailBreakV_28K" for jailbreak prompts (default),
                "RedTeam_2K" for red team queries.
            split: Dataset split to load.
                For JailBreakV_28K config: "mini_JailBreakV_28K" (280 rows) or
                    "JailBreakV_28K" (28,300 rows). Defaults to "mini_JailBreakV_28K".
                For RedTeam_2K config: "RedTeam_2K" (2,000 rows). Defaults to "RedTeam_2K".
        """
        self.source = source
        self.config = config

        if split is not None:
            self.split = split
        elif config == "JailBreakV_28K":
            self.split = "mini_JailBreakV_28K"
        else:
            self.split = "RedTeam_2K"

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "jailbreakv_28k"

    async def fetch_dataset(self, *, cache: bool = True) -> SeedDataset:
        """
        Fetch JailBreakV-28K dataset and return as SeedDataset.

        Args:
            cache: Whether to cache the fetched dataset. Defaults to True.

        Returns:
            SeedDataset: A SeedDataset containing jailbreak prompts with harm_categories
                derived from the dataset's "policy" column.

        Raises:
            ValueError: If the dataset is empty after processing.
            Exception: If the dataset cannot be loaded or processed.
        """
        try:
            logger.info(f"Loading JailBreakV-28K dataset (config={self.config}, split={self.split})")

            data = await self._fetch_from_huggingface(
                dataset_name=self.source,
                config=self.config,
                split=self.split,
                cache=cache,
            )

            seed_prompts = []

            for item in data:
                if self.config == "JailBreakV_28K":
                    prompt_text = item.get("jailbreak_query", "").strip()
                else:
                    prompt_text = item.get("question", "").strip()

                if not prompt_text:
                    logger.warning("[JailBreakV-28K] Skipping item with empty prompt field")
                    continue

                policy = item.get("policy", "")
                harm_categories = [policy] if policy else []

                metadata = {}
                if self.config == "JailBreakV_28K":
                    redteam_query = item.get("redteam_query", "")
                    if redteam_query:
                        metadata["redteam_query"] = redteam_query
                    fmt = item.get("format", "")
                    if fmt:
                        metadata["format"] = fmt
                    source_from = item.get("from", "")
                    if source_from:
                        metadata["from"] = source_from

                seed_prompt = SeedPrompt(
                    value=prompt_text,
                    data_type="text",
                    dataset_name=self.dataset_name,
                    harm_categories=harm_categories,
                    source="https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k",
                    authors=["JailbreakV-28K Team"],
                    metadata=metadata,
                )

                seed_prompts.append(seed_prompt)

            if not seed_prompts:
                raise ValueError("SeedDataset cannot be empty.")

            logger.info(f"Successfully loaded {len(seed_prompts)} prompts from JailBreakV-28K dataset")

            return SeedDataset(seeds=seed_prompts, dataset_name=self.dataset_name)

        except Exception as e:
            logger.error(f"Failed to load JailBreakV-28K dataset: {str(e)}")
            raise Exception(f"Error loading JailBreakV-28K dataset: {str(e)}")
