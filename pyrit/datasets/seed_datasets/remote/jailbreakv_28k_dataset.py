# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib
import uuid
import zipfile
from enum import Enum
from typing import Dict, List, Literal, Optional

from pyrit.datasets.seed_datasets.remote.remote_dataset_loader import (
    _RemoteDatasetLoader,
)
from pyrit.models import SeedDataset, SeedPrompt

logger = logging.getLogger(__name__)


class HarmCategory(Enum):
    """Harm categories in the JailBreakV-28K dataset."""

    UNETHICAL_BEHAVIOR = "Unethical Behavior"
    ECONOMIC_HARM = "Economic Harm"
    HATE_SPEECH = "Hate Speech"
    GOVERNMENT_DECISION = "Government Decision"
    PHYSICAL_HARM = "Physical Harm"
    FRAUD = "Fraud"
    POLITICAL_SENSITIVITY = "Political Sensitivity"
    MALWARE = "Malware"
    ILLEGAL_ACTIVITY = "Illegal Activity"
    BIAS = "Bias"
    VIOLENCE = "Violence"
    ANIMAL_ABUSE = "Animal Abuse"
    TAILORED_UNLICENSED_ADVICE = "Tailored Unlicensed Advice"
    PRIVACY_VIOLATION = "Privacy Violation"
    HEALTH_CONSULTATION = "Health Consultation"
    CHILD_ABUSE_CONTENT = "Child Abuse Content"


class _JailbreakV28KDataset(_RemoteDatasetLoader):
    """
    Loader for the JailBreakV-28K multimodal dataset.

    The JailBreakV-28K dataset is a benchmark for assessing the robustness of
    multimodal large language models against jailbreak attacks. Each example consists
    of an image and a text query, linked by the same prompt_group_id.

    Note: Most images are not available on HuggingFace. You must download the full image
    set from Google Drive by filling out the form at:
    https://docs.google.com/forms/d/e/1FAIpQLSc_p1kCs3p9z-3FbtSeF7uLYsiQk0tvsGi6F0e_z5xCEmN1gQ/viewform

    Reference: https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k
    Paper: https://arxiv.org/abs/2404.03027
    Authors: Weidi Luo, Siyuan Ma, Xiaogeng Liu, Chaowei Xiao, Xiaoyu Guo
    License: MIT

    Warning: Due to the nature of these prompts, consult your legal department
    before testing them with LLMs to ensure compliance and reduce potential risks.
    """

    def __init__(
        self,
        *,
        source: str = "JailbreakV-28K/JailBreakV-28k",
        zip_dir: str = str(pathlib.Path.home()),
        split: Literal["JailBreakV_28K", "mini_JailBreakV_28K"] = "mini_JailBreakV_28K",
        text_field: Literal["jailbreak_query", "redteam_query"] = "redteam_query",
        harm_categories: Optional[List[HarmCategory]] = None,
    ) -> None:
        """
        Initialize the JailBreakV-28K dataset loader.

        Args:
            source: HuggingFace dataset identifier. Defaults to "JailbreakV-28K/JailBreakV-28k".
            zip_dir: Directory containing the JailBreakV_28K.zip file with images.
                Defaults to home directory.
            split: Dataset split to load. Defaults to "mini_JailBreakV_28K".
                Options are "JailBreakV_28K" and "mini_JailBreakV_28K".
            text_field: Field to use as the prompt text. Defaults to "redteam_query".
                Options are "jailbreak_query" and "redteam_query".
            harm_categories: List of harm categories to filter examples.
                If None, all categories are included (default).

        Raises:
            ValueError: If any of the specified harm categories are invalid.
        """
        self.source = source
        self.zip_dir = pathlib.Path(zip_dir)
        self.split = split
        self.text_field = text_field
        self.harm_categories = harm_categories

        # Validate harm categories if provided
        if harm_categories is not None:
            valid_categories = {category.value for category in HarmCategory}
            invalid_categories = (
                set(cat.value if isinstance(cat, HarmCategory) else cat for cat in harm_categories)
                - valid_categories
            )
            if invalid_categories:
                raise ValueError(f"Invalid harm categories: {', '.join(invalid_categories)}")

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "jailbreakv_28k"

    async def fetch_dataset(self, *, cache: bool = True) -> SeedDataset:
        """
        Fetch JailBreakV-28K dataset and return as SeedDataset.

        The dataset contains both image and text prompts linked by prompt_group_id.
        You can extract the grouped prompts using the group_seed_prompts_by_prompt_group_id method.

        Args:
            cache: Whether to cache the fetched dataset. Defaults to True.

        Returns:
            SeedDataset: A SeedDataset containing the multimodal examples.

        Raises:
            FileNotFoundError: If the required ZIP file is not found.
            ValueError: If the number of prompts is below the minimum threshold.
            Exception: If the dataset cannot be loaded or processed.
        """
        # Extract images from ZIP if needed
        zip_file_path = self.zip_dir / "JailBreakV_28K.zip"
        zip_extracted_path = self.zip_dir / "JailBreakV_28K"

        if not zip_file_path.exists():
            raise FileNotFoundError(
                f"ZIP file not found at {zip_file_path}. "
                "Please download images from Google Drive using the form at: "
                "https://docs.google.com/forms/d/e/1FAIpQLSc_p1kCs3p9z-3FbtSeF7uLYsiQk0tvsGi6F0e_z5xCEmN1gQ/viewform"
            )

        # Only unzip if the target directory does not already exist
        if not zip_extracted_path.exists():
            logger.info(f"Extracting {zip_file_path} to {self.zip_dir}")
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(self.zip_dir)

        try:
            logger.info(f"Loading JailBreakV-28K dataset from {self.source}")

            # Load dataset from HuggingFace using the helper method
            data = await self._fetch_from_huggingface(
                dataset_name=self.source,
                config="JailBreakV_28K",
                split=self.split,
                cache=cache,
            )

            # Normalize the harm categories for filtering
            harm_categories_normalized = (
                None
                if self.harm_categories is None
                else [self._normalize_policy(cat.value) for cat in self.harm_categories]
            )

            seed_prompts = []
            missing_images = 0
            total_items_processed = 0
            per_call_cache: Dict[str, str] = {}

            for item in data:
                policy = self._normalize_policy(item.get("policy", ""))

                # Skip if user requested policy filter and item's policy does not match
                if harm_categories_normalized is not None and policy not in harm_categories_normalized:
                    continue

                # Count items that pass the filter
                total_items_processed += 1

                image_rel_path = item.get("image_path", "")
                if not image_rel_path:
                    missing_images += 1
                    continue

                image_abs_path = self._resolve_image_path(
                    rel_path=image_rel_path,
                    local_directory=zip_extracted_path,
                    call_cache=per_call_cache,
                )

                if not image_abs_path:
                    missing_images += 1
                    continue

                # Create linked text and image prompts
                group_id = uuid.uuid4()

                text_seed_prompt = SeedPrompt(
                    value=item.get(self.text_field, ""),
                    data_type="text",
                    name="JailBreakV-28K",
                    dataset_name=self.dataset_name,
                    harm_categories=[policy],
                    description=(
                        "Benchmark for Assessing the Robustness of "
                        "Multimodal Large Language Models against Jailbreak Attacks."
                    ),
                    authors=["Weidi Luo", "Siyuan Ma", "Xiaogeng Liu", "Chaowei Xiao", "Xiaoyu Guo"],
                    groups=["The Ohio State University", "Peking University", "University of Wisconsin-Madison"],
                    source="https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k",
                    prompt_group_id=group_id,
                    sequence=0,
                )

                image_seed_prompt = SeedPrompt(
                    value=image_abs_path,
                    data_type="image_path",
                    name="JailBreakV-28K",
                    dataset_name=self.dataset_name,
                    harm_categories=[policy],
                    description=(
                        "Benchmark for Assessing the Robustness of "
                        "Multimodal Large Language Models against Jailbreak Attacks."
                    ),
                    authors=["Weidi Luo", "Siyuan Ma", "Xiaogeng Liu", "Chaowei Xiao", "Xiaoyu Guo"],
                    groups=["The Ohio State University", "Peking University", "University of Wisconsin-Madison"],
                    source="https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k",
                    prompt_group_id=group_id,
                    sequence=0,
                )

                seed_prompts.append(text_seed_prompt)
                seed_prompts.append(image_seed_prompt)

        except Exception as e:
            logger.error(f"Failed to load JailBreakV-28K dataset: {str(e)}")
            raise

        # Validation: Check if 50% or more of the responses are unpaired
        if total_items_processed == 0:
            raise ValueError(
                "JailBreakV-28K fetch produced 0 items after filtering. "
                "Try adjusting your harm_categories filter or check the dataset source."
            )

        successful_pairs = len(seed_prompts) // 2  # Each pair has text + image
        unpaired_percentage = (missing_images / total_items_processed) * 100

        if unpaired_percentage >= 50:
            raise ValueError(
                f"JailBreakV-28K fetch failed: {unpaired_percentage:.1f}% of items are missing images "
                f"({missing_images} out of {total_items_processed} items processed). "
                f"Only {successful_pairs} valid pairs were created. "
                f"At least 50% of items must have valid images. "
                f"Please ensure the ZIP file contains the full image set."
            )

        if missing_images > 0:
            logger.warning(
                f"Failed to resolve {missing_images} image paths in JailBreakV-28K dataset "
                f"({unpaired_percentage:.1f}% unpaired)"
            )

        logger.info(
            f"Successfully loaded {successful_pairs} multimodal pairs "
            f"({len(seed_prompts)} total prompts) from JailBreakV-28K dataset"
        )

        return SeedDataset(seeds=seed_prompts, dataset_name=self.dataset_name)

    def _normalize_policy(self, policy: str) -> str:
        """
        Create a machine-friendly variant of the policy category.

        Args:
            policy: The human-readable policy category.

        Returns:
            str: The normalized policy category.
        """
        return policy.strip().lower().replace(" ", "_").replace("-", "_")

    def _resolve_image_path(
        self,
        *,
        rel_path: str,
        local_directory: pathlib.Path,
        call_cache: Dict[str, str],
    ) -> str:
        """
        Resolve a repository-relative image path to a local absolute path.

        Uses a cache to avoid re-checking the same file multiple times.

        Args:
            rel_path: Path relative to the dataset repository root (e.g., "images/0001.png").
            local_directory: Directory to search for the image.
            call_cache: Cache dictionary to store resolved paths.

        Returns:
            str: Absolute local path if resolved, else empty string.
        """
        if not rel_path:
            return ""

        # Check if image has already been cached
        if rel_path in call_cache:
            return call_cache[rel_path]

        image_path = local_directory / rel_path

        try:
            if image_path.exists():
                abs_path = str(image_path)
            else:
                logger.debug(f"File {image_path} not found in {local_directory}")
                abs_path = ""

            call_cache[rel_path] = abs_path
            return abs_path

        except Exception as e:
            logger.error(f"Failed to resolve image path {rel_path}: {str(e)}")
            call_cache[rel_path] = ""
            return ""
