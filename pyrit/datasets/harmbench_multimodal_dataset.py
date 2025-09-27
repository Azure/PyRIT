# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import uuid
from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional

from pyrit.common.net_utility import make_request_and_raise_if_error_async
from pyrit.datasets.dataset_helper import fetch_examples
from pyrit.models import SeedPromptDataset, data_serializer_factory
from pyrit.models.seed_prompt import SeedPrompt

logger = logging.getLogger(__name__)


class SemanticCategory(Enum):
    CYBERCRIME_INTRUSION = "cybercrime_intrusion"  # n=54
    ILLEGAL = "illegal"  # 36
    HARMFUL = "harmful"  # 9
    CHEMICAL_BIOLOGICAL = "chemical_biological"  # 4
    HARASSMENT_BULLYING = "harassment_bullying"  # 4
    MISINFORMATION_DISINFORMATION = "misinformation_disinformation"  # 3


async def fetch_harmbench_multimodal_dataset_async(
    *,
    source: str = (
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/c0423b9/data/behavior_datasets/"
        "harmbench_behaviors_multimodal_all.csv"
    ),
    source_type: Literal["public_url", "file"] = "public_url",
    cache: bool = True,
    data_home: Optional[Path] = None,
    categories: Optional[List[SemanticCategory]] = None,
) -> SeedPromptDataset:
    """
    Fetch HarmBench multimodal examples and create a SeedPromptDataset.

    The HarmBench multimodal dataset contains 110 harmful behaviors.
    Each example consists of an image ("image_path") and a behavior string referencing the image ("text").
    The text and image prompts that belong to the same example are linked using the same ``prompt_group_id``.
    You can extract the grouped prompts using the ``group_seed_prompts_by_prompt_group_id`` method.

    Note: The first call may be slow as images need to be downloaded from the remote repository.
    Subsequent calls will be faster since images are cached locally and won't need to be re-downloaded.

    Args:
        source (str): The source from which to fetch examples. Defaults to the HarmBench repository.
        source_type (Literal["public_url", "file"]): The type of source. Defaults to 'public_url'.
        cache (bool): Whether to cache the fetched examples. Defaults to True.
        data_home (Optional[Path]): Directory to store cached data. Defaults to None.
        categories (Optional[List[SemanticCategory]]): List of semantic categories
            to filter examples. If None, all categories are included (default).

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the multimodal examples.

    Raises:
        ValueError: If any of the specified categories are invalid.

    Note:
        For more information related to the HarmBench project and the original dataset, visit:
        https://www.harmbench.org/ \n
        Paper: https://arxiv.org/abs/2402.04249 \n
        Authors:
            Mantas Mazeika & Long Phan & Xuwang Yin & Andy Zou & Zifan Wang & Norman Mu & Elham Sakhaee
            & Nathaniel Li & Steven Basart & Bo Li & David Forsyth & Dan Hendrycks
    """
    if categories is not None:
        valid_categories = {category.value for category in SemanticCategory}
        invalid_categories = (
            set(cat.value if isinstance(cat, SemanticCategory) else cat for cat in categories) - valid_categories
        )
        if invalid_categories:
            raise ValueError(f"Invalid semantic categories: {', '.join(invalid_categories)}")

    required_keys = {"Behavior", "BehaviorID", "FunctionalCategory", "SemanticCategory", "ImageFileName"}
    examples = fetch_examples(source, source_type, cache, data_home)
    prompts = []

    for example in examples:
        missing_keys = required_keys - example.keys()
        if missing_keys:
            raise ValueError(f"Missing keys in example: {', '.join(missing_keys)}")

        if example["FunctionalCategory"] != "multimodal":
            continue

        semantic_category = example["SemanticCategory"]

        if categories is not None:
            category_values = {cat.value for cat in categories}
            if semantic_category not in category_values:
                continue

        behavior_text = example["Behavior"]
        behavior_id = example["BehaviorID"]
        image_filename = example["ImageFileName"]
        image_description = example.get("ImageDescription", "")
        redacted_description = example.get("RedactedImageDescription", "")

        # A unique group ID to link the text and image prompts since they are part of the same example
        group_id = uuid.uuid4()

        text_prompt = SeedPrompt(
            value=behavior_text,
            data_type="text",
            name=f"HarmBench Multimodal Text - {behavior_id}",
            dataset_name="HarmBench Multimodal Examples",
            harm_categories=[semantic_category],
            description=(f"A text prompt from the HarmBench multimodal dataset, BehaviorID: {behavior_id}"),
            source=source,
            prompt_group_id=group_id,
            sequence=0,
            metadata={
                "behavior_id": behavior_id,
            },
        )
        prompts.append(text_prompt)

        # Note: All images in the HarmBench dataset are stored as .png files, even if the ImageFileName
        # field specifies a different extension (.jpg or .jpeg). Hence we always use .png extension here.
        image_url = (
            "https://raw.githubusercontent.com/centerforaisafety/HarmBench/c0423b9/data/multimodal_behavior_images/"
            f"{image_filename.rsplit('.', 1)[0]}.png"
        )
        local_image_path = await _fetch_and_save_image_async(image_url, behavior_id)

        image_prompt = SeedPrompt(
            value=local_image_path,
            data_type="image_path",
            name=f"HarmBench Multimodal Image - {behavior_id}",
            dataset_name="HarmBench Multimodal Examples",
            harm_categories=[semantic_category],
            description=f"An image prompt from the HarmBench multimodal dataset, BehaviorID: {behavior_id}",
            source=source,
            prompt_group_id=group_id,
            sequence=0,
            metadata={
                "behavior_id": behavior_id,
                "image_description": image_description,
                "redacted_image_description": redacted_description,
                "original_image_url": image_url,
            },
        )
        prompts.append(image_prompt)

    seed_prompt_dataset = SeedPromptDataset(prompts=prompts)
    return seed_prompt_dataset


async def _fetch_and_save_image_async(image_url: str, behavior_id: str) -> str:
    filename = f"harmbench_{behavior_id}.png"
    serializer = data_serializer_factory(category="seed-prompt-entries", data_type="image_path", extension="png")

    # Return existing path if image already exists for this BehaviorID
    serializer.value = str(serializer._memory.results_path + serializer.data_sub_directory + f"/{filename}")
    try:
        if await serializer._memory.results_storage_io.path_exists(serializer.value):
            return serializer.value
    except Exception as e:
        logger.warning(f"Failed to check whether image for {behavior_id} already exists: {e}")

    response = await make_request_and_raise_if_error_async(endpoint_uri=image_url, method="GET")
    await serializer.save_data(data=response.content, output_filename=filename.replace(".png", ""))

    return str(serializer.value)
