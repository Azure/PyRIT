# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from pathlib import Path
from typing import Literal, Optional

from pyrit.datasets.dataset_helper import fetch_examples
from pyrit.models import SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt


def fetch_harmbench_multimodal_dataset(
    source: str = (
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/c0423b9/data/behavior_datasets/"
        "harmbench_behaviors_multimodal_all.csv"
    ),
    source_type: Literal["public_url", "file"] = "public_url",
    cache: bool = True,
    data_home: Optional[Path] = None,
) -> SeedPromptDataset:
    """
    Fetch HarmBench multimodal examples and create a SeedPromptDataset.

    Args:
        source (str): The source from which to fetch examples. Defaults to the HarmBench repository.
        source_type (Literal["public_url", "file"]): The type of source. Defaults to 'public_url'.
        cache (bool): Whether to cache the fetched examples. Defaults to True.
        data_home (Optional[Path]): Directory to store cached data. Defaults to None.

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the multimodal examples.

    Note:
        For more information related to the HarmBench project and the original dataset, visit: https://www.harmbench.org/
        Paper: https://arxiv.org/abs/2402.04249
        Authors:
            Mantas Mazeika & Long Phan & Xuwang Yin & Andy Zou & Zifan Wang & Norman Mu & Elham Sakhaee
            & Nathaniel Li & Steven Basart & Bo Li & David Forsyth & Dan Hendrycks
    """
    required_keys = {"Behavior", "BehaviorID", "FunctionalCategory", "SemanticCategory", "ImageFileName"}
    examples = fetch_examples(source, source_type, cache, data_home)
    prompts = []

    for example in examples:
        missing_keys = required_keys - example.keys()
        if missing_keys:
            raise ValueError(f"Missing keys in example: {', '.join(missing_keys)}")

        if example["FunctionalCategory"] != "multimodal":
            continue

        behavior_text = example["Behavior"]
        behavior_id = example["BehaviorID"]
        semantic_category = example["SemanticCategory"]
        image_filename = example["ImageFileName"]
        image_description = example.get("ImageDescription", "")
        redacted_description = example.get("RedactedImageDescription", "")

        # A unique group ID to link the text and image prompts
        # since they are part of the same example
        group_id = uuid.uuid4()

        text_prompt = SeedPrompt(
            value=behavior_text,
            data_type="text",
            name=f"HarmBench Multimodal Text - {behavior_id}",
            dataset_name="HarmBench Multimodal Examples",
            harm_categories=list(semantic_category),
            description=(f"A text prompt from the HarmBench multimodal dataset, BehaviorID: {behavior_id}"),
            source=source,
            prompt_group_id=group_id,
            sequence=0,
            metadata={
                "behavior_id": behavior_id,
            },
        )
        prompts.append(text_prompt)

        image_prompt = SeedPrompt(
            # Note: All images in the HarmBench dataset are stored as .png files, even if
            # the ImageFileName field specifies a different extension (.jpg or .jpeg).
            # https://github.com/centerforaisafety/HarmBench/tree/c0423b9/data/multimodal_behavior_images
            value=f"https://raw.githubusercontent.com/centerforaisafety/HarmBench/c0423b9/data/multimodal_behavior_images/{image_filename.rsplit('.', 1)[0]}.png",
            data_type="image_path",
            name=f"HarmBench Multimodal Image - {behavior_id}",
            dataset_name="HarmBench Multimodal Examples",
            harm_categories=list(semantic_category),
            description=f"An image prompt from the HarmBench multimodal dataset, BehaviorID: {behavior_id}",
            source=example.get("Source", ""),
            prompt_group_id=group_id,
            sequence=0,
            metadata={
                "behavior_id": behavior_id,
                "image_description": image_description,
                "redacted_image_description": redacted_description,
            },
        )
        prompts.append(image_prompt)

    seed_prompt_dataset = SeedPromptDataset(prompts=prompts)
    return seed_prompt_dataset
