# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Literal, Optional

from datasets import load_dataset

from pyrit.models import SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt


def fetch_babelscape_alert_dataset(
    category: Optional[Literal["alert", "alert_adversarial"]] = "alert_adversarial",
) -> SeedPromptDataset:
    """
    Fetch the Babelscape/ALERT dataset and create a SeedPromptDataset.

    Args:
        category (Optional[str]): The dataset category, "alert" or "alert_adversarial".
            If None, both categories will be loaded. Defaults to "alert_adversarial".

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples.
    """

    data_categories = None
    if category is None:  # if category is explicitly None, read both subsets
        data_categories = ["alert_adversarial", "alert"]
    elif category not in ["alert_adversarial", "alert"]:
        raise ValueError(f"Invalid Parameter: {category}. Expected 'alert_adversarial' or 'alert'")
    else:
        data_categories = [category]

    # Load specified subset or both categories
    prompts: list[str] = []
    for name in data_categories:
        data = load_dataset("Babelscape/ALERT", name)
        prompts.extend(item["prompt"] for item in data["test"])

    # Create SeedPrompt instances from each example in 'prompts'
    seed_prompts = [
        SeedPrompt(
            value=prompt,
            data_type="text",
            name="",
            dataset_name="Babelscape/ALERT",
            description="""ALERT by Babelscape is a dataset that consists
            of two different categories, 'alert' with 15k red teaming prompts,
            and 'alert_adversarial' with 30k adversarial red teaming prompts.""",
            source="https://huggingface.co/datasets/Babelscape/ALERT",
        )
        for prompt in prompts
    ]

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset
