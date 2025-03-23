# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datasets import load_dataset

from pyrit.models import SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt


def fetch_darkbench_dataset() -> SeedPromptDataset:
    """
    Fetch DarkBench examples and create a SeedPromptDataset.

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples.
    """
    data = load_dataset("anonymous152311/darkbench", "default")

    prompts = [item["Example"] for item in data["train"]]

    seed_prompts = [
        SeedPrompt(
            value=prompt,
            data_type="text",
            name="",
            dataset_name="DarkBench",
            description="""DarkBench dataset on dark patterns from HuggingFace,
                created by anonymous152311 (https://huggingface.co/anonymous152311).""",
            source="https://huggingface.co/datasets/anonymous152311/darkbench",
        )
        for prompt in prompts
    ]

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset
