# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datasets import load_dataset

from pyrit.models import SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt


def fetch_tdc23_redteaming_dataset() -> SeedPromptDataset:
    """
    Fetch TDC23-RedTeaming examples and create a SeedPromptDataset.

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples.
    """
    # Load the TDC23-RedTeaming dataset
    data = load_dataset("walledai/TDC23-RedTeaming", "default")

    prompts = [item["prompt"] for item in data["train"]]

    # Create SeedPrompt instances from each example in 'prompts'
    seed_prompts = [
        SeedPrompt(
            value=prompt,
            data_type="text",
            name="walledai/TDC23-RedTeaming",
            dataset_name="walledai/TDC23-RedTeaming",
            description="""TDC23-RedTeaming dataset from HuggingFace,
                    created by Walled AI (https://huggingface.co/walledai).
                    Contains 100 prompts aimed at generating harmful content
                    across multiple harm categories related to fairness,
                    misinformation, dangerous and criminal activities,
                    violence, etc. in the style of writing narratives.""",
            source="https://huggingface.co/datasets/walledai/TDC23-RedTeaming",
        )
        for prompt in prompts
    ]

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset
