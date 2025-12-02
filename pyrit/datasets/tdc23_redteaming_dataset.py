# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datasets import load_dataset

from pyrit.models import SeedDataset, SeedPrompt


def fetch_tdc23_redteaming_dataset() -> SeedDataset:
    """
    Fetch TDC23-RedTeaming examples and create a SeedDataset.

    Returns:
        SeedDataset: A SeedDataset containing the examples.
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

    seed_dataset = SeedDataset(seeds=seed_prompts)
    return seed_dataset
