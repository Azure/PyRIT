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

    seed_prompts = [
        SeedPrompt(
            value=item["Example"],
            data_type="text",
            name="",
            dataset_name="DarkBench",
            harm_categories=[item["Deceptive Pattern"]],
            description="""The DarkBench dataset focuses on dark patterns and is available on Hugging Face,
                created by anonymous152311 (https://huggingface.co/anonymous152311). The dataset includes
                660 examples, each labeled with a 'Deceptive Pattern' category. These categories indicate
                different types of deceptive strategies used in the data, such as:
                Anthropomorphization, Brand bias, Harmful generation, Sneaking, Sycophancy, or User retention.""",
            source="https://huggingface.co/datasets/anonymous152311/darkbench",
        )
        for item in data["train"]
    ]

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset
