# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datasets import load_dataset

from pyrit.models import SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt


def fetch_llm_latent_adversarial_training_harmful_dataset() -> SeedPromptDataset:
    data = load_dataset("LLM-LAT/harmful-dataset", "default")

    prompts = [item["prompt"] for item in data["train"]]

    # Create SeedPrompt instances from each example in 'prompts'
    seed_prompts = [
        SeedPrompt(
            value=prompt,
            data_type="text",
            name="LLM-LAT/harmful-dataset",
            dataset_name="LLM-LAT/harmful-dataset",
            description="This dataset contains prompts used to assess and analyze harmful behaviors in llm",
            source="https://huggingface.co/datasets/LLM-LAT/harmful-dataset",
        )
        for prompt in prompts
    ]

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset
