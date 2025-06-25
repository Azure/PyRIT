# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datasets import load_dataset

from pyrit.models import SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt


def fetch_ccp_sensitive_prompts_dataset() -> SeedPromptDataset:
    """
    Fetch CCP-sensitive-prompts examples and create a SeedPromptDataset.

    Returns:
        SeedPromptDataset: A dataset of CCP-sensitive prompts.

    Note:
        For more information, see https://huggingface.co/datasets/promptfoo/CCP-sensitive-prompts

        **Author**: promptfoo
        **Purpose**: Collection of prompts that cover sensitive topics in China, and are likely to be censored by
            Chinese models.
    """
    data = load_dataset(
        "promptfoo/CCP-sensitive-prompts",
        split="train",
    )

    return SeedPromptDataset(
        prompts=[
            SeedPrompt(
                value=row["prompt"],
                data_type="text",
                name="",
                dataset_name="CCP-sensitive-prompts",
                harm_categories=[row["subject"]],
                description=("Prompts covering topics sensitive to the CCP."),
                groups=["promptfoo"],
                source="https://huggingface.co/datasets/promptfoo/CCP-sensitive-prompts",
            )
            for row in data
        ]
    )
