# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datasets import load_dataset

from pyrit.models import SeedDataset, SeedPrompt


def fetch_ccp_sensitive_prompts_dataset() -> SeedDataset:
    """
    Fetch CCP-sensitive-prompts examples and create a SeedDataset.

    The dataset was collected by Promptfoo to cover sensitive topics in China.
    These are likely to be censored by certain models.

    Returns:
        SeedDataset: A dataset of CCP-sensitive prompts.

    Note:
        For more information, see https://huggingface.co/datasets/promptfoo/CCP-sensitive-prompts
    """
    data = load_dataset(
        "promptfoo/CCP-sensitive-prompts",
        split="train",
    )

    return SeedDataset(
        seeds=[
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
