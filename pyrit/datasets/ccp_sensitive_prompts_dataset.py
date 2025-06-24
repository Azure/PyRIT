# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from typing import Optional

from pyrit.datasets.dataset_helper import FILE_TYPE_HANDLERS, fetch_examples
from pyrit.models import SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt


def fetch_ccp_sensitive_prompts_dataset(
    source: str = (
        "https://huggingface.co/datasets/promptfoo/CCP-sensitive-prompts/resolve/main/ccp-sensitive-prompts.csv"
    ),
    cache: bool = True,
    data_home: Optional[Path] = None,
) -> SeedPromptDataset:
    """
    Fetch CCP-sensitive-prompts examples and create a SeedPromptDataset.

    Args:
        source (str): The URL of the CCP-sensitive-prompts CSV file.
        cache (bool): Whether to cache the fetched file.
        data_home (Optional[Path]): Directory to store cached data. Defaults to None.

    Returns:
        SeedPromptDataset: A dataset of CCP-sensitive prompts.

    .. note::
        For more information, see https://huggingface.co/datasets/promptfoo/CCP-sensitive-prompts

        **Author**: promptfoo (Hugging Face user)
        **Purpose**: Collection of prompts that cover sensitive topics in China, and are likely to be censored by
            Chinese models.
    """
    file_type = source.split(".")[-1]
    if file_type not in FILE_TYPE_HANDLERS:
        valid = ", ".join(FILE_TYPE_HANDLERS.keys())
        raise ValueError(f"Invalid file_type {valid}")

    # Required keys
    required_keys = {"subject", "prompt"}

    examples = fetch_examples(source, "public_url", cache, data_home)
    seed_prompts = []
    for ex in examples:
        missing = required_keys - ex.keys()
        if missing:
            raise ValueError(f"Missing keys: {missing} in {ex}")

        # Create SeedPrompt
        seed_prompts.append(
            SeedPrompt(
                value=ex["prompt"],
                data_type="text",
                name="CCP Sensitive Prompts",
                dataset_name="CCP-sensitive-prompts",
                harm_categories=[ex["subject"]],
                description=("Prompts censored by Chinese models, covering topics sensitive to the CCP."),
            )
        )

    return SeedPromptDataset(prompts=seed_prompts)
