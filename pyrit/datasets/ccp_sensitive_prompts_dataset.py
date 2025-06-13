# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from typing import Literal, Optional

from .dataset_helper import FILE_TYPE_HANDLERS, fetch_examples
from ..models import SeedPromptDataset
from ..models.seed_prompt import SeedPrompt


def fetch_ccp_sensitive_prompts_dataset(
    source: str = (
        "https://huggingface.co/datasets/promptfoo/CCP-sensitive-prompts/resolve/main/ccp-sensitive-prompts.csv"
    ),
    source_type: Literal["public_url"] = "public_url",
    cache: bool = True,
    data_home: Optional[Path] = None,
) -> SeedPromptDataset:
    """
    Fetch CCP-sensitive-prompts examples and create a SeedPromptDataset.

    Args:
        source (str): The URL of the CCP-sensitive-prompts CSV file.
        source_type (Literal["public_url"]): The type of source ('public_url').
        cache (bool): Whether to cache the fetched file.
        data_home (Optional[Path]): Directory to store cached data. Defaults to None.

    Returns:
        SeedPromptDataset: A dataset of CCP-sensitive prompts.

    Note:
        For more information, see https://huggingface.co/datasets/promptfoo/CCP-sensitive-prompts
    """
    file_type = source.split(".")[-1]
    if file_type not in FILE_TYPE_HANDLERS:
        valid = ", ".join(FILE_TYPE_HANDLERS.keys())
        raise ValueError(f"Invalid file_type")

    # Required keys 
    required_keys = {"subject", "prompt"}
    raw_prompts = []
    subjects = set()

    examples = fetch_examples(source, source_type, cache, data_home)

    # collect data
    for ex in examples:
        missing = required_keys - ex.keys()
        if missing:
            raise ValueError(f"Missing keys")
        raw_prompts.append(ex["prompt"])
        subjects.add(ex["subject"])

    # Create SeedPrompt
    seed_prompts = [
        SeedPrompt(
            value=txt,
            data_type="text",
            name="CCP Sensitive Prompts",
            dataset_name="CCP-sensitive-prompts",
            harm_categories=list(subjects),
            description=(
                "These prompts cover sensitive topics in China, and are likely to be censored by Chinese models."
            ),
        )
        for txt in raw_prompts
    ]

    return SeedPromptDataset(prompts=seed_prompts)
