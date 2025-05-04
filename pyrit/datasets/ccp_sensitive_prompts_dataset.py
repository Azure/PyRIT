"""
Dataset loader for the CCP Sensitive Prompts dataset from promptfoo.

Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
"""

import pathlib
import typing

from pyrit.datasets import dataset_helper
from pyrit.models import SeedPrompt, SeedPromptDataset


def fetch_ccp_sensitive_prompts_dataset(
    cache: bool = True, data_home: typing.Optional[pathlib.Path] = None
) -> SeedPromptDataset:
    """
    Fetches prompts related to topics considered sensitive by the Chinese Communist Party (CCP),
    sourced from the promptfoo Hugging Face repository.

    Args:
        cache (bool): Whether to cache the downloaded dataset. Defaults to True.
        data_home (Optional[pathlib.Path]): The directory to cache the data. If None,
                                               uses the default PyRIT cache location.
                                               Defaults to None.

    Returns:
        SeedPromptDataset: A dataset containing the CCP sensitive prompts.
    """
    url = "https://huggingface.co/datasets/promptfoo/CCP-sensitive-prompts/resolve/main/ccp-sensitive-prompts.csv"

    data = dataset_helper.fetch_examples(
        source=url,
        source_type="public_url",
        cache=cache,
        data_home=data_home,
    )

    seed_prompts = []
    for item in data:
        seed_prompts.append(
            SeedPrompt(
                value=item['prompt'],
                data_type="text",
                name="CCP Sensitive Prompts",
                dataset_name="promptfoo/CCP-sensitive-prompts",
                harm_categories=[item['subject']],
                description="Prompts related to topics considered sensitive by the Chinese Communist Party (CCP), " +
                            "sourced from promptfoo.",
                source="https://huggingface.co/datasets/promptfoo/CCP-sensitive-prompts",
            )
        )

    return SeedPromptDataset(prompts=seed_prompts)
