# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ast
from pathlib import Path
from typing import List, Literal, Optional

from pyrit.datasets.dataset_helper import fetch_examples
from pyrit.models import SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt


def fetch_aya_redteaming_dataset(
    cache: bool = True,
    data_home: Optional[Path] = None,
    language: Literal["English", "Hindi", "French", "Spanish", "Arabic", "Russian", "Serbian", "Tagalog"] = "English",
    harm_categories: Optional[
        List[
            Literal[
                "Bullying & Harassment",
                "Discrimination & Injustice",
                "Graphic material",
                "Harms of Representation Allocation and Quality of Service",
                "Hate Speech",
                "Non-consensual sexual content",
                "Profanity",
                "Self-Harm",
                "Violence, Threats & Incitement",
            ]
        ]
    ] = None,
    harm_scope: Optional[Literal["global", "local"]] = None,
) -> SeedPromptDataset:
    """
    Fetch examples from the Aya Red-teaming dataset with optional filtering and create a SeedPromptDataset.

    Args:
        cache (bool): Whether to cache the fetched examples. Defaults to True.
        data_home: Directory to store cached data. Defaults to None.
        language: The language to fetch prompts for. Defaults to "English".
        harm_categories: List of harm categories to filter the examples.
            Defaults to None, which means all categories are included.
            Otherwise, only prompts with at least one matching category are included.
        harm_scope: Whether to fetch globally or locally harmful prompts.
            Defaults to None, which means all examples are included.

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the filtered examples.

    Note:
        For more information and access to the original dataset and related materials, visit:
        https://huggingface.co/datasets/CohereForAI/aya_redteaming/blob/main/README.md \n
        Related paper: https://arxiv.org/abs/2406.18682 \n
        The dataset license: Apache 2.0

    Warning:
        Due to the nature of these prompts, it may be advisable to consult your relevant legal
        department before testing them with LLMs to ensure compliance and reduce potential risks.
    """
    _lang = {
        "English": "eng",
        "Hindi": "hin",
        "French": "fra",
        "Spanish": "spa",
        "Arabic": "arb",
        "Russian": "rus",
        "Serbian": "srp",
        "Tagalog": "tgl",
    }

    examples = fetch_examples(
        source=f"https://huggingface.co/datasets/CohereForAI/aya_redteaming/raw/main/aya_{_lang[language]}.jsonl",
        source_type="public_url",
        cache=cache,
        data_home=data_home,
    )

    seed_prompts = []

    for example in examples:
        categories = ast.literal_eval(example["harm_category"])
        if harm_categories is None or any(cat in categories for cat in harm_categories):
            if harm_scope is None or example["global_or_local"] == harm_scope:
                seed_prompts.append(
                    SeedPrompt(
                        value=example["prompt"],
                        data_type="text",
                        name="Aya Red-teaming Examples",
                        dataset_name="Aya Red-teaming Examples",
                        harm_categories=categories,
                        source="https://huggingface.co/datasets/CohereForAI/aya_redteaming",
                    )
                )

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset
