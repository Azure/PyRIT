# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Literal, Optional

from datasets import load_dataset

from pyrit.models import SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt


def fetch_pku_safe_rlhf_dataset(
    include_safe_prompts: bool = True,
    filter_harm_categories: Optional[
        List[
            Literal[
                "Animal Abuse",
                "Copyright Issues",
                "Cybercrime",
                "Discriminatory Behavior",
                "Disrupting Public Order",
                "Drugs",
                "Economic Crime",
                "Endangering National Security",
                "Endangering Public Health",
                "Environmental Damage",
                "Human Trafficking",
                "Insulting Behavior",
                "Mental Manipulation",
                "Physical Harm",
                "Privacy Violation",
                "Psychological Harm",
                "Sexual Content",
                "Violence",
                "White-Collar Crime",
            ]
        ]
    ] = None,
) -> SeedPromptDataset:
    """
    Fetch PKU-SafeRLHF examples and create a SeedPromptDataset.

    Args:
        include_safe_prompts (bool): All prompts in the dataset are returned if True; the dataset has
            RLHF markers for unsafe responses, so if False we only return the unsafe subset.
        filter_harm_categories: List of harm categories to filter the examples.
            Defaults to None, which means all categories are included.
            Otherwise, only prompts with at least one matching category are included.

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples.

    Note:
        For more information and access to the original dataset and related materials, visit:
        https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF. Based on research in paper:
        https://arxiv.org/pdf/2406.15513 written by Jiaming Ji and Donghai Hong and Borong Zhang and Boyuan Chen
        and Josef Dai and Boren Zheng and Tianyi Qiu and Boxun Li and Yaodong Yang
    """
    data = load_dataset("PKU-Alignment/PKU-SafeRLHF", "default")

    seed_prompts = []

    for item in data["train"]:
        is_unsafe = not (item["is_response_0_safe"] and item["is_response_1_safe"])

        # Skip if safe prompts are excluded and the item is safe
        if not include_safe_prompts and not is_unsafe:
            continue

        # Collect harm categories from both responses, avoid duplicates
        harm_categories = {
            category
            for response_key in ["response_0_harm_category", "response_1_harm_category"]
            for category, is_harmful in item[response_key].items()
            if is_harmful
        }

        # Add the prompt in two cases:
        # 1. No harm categories are provided for filtering
        # 2. At least one harm category matches the specified filter criteria
        if not filter_harm_categories or any(category in filter_harm_categories for category in harm_categories):
            seed_prompts.append(
                SeedPrompt(
                    value=item["prompt"],
                    data_type="text",
                    name="PKU-SafeRLHF",
                    dataset_name="PKU-SafeRLHF",
                    harm_categories=sorted(list(harm_categories)),
                    description="""This is a Hugging Face dataset that labels a prompt and 2 responses categorizing
                        their helpfulness or harmfulness. Only the 'prompt' column is extracted.""",
                    source="https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF",
                )
            )

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset
