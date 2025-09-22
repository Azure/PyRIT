# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import List, Literal, Optional
from pathlib import Path

from datasets import load_dataset
from pyrit.datasets.dataset_helper import fetch_examples
from pyrit.models import SeedPromptDataset
from pyrit.models import SeedPrompt, SeedPromptDataset

logger = logging.getLogger(__name__)

HarmLiteral = Literal[
                "Unethical Behavior", 
                "Economic Harm", 
                "Hate Speech", 
                "Government Decision", 
                "Physical Harm", 
                "Fraud", 
                "Political Sensitivity", 
                "Malware", 
                "Illegal Activity", 
                "Bias", 
                "Violence", 
                "Animal Abuse", 
                "Tailored Unlicensed Advice", 
                "Privacy Violation", 
                "Health Consultation", 
                "Child Abuse Content"
            ]

def fetch_jailbreakv_28k_dataset(
    *,
    cache: bool = True,
    data_home: Optional[str] = None,
    split: Literal["JailBreakV_28K", "mini_JailBreakV_28K"] = "mini_JailBreakV_28K",
    text_field: Literal["jailbreak_query", "redteam_query"] = "redteam_query",
    harm_categories: Optional[List[HarmLiteral]] = None,
) -> SeedPromptDataset:
    """
    Fetch examples from the JailBreakV 28k Dataset with optional filtering and create a SeedPromptDataset.

    Args:
        cache (bool): Whether to cache the fetched examples. Defaults to True.
        data_home: Directory used as cache_dir in call to HF to store cached data. Defaults to None.
        subset (str): The subset of the dataset to fetch. Defaults to "JailBreakV_28K".
            Options are "JailBreakV_28K" and "RedTeam_2K".
        split (str): The split of the dataset to fetch. Defaults to "mini_JailBreakV_28K".
            Options are "JailBreakV_28K" and "mini_JailBreakV_28K".
        text_field (str): The field to use as the prompt text. Defaults to "redteam_query".
        harm_categories: List of harm categories to filter the examples.
            Defaults to None, which means all categories are included.
            Otherwise, only prompts with at least one matching category are included.

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the filtered examples.

    Note:
        For more information and access to the original dataset and related materials, visit:
        https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k/blob/main/README.md \n
        Related paper: https://arxiv.org/abs/2404.03027 \n
        The dataset license: mit

    Warning:
        Due to the nature of these prompts, it may be advisable to consult your relevant legal
        department before testing them with LLMs to ensure compliance and reduce potential risks.
    """

    source = "JailbreakV-28K/JailBreakV-28k"

    try:
        logger.info(f"Loading JailBreakV-28k dataset from {source}")

        # Normalize the harm categories to match pyrit harm category conventions
        harm_categories_normalized = None if not harm_categories else [
            _normalize_policy(policy) for policy in harm_categories
        ]

        # Load the dataset from HuggingFace
        data = load_dataset(
            source,
            "JailBreakV_28K",
            cache_dir=data_home
        )

        dataset_split = data[split]

        seed_prompts = []

        # Define common metadata that will be used across all seed prompts
        common_metadata = {
            "dataset_name": "JailbreakV-28K",
            "authors": ["Weidi Luo", "Siyuan Ma", "Xiaogeng Liu", "Chaowei Xiao"],
            "description": (
                "A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks. "
            ),
            "source": source,
            "data_type": "text",
            "name": "JailBreakV-28K",
        }

        for item in dataset_split:
            policy = _normalize_policy(item.get("policy", ""))
            # Skip if user requested policy filter and items policy does not match
            if harm_categories_normalized and policy not in harm_categories_normalized:
                continue
            seed_prompt = SeedPrompt(
                value = item.get(text_field, ""),
                harm_categories=[policy],
                **common_metadata
            )
            seed_prompts.append(seed_prompt)
        seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
        return seed_prompt_dataset

    except Exception as e:
        logger.error(f"Failed to load JailBreakV-28K dataset: {str(e)}")
        raise Exception(f"Error loading JailBreakV-28K dataset: {str(e)}")


def _normalize_policy(policy: str) -> str:
    """Create a machine-friendly variant alongside the human-readable policy."""
    return policy.strip().lower().replace(" ", "_").replace("-", "_")