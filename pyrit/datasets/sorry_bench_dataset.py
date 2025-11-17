# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
from typing import List, Optional

from datasets import load_dataset

from pyrit.models import SeedDataset, SeedPrompt

logger = logging.getLogger(__name__)


VALID_CATEGORIES = [
    "Personal Insulting Words",
    "Social-group Insulting Words",
    "Threatening Words",
    "Lewd and Obscene Words",
    "Libelous Words",
    "Self-Harm",
    "Violent Crimes",
    "Harassment",
    "Sexual Crimes",
    "Property Crimes",
    "Public Order Crimes",
    "System Intrusion",
    "Malware Code Generation",
    "Fraud",
    "Financial Crimes",
    "IP Infringement",
    "PII Violations",
    "Illegal Crafting",
    "Terrorism",
    "Child-related Crimes",
    "Animal-related Crimes",
    "Environmental Crimes",
    "Evasion of Law Enforcement",
    "Human Elimination",
    "Advice on Adult Content",
    "Sexual Explicit Content Generation",
    "Non-sexual Explicit Content Generation",
    "Fake News Generation",
    "False Advertising",
    "Advice on Discrimination Behaviors",
    "Military Use",
    "Political Belief Promotion",
    "Ethical Belief Promotion",
    "Religion Promotion",
    "Social Stereotype Promotion",
    "Extremist Content Promotion",
    "Conspiracy Theories Promotion",
    "False Common Knowledge Promotion",
    "Harmful & Unhealthy Behaviors Promotion",
    "Medical Advice",
    "Financial Advice",
    "Legal Consulting Advice",
    "Governance Decision Advice",
    "Dangerous Machinery Operation Advice",
]

VALID_PROMPT_STYLES = [
    "base",
    "ascii",
    "atbash",
    "authority_endorsement",
    "caesar",
    "evidence-based_persuasion",
    "expert_endorsement",
    "logical_appeal",
    "misrepresentation",
    "misspellings",
    "morse",
    "question",
    "role_play",
    "slang",
    "technical_terms",
    "translate-fr",
    "translate-ml",
    "translate-mr",
    "translate-ta",
    "translate-zh-cn",
    "uncommon_dialects",
]


def fetch_sorry_bench_dataset(
    *,
    cache_dir: Optional[str] = None,
    categories: Optional[List[str]] = None,
    prompt_style: Optional[str] = None,
    token: Optional[str] = None,
) -> SeedDataset:
    """
    Fetch Sorry-Bench dataset from Hugging Face (updated 2025/03 version).

    The Sorry-Bench dataset contains adversarial prompts designed to test LLM safety
    across 44 categories with 21 different prompt styles (base + 20 linguistic mutations).

    Reference: https://arxiv.org/abs/2406.14598

    Args:
        cache_dir (Optional[str]): Optional cache directory for Hugging Face datasets.
        categories (Optional[List[str]]): Optional list of categories to filter. Full list in:
            https://huggingface.co/datasets/sorry-bench/sorry-bench-202503/blob/main/meta_info.py
        prompt_style (Optional[str]): Optional prompt style to filter. Available styles:
            "base", "ascii", "caesar", "slang", "authority_endorsement", etc.
            Default: "base" (only base prompts, no mutations)
            Full list: https://huggingface.co/datasets/sorry-bench/sorry-bench-202503
        token (Optional[str]): Hugging Face authentication token. If not provided,
            will attempt to read from HUGGINGFACE_TOKEN environment variable. This is needed for
            accessing gated datasets on Hugging Face.

    Returns:
        SeedDataset: SeedDataset containing Sorry-Bench prompts with harm categories.

    Raises:
        ValueError: If invalid categories or prompt_style are provided.
    """
    if prompt_style is None:
        prompt_style = "base"

    if token is None:
        token = os.environ.get("HUGGINGFACE_TOKEN")

    if prompt_style not in VALID_PROMPT_STYLES:
        raise ValueError(f"Invalid prompt_style '{prompt_style}'. Must be one of: {', '.join(VALID_PROMPT_STYLES)}")

    if categories:
        invalid_categories = [cat for cat in categories if cat not in VALID_CATEGORIES]
        if invalid_categories:
            raise ValueError(
                f"Invalid categories: {invalid_categories}. Must be from the list of 44 valid categories. "
                f"See: https://huggingface.co/datasets/sorry-bench/sorry-bench-202503/blob/main/meta_info.py"
            )

    try:
        source = "sorry-bench/sorry-bench-202503"
        logger.info(f"Loading Sorry-Bench dataset from {source}")
        data = load_dataset(source, cache_dir=cache_dir, token=token)

        dataset_split = data["train"]

        common_metadata = {
            "dataset_name": "Sorry-Bench",
            "authors": ["Sorry-Bench Team"],
            "description": "Adversarial prompts for testing LLM safety across 44 categories",
            "source": source,
            "data_type": "text",
            "name": "Sorry-Bench 2025-03",
        }

        seed_prompts = []

        for item in dataset_split:
            category = item.get("category", "")
            item_prompt_style = item.get("prompt_style", "")
            turns = [t for t in item.get("turns", []) if t is not None]
            question_id = item.get("question_id")

            if not turns:
                logger.warning(f"Skipping item {question_id} with empty turns")
                continue

            prompt_text = turns[0].strip()

            if not prompt_text:
                logger.warning(f"Skipping item {question_id} with empty prompt text")
                continue

            if categories and category not in categories:
                continue

            if prompt_style != item_prompt_style:
                continue

            seed_prompt = SeedPrompt(
                value=prompt_text,
                harm_categories=[category],
                groups=[item_prompt_style] if item_prompt_style else [],
                metadata={
                    "sorry_bench_category": category,
                    "prompt_style": item_prompt_style,
                    "question_id": question_id,
                },
                **common_metadata,  # type: ignore[arg-type]
            )

            seed_prompts.append(seed_prompt)

        if not seed_prompts:
            raise ValueError("SeedDataset cannot be empty. Check your filter criteria.")

        logger.info(f"Successfully loaded {len(seed_prompts)} prompts from Sorry-Bench dataset")

        return SeedDataset(prompts=seed_prompts)

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to load Sorry-Bench dataset: {str(e)}")
        raise
