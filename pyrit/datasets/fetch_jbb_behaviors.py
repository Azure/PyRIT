# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Optional

from datasets import load_dataset

from pyrit.models import SeedPrompt, SeedPromptDataset

logger = logging.getLogger(__name__)


def fetch_jbb_behaviors_dataset(
    source: str = "JailbreakBench/JBB-Behaviors",
    data_home: Optional[str] = None,
) -> SeedPromptDataset:
    """
    Fetch the JailbreakBench JBB-Behaviors dataset from HuggingFace and create a SeedPromptDataset.

    This dataset contains harmful behaviors for jailbreaking evaluation,
    as described in the paper: https://arxiv.org/abs/2404.01318

    Args:
        source (str): The HuggingFace dataset identifier. Defaults to "JailbreakBench/JBB-Behaviors".
        data_home (str, optional): The directory to cache the dataset. If None, uses default cache.

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the JBB behaviors with harm_categories set.

    Raises:
        Exception: If the dataset cannot be loaded or processed.

    Note:
        Content Warning: This dataset contains prompts aimed at provoking harmful responses
        and may contain offensive content. Users should check with their legal department
        before using these prompts against production LLMs.

        For more information and access to the original dataset and related materials, visit:
        https://arxiv.org/abs/2404.01318
    """
    try:
        logger.info(f"Loading JBB-Behaviors dataset from {source}")

        # Load the dataset from HuggingFace
        data = load_dataset(source, cache_dir=data_home)

        # Get the appropriate split (typically 'train' or the first available split)
        dataset_split = data["train"] if "train" in data else data[list(data.keys())[0]]

        # Define common metadata that will be used across all seed prompts
        common_metadata = {
            "dataset_name": "JailbreakBench JBB-Behaviors",
            "authors": ["JailbreakBench Team"],
            "description": (
                "A dataset of harmful behaviors for jailbreaking evaluation from JailbreakBench. "
                "Contains behaviors designed to test AI safety measures."
            ),
            "source": source,
            "data_type": "text",
            "name": "JBB-Behaviors",
        }

        seed_prompts = []

        for item in dataset_split:
            # Extract the required fields
            behavior = item.get("behavior", "").strip()
            category = item.get("category", "")

            if not behavior:
                logger.warning("Skipping item with empty behavior")
                continue

            # Map JBB categories to PyRIT harm categories
            harm_categories = _map_jbb_category_to_harm_category(category)

            # Create SeedPrompt object with all metadata
            seed_prompt = SeedPrompt(
                value=behavior,
                harm_categories=harm_categories,
                groups=[category] if category else [],
                metadata={
                    "jbb_category": category,
                    "original_source": "JailbreakBench",
                },
                **common_metadata,  # type: ignore[arg-type]
            )

            seed_prompts.append(seed_prompt)

        if not seed_prompts:
            raise ValueError("SeedPromptDataset cannot be empty.")

        logger.info(f"Successfully loaded {len(seed_prompts)} behaviors from JBB-Behaviors dataset")

        # Create and return SeedPromptDataset
        seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
        return seed_prompt_dataset

    except Exception as e:
        logger.error(f"Failed to load JBB-Behaviors dataset: {str(e)}")
        raise Exception(f"Error loading JBB-Behaviors dataset: {str(e)}")


def _map_jbb_category_to_harm_category(jbb_category: str) -> list[str]:
    """
    Maps JBB categories to PyRIT harm categories.

    Args:
        jbb_category (str): The category from the JBB dataset.

    Returns:
        list[str]: List of corresponding PyRIT harm categories.
    """
    # Based on common harm taxonomies and JailbreakBench paper categories
    category_mapping = {
        # Violence and harm
        "violence": ["violence"],
        "physical_harm": ["violence", "physical_harm"],
        "weapons": ["violence", "weapons"],
        "terrorism": ["violence", "terrorism"],
        # Hate and harassment
        "hate": ["hate"],
        "harassment": ["harassment"],
        "bullying": ["harassment", "bullying"],
        "discrimination": ["discrimination"],
        "bias": ["bias", "discrimination"],
        # Sexual content
        "sexual": ["sexual"],
        "adult_content": ["sexual"],
        "non_consensual": ["sexual", "non_consensual"],
        # Self-harm and suicide
        "self_harm": ["self_harm"],
        "suicide": ["self_harm", "suicide"],
        "eating_disorders": ["self_harm", "eating_disorders"],
        # Criminal activities
        "criminal": ["criminal_planning"],
        "illegal": ["illegal_activity"],
        "fraud": ["criminal_planning", "fraud"],
        "theft": ["criminal_planning", "theft"],
        "cybercrime": ["criminal_planning", "cybercrime"],
        # Privacy and personal information
        "privacy": ["privacy"],
        "personal_info": ["privacy", "personal_information"],
        "doxxing": ["privacy", "doxxing"],
        # Deception and misinformation
        "deception": ["deception"],
        "misinformation": ["misinformation"],
        "conspiracy": ["misinformation", "conspiracy"],
        "fake_news": ["misinformation"],
        # Unethical behavior
        "unethical": ["unethical"],
        "manipulation": ["unethical", "manipulation"],
        "exploitation": ["unethical", "exploitation"],
        # Substance abuse
        "drugs": ["substance_abuse"],
        "alcohol": ["substance_abuse"],
        # Financial harm
        "financial_harm": ["financial_harm"],
        "scams": ["financial_harm", "scams"],
        # Child safety
        "child_abuse": ["child_safety"],
        "grooming": ["child_safety", "grooming"],
        # Government and political
        "government": ["government"],
        "political": ["political"],
        "election": ["political", "election"],
    }

    if not jbb_category or not jbb_category.strip():
        return ["unknown"]

    # Convert to lowercase for case-insensitive matching
    jbb_category_lower = jbb_category.lower().strip()

    # Try exact match first
    if jbb_category_lower in category_mapping:
        return category_mapping[jbb_category_lower]

    # Try partial matching for compound categories
    matched_categories = set()
    for key, harm_cats in category_mapping.items():
        if key in jbb_category_lower or jbb_category_lower in key:
            matched_categories.update(harm_cats)

    if matched_categories:
        return list(matched_categories)

    # Special handling for common patterns
    if any(term in jbb_category_lower for term in ["violent", "kill", "murder", "bomb"]):
        return ["violence"]
    elif any(term in jbb_category_lower for term in ["hate", "racist", "sexist"]):
        return ["hate", "discrimination"]
    elif any(term in jbb_category_lower for term in ["sexual", "porn", "nsfw"]):
        return ["sexual"]
    elif any(term in jbb_category_lower for term in ["illegal", "crime", "criminal"]):
        return ["criminal_planning", "illegal_activity"]
    elif any(term in jbb_category_lower for term in ["harm", "hurt", "damage"]):
        return ["violence", "harm"]

    # Default fallback - log unknown categories for future mapping
    logger.warning(f"Unknown JBB category '{jbb_category}', using default harm category")
    return ["unknown"]


def fetch_jbb_behaviors_by_harm_category(harm_category: str, **kwargs) -> SeedPromptDataset:
    """
    Fetch JBB-Behaviors filtered by a specific harm category.

    Args:
        harm_category (str): The harm category to filter by.
        **kwargs: Additional arguments passed to fetch_jbb_behaviors_dataset.

    Returns:
        SeedPromptDataset: Filtered SeedPromptDataset containing only prompts with the specified harm category.
    """
    # Get all prompts
    all_dataset = fetch_jbb_behaviors_dataset(**kwargs)

    # Filter by harm category (case-insensitive)
    filtered_prompts = [
        prompt
        for prompt in all_dataset.prompts
        if any(harm_category.lower() in harm_cat.lower() for harm_cat in prompt.harm_categories)
    ]

    if not filtered_prompts:
        raise ValueError("SeedPromptDataset cannot be empty.")

    logger.info(f"Filtered {len(filtered_prompts)} prompts for harm category '{harm_category}'")

    # Create new dataset with filtered prompts
    filtered_dataset = SeedPromptDataset(prompts=filtered_prompts)
    return filtered_dataset


def fetch_jbb_behaviors_by_jbb_category(jbb_category: str, **kwargs) -> SeedPromptDataset:
    """
    Fetch JBB-Behaviors filtered by the original JBB category.

    Args:
        jbb_category (str): The original JBB category to filter by.
        **kwargs: Additional arguments passed to fetch_jbb_behaviors_dataset.

    Returns:
        SeedPromptDataset: Filtered SeedPromptDataset containing only prompts with the specified JBB category.
    """
    # Get all prompts
    all_dataset = fetch_jbb_behaviors_dataset(**kwargs)

    # Filter by JBB category (case-insensitive)
    filtered_prompts = [
        prompt
        for prompt in all_dataset.prompts
        if (
            prompt.metadata
            and str(prompt.metadata.get("jbb_category", "")).lower() == jbb_category.lower()
        )
    ]

    if not filtered_prompts:
        raise ValueError("SeedPromptDataset cannot be empty.")

    logger.info(f"Filtered {len(filtered_prompts)} prompts for JBB category '{jbb_category}'")

    # Create new dataset with filtered prompts
    filtered_dataset = SeedPromptDataset(prompts=filtered_prompts)
    return filtered_dataset