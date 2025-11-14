# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from pathlib import Path
from typing import Dict, List, Optional

import requests

from pyrit.datasets.dataset_helper import fetch_examples
from pyrit.models import SeedDataset, SeedPrompt

logger = logging.getLogger(__name__)

VALID_CATEGORIES = ["persona", "sycophancy", "advanced-ai-risk", "winogenerated"]


def fetch_anthropic_evals_dataset(
    category: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> SeedDataset:
    """
    Fetch Anthropic's model-written-evals dataset from GitHub.

    The model-written-evals dataset contains evaluations for testing LLM behaviors
    across 4 main categories:
    - persona: 119 datasets testing personality traits, beliefs, and potentially dangerous goals
    - advanced-ai-risk: 32 datasets (16 human + 16 LM-generated) testing AI safety concerns
    - sycophancy: 3 datasets testing whether models repeat back user views
    - winogenerated: 2 datasets testing gender bias in occupations

    Reference: https://arxiv.org/abs/2212.09251
    GitHub: https://github.com/anthropics/evals

    Args:
        category: Optional category to filter. Available categories:
            "persona", "sycophancy", "advanced-ai-risk", "winogenerated"
            Default: None (returns all categories)
        cache_dir: Optional cache directory for downloaded files

    Returns:
        SeedDataset containing Anthropic eval questions with metadata.

    Raises:
        ValueError: If invalid category is provided or dataset is empty.
    """
    if category is not None and category not in VALID_CATEGORIES:
        raise ValueError(f"Invalid category '{category}'. Must be one of: {', '.join(VALID_CATEGORIES)}")

    categories_to_fetch = [category] if category else VALID_CATEGORIES

    try:
        logger.info("Loading Anthropic model-written-evals dataset")

        common_metadata = {
            "dataset_name": "Anthropic Model-Written Evals",
            "authors": ["Anthropic"],
            "description": "Evaluations for testing LLM behaviors across persona, sycophancy, AI risk, and bias",
            "source": "https://github.com/anthropics/evals",
            "data_type": "text",
            "name": "Anthropic Evals",
        }

        seed_prompts = []

        for cat in categories_to_fetch:
            logger.info(f"Fetching files from category: {cat}")
            files = _get_category_files(cat)

            for file_info in files:
                file_name = file_info["name"]
                if file_name.endswith(".jsonl"):
                    file_url = file_info["download_url"]
                    eval_name = file_name.removesuffix(".jsonl")

                    logger.info(f"Loading evaluation: {eval_name}")

                    examples = fetch_examples(
                        source=file_url,
                        source_type="public_url",
                        cache=True,
                        data_home=Path(cache_dir) if cache_dir else None,
                    )

                    for item in examples:
                        prompt_value = (item.get("question") or item.get("sentence_with_blank", "")).strip()
                        if not prompt_value:
                            logger.warning(f"Skipping item with empty prompt in {file_name}")
                            continue

                        seed_prompt = SeedPrompt(
                            value=prompt_value,
                            harm_categories=[],
                            groups=[eval_name],
                            metadata={
                                "category": cat,
                                "answer_matching_behavior": item.get("answer_matching_behavior", ""),
                                "answer_not_matching_behavior": item.get("answer_not_matching_behavior", ""),
                            },
                            **common_metadata,  # type: ignore[arg-type]
                        )
                        seed_prompts.append(seed_prompt)

        if not seed_prompts:
            raise ValueError("SeedDataset cannot be empty. Check your filter criteria.")

        logger.info(f"Successfully loaded {len(seed_prompts)} prompts from Anthropic evals dataset")

        return SeedDataset(prompts=seed_prompts)

    except Exception as e:
        logger.error(f"Failed to load Anthropic evals dataset: {str(e)}")
        raise


def _get_category_files(category: str) -> List[Dict[str, str]]:
    """
    Get list of files in a category directory from GitHub API.
    Recursively searches subdirectories for .jsonl files.

    Args:
        category: The category directory name

    Returns:
        List of file info dictionaries with 'name' and 'download_url' keys
    """
    api_url = f"https://api.github.com/repos/anthropics/evals/contents/{category}"

    response = requests.get(api_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch file list for category '{category}'. Status code: {response.status_code}")

    files = []
    for item in response.json():
        if item["type"] == "file" and item["name"].endswith(".jsonl"):
            files.append({"name": item["name"], "download_url": item["download_url"]})
        elif item["type"] == "dir":
            subdir_response = requests.get(item["url"])
            if subdir_response.status_code == 200:
                for subitem in subdir_response.json():
                    if subitem["type"] == "file" and subitem["name"].endswith(".jsonl"):
                        files.append({"name": subitem["name"], "download_url": subitem["download_url"]})
    return files
