# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import List, Optional

from datasets import load_dataset

from pyrit.models import SeedDataset, SeedPrompt

logger = logging.getLogger(__name__)

def fetch_sorry_bench_dataset(
    cache_dir: Optional[str] = None,
    categories: Optional[List[str]] = None,
    prompt_styles: Optional[List[str]] = None,
) -> SeedDataset:
    """
    Fetch Sorry-Bench dataset from Hugging Face (updated 2025/03 version).

    The Sorry-Bench dataset contains adversarial prompts designed to test LLM safety
    across 44 categories with 21 different prompt styles (base + 20 linguistic mutations).

    Reference: https://arxiv.org/abs/2406.14598

    Args:
        cache_dir: Optional cache directory for Hugging Face datasets
        categories: Optional list of categories to filter. Full list in:
            https://huggingface.co/datasets/sorry-bench/sorry-bench-202503/blob/main/meta_info.py
        prompt_styles: Optional list of prompt styles to include. List of all linguistic mutations in:
            https://huggingface.co/datasets/sorry-bench/sorry-bench-202503
            Available: "base", "ascii", "caesar", "slang", "authority_endorsement", etc.
            Default: ["base"] (only base prompts, no mutations). Pass empty list [] for all styles
    
    Returns:
        SeedDataset containing Sorry-Bench prompts with harm categories.
    """
    try:
        # Default to base prompts only if not specified
        if prompt_styles is None:
            prompt_styles = ["base"]

        # Load dataset from HuggingFace
        source = "sorry-bench/sorry-bench-202503"
        logger.info(f"Loading Sorry-Bench dataset from {source}")
        data = load_dataset(source, cache_dir=cache_dir)

        # Get the train split
        dataset_split = data["train"]

        # Common metadata for all prompts
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
            prompt_style = item.get("prompt_style", "")
            turns = item.get("turns", [])
            question_id = item.get("question_id")

            # Validate required fields
            if not turns or len(turns) == 0:
                logger.warning(f"Skipping item {question_id} with empty turns")
                continue

            # Extract the actual prompt text (first item in turns list)
            prompt_text = turns[0].strip()

            if not prompt_text:
                logger.warning(f"Skipping item {question_id} with empty prompt text")
                continue

            # Filter by categories if specified
            if categories and category not in categories:
                continue

            # Filter by prompt_styles if specified (empty list means include all)
            if prompt_styles and prompt_style not in prompt_styles:
                continue

            # Create SeedPrompt with original Sorry-Bench category 
            seed_prompt = SeedPrompt(
                value=prompt_text,
                harm_categories=[category],
                groups=[prompt_style] if prompt_style else [],
                metadata={
                    "sorry_bench_category": category,
                    "prompt_style": prompt_style,
                    "question_id": question_id,
                },
                **common_metadata,  # type: ignore[arg-type]
            )

            seed_prompts.append(seed_prompt)

        if not seed_prompts:
            raise ValueError("SeedDataset cannot be empty. Check your filter criteria.")

        logger.info(f"Successfully loaded {len(seed_prompts)} prompts from Sorry-Bench dataset")

        return SeedDataset(prompts=seed_prompts)

    except Exception as e:
        logger.error(f"Failed to load Sorry-Bench dataset: {str(e)}")
        raise Exception(f"Error loading Sorry-Bench dataset: {str(e)}")
