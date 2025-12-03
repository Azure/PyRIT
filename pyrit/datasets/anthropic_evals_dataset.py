# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import requests

from pyrit.datasets.dataset_helper import fetch_examples
from pyrit.models import (
    QuestionAnsweringDataset,
    QuestionAnsweringEntry,
    QuestionChoice,
)

logger = logging.getLogger(__name__)

VALID_CATEGORIES = ["persona", "sycophancy", "advanced-ai-risk", "winogenerated"]


def _parse_answer_choices(item: Dict) -> tuple[List[QuestionChoice], int]:
    """
    Parse answer choices from Anthropic eval item format.

    Returns:
        Tuple of (choices, correct_answer_index) where correct_answer_index points to the matching behavior.
    """
    choices = []
    correct_index = 0

    if "pronoun_options" in item:
        pronoun_options = item.get("pronoun_options", [])
        for idx, pronoun in enumerate(pronoun_options):
            choices.append(QuestionChoice(index=idx, text=pronoun))

        matching_behavior = item.get("answer_matching_behavior", "")
        match = re.search(r"\([A-Z]\)\s*(.+)", matching_behavior)
        if match:
            matching_pronoun = match.group(1).strip()
            try:
                correct_index = pronoun_options.index(matching_pronoun)
            except ValueError:
                correct_index = 0
        return choices, correct_index

    matching = item.get("answer_matching_behavior", "")
    not_matching = item.get("answer_not_matching_behavior", "")

    all_answers = []
    if matching:
        all_answers.append((0, matching))
    if not_matching:
        all_answers.append((1, not_matching))

    def extract_letter(answer_text: str) -> str:
        match = re.match(r"\(([A-Z])\)", answer_text.strip())
        return match.group(1) if match else "Z"

    all_answers_sorted = sorted(all_answers, key=lambda x: extract_letter(x[1]))

    for idx, (original_idx, answer_text) in enumerate(all_answers_sorted):
        choices.append(QuestionChoice(index=idx, text=answer_text.strip()))
        if original_idx == 0:
            correct_index = idx

    return choices, correct_index


def fetch_anthropic_evals_dataset(
    category: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> QuestionAnsweringDataset:
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
        QuestionAnsweringDataset containing Anthropic eval questions with structured choices.

    Raises:
        ValueError: If invalid category is provided or dataset is empty.
    """
    if category is not None and category not in VALID_CATEGORIES:
        raise ValueError(f"Invalid category '{category}'. Must be one of: {', '.join(VALID_CATEGORIES)}")

    categories_to_fetch = [category] if category else VALID_CATEGORIES

    try:
        logger.info("Loading Anthropic model-written-evals dataset")

        question_entries = []

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
                        question_text = (item.get("question") or item.get("sentence_with_blank", "")).strip()
                        if not question_text:
                            logger.warning(f"Skipping item with empty prompt in {file_name}")
                            continue

                        choices, correct_index = _parse_answer_choices(item)

                        if not choices:
                            logger.warning(f"Skipping item with no valid choices in {file_name}")
                            continue

                        entry = QuestionAnsweringEntry(
                            question=question_text,
                            answer_type="int",
                            correct_answer=correct_index,
                            choices=choices,
                        )
                        question_entries.append(entry)

        if not question_entries:
            raise ValueError("QuestionAnsweringDataset cannot be empty. Check your filter criteria.")

        logger.info(f"Successfully loaded {len(question_entries)} questions from Anthropic evals dataset")

        dataset = QuestionAnsweringDataset(
            name="anthropic-evals",
            version="1.0",
            description="""Evaluations for testing LLM behaviors across persona, sycophancy, AI risk, and bias.
            The model-written-evals dataset from Anthropic contains 154+ evaluation datasets across 4 categories.""",
            author="Anthropic",
            group=category if category else "all",
            source="https://github.com/anthropics/evals",
            questions=question_entries,
        )

        return dataset

    except Exception as e:
        logger.error(f"Failed to load Anthropic evals dataset: {str(e)}")
        raise


def _get_category_files(category: str) -> List[Dict[str, str]]:
    """
    Get list of .jsonl files in a category directory from GitHub API, including subdirectories.

    Returns:
        List of file info dictionaries with 'name' and 'download_url' keys.

    Raises:
        Exception: If GitHub API request fails.
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
