# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib
import uuid
import zipfile
from typing import Dict, List, Literal, Optional

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

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
    "Child Abuse Content",
]


def fetch_jailbreakv_28k_dataset(
    *,
    data_home: Optional[str] = None,
    zip_dir: Optional[str] = None,
    split: Literal["JailBreakV_28K", "mini_JailBreakV_28K"] = "mini_JailBreakV_28K",
    text_field: Literal["jailbreak_query", "redteam_query"] = "redteam_query",
    harm_categories: Optional[List[HarmLiteral]] = None,
    min_prompts: int = 50,
) -> SeedPromptDataset:
    """
    Fetch examples from the JailBreakV 28k Dataset with optional filtering and create a SeedPromptDataset.
    Many images are missing from the dataset in HF and the team host the full image file in Google Drive.
    Prioritizes the HF dataset and falls back to a cached download of Google Drive contents.
    As of 10/2025 the HF dataset is missing most images, backup path should be provided.
    Please download the zip file from the HF owners Google Drive at this share link:
    https://drive.google.com/file/d/1ZrvSHklXiGYhpiVoxUH8FWc5k0fv2xVZ/view
    Note that the file is 15 GB compressed due to images not compressing well.
    To reduce disk usage after extracting the first time it is fine to delete the zip.

    Args:
        data_home: Directory used as cache_dir in call to HF to store cached data. Defaults to None.
            If None, the default cache directory will be used.
        zip_dir (str): The directory containing the zip file. Defaults to None.
            As of 10/2025 the HF dataset is missing most images, backup path should be provided.
        split (str): The split of the dataset to fetch. Defaults to "mini_JailBreakV_28K".
            Options are "JailBreakV_28K" and "mini_JailBreakV_28K".
        text_field (str): The field to use as the prompt text. Defaults to "redteam_query".
            Options are "jailbreak_query" and "redteam_query".
        harm_categories: List of harm categories to filter the examples.
            Defaults to None, which means all categories are included.
            Otherwise, only prompts with at least one matching category are included.
        min_prompts (int): The minimum number of prompts to return. Defaults to 50.
            If the number of prompts after filtering is less than this value, an error is raised.

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the filtered examples.

    Note:
        For more information and access to the original dataset and related materials, visit:
        https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k/blob/main/README.md \n
        Related paper: https://arxiv.org/abs/2404.03027 \n
        The dataset license: MIT
        Authors: Weidi Luo, Siyuan Ma, Xiaogeng Liu, Chaowei Xiao, Xiaoyu Guo

    Warning:
        Due to the nature of these prompts, it may be advisable to consult your relevant legal
        department before testing them with LLMs to ensure compliance and reduce potential risks.
    """

    source = "JailbreakV-28K/JailBreakV-28k"

    # Unzip the file if it is not already extracted
    zip_extracted_path = None
    if zip_dir:
        zip_file_path = pathlib.Path(zip_dir) / "JailBreakV_28K.zip"
        zip_extracted_path = pathlib.Path(zip_dir) / "JailBreakV_28K"
        # Check if the zip file exists
        if not zip_file_path.exists():
            raise FileNotFoundError("No zip file provided for JailBreakV-28K dataset. Many images likely missing.")
        # Only unzip if the target directory does not already exist
        if not zip_extracted_path.exists():
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(pathlib.Path(zip_dir))
    else:
        logger.warning("No zip file provided for JailBreakV-28K dataset. Many images likely missing.")

    try:
        logger.info(f"Loading JailBreakV-28k dataset from {source}")

        # Normalize the harm categories to match pyrit harm category conventions
        harm_categories_normalized = (
            None if not harm_categories else [_normalize_policy(policy) for policy in harm_categories]
        )

        # Load the dataset from HuggingFace
        data = load_dataset(source, "JailBreakV_28K", cache_dir=data_home)

        dataset_split = data[split]

        per_call_cache: Dict[str, str] = {}

        seed_prompts = []

        # Define common metadata that will be used across all seed prompts
        common_metadata = {
            "dataset_name": "JailbreakV-28K",
            "authors": ["Weidi Luo", "Siyuan Ma", "Xiaogeng Liu", "Chaowei Xiao", "Xiaoyu Guo"],
            "description": (
                "Benchmark for Assessing the Robustness of "
                "Multimodal Large Language Models against Jailbreak Attacks. "
            ),
            "groups": ["The Ohio State University", "Peking University", "University of Wisconsin-Madison"],
            "source": "https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k",
            "name": "JailBreakV-28K",
        }

        # tracker for items in the dataset where image_path does not match an image in the repo
        missing_images = 0

        for item in dataset_split:
            policy = _normalize_policy(item.get("policy", ""))
            # Skip if user requested policy filter and items policy does not match
            if not (harm_categories_normalized) or policy in harm_categories_normalized:
                image_rel_path = item.get("image_path", "")
                image_abs_path = ""
                if image_rel_path:
                    image_abs_path = _resolve_image_path(
                        image_rel_path,
                        repo_id=source,
                        data_home=data_home,
                        backup_root=zip_extracted_path,
                        call_cache=per_call_cache,
                    )
                if not image_abs_path:
                    missing_images += 1
                    continue

                group_id = uuid.uuid4()
                text_seed_prompt = SeedPrompt(
                    value=item.get(text_field, ""),
                    harm_categories=[policy],
                    prompt_group_id=group_id,
                    data_type="text",
                    **common_metadata,  # type: ignore[arg-type]
                )
                image_seed_prompt = SeedPrompt(
                    value=image_abs_path,
                    harm_categories=[policy],
                    prompt_group_id=group_id,
                    data_type="image_path",
                    **common_metadata,  # type: ignore[arg-type]
                )
                seed_prompts.append(text_seed_prompt)
                seed_prompts.append(image_seed_prompt)

    except Exception as e:
        logger.error(f"Failed to load JailBreakV-28K dataset: {str(e)}")
        raise Exception(f"Error loading JailBreakV-28K dataset: {str(e)}")
    if len(seed_prompts) < min_prompts:
        raise ValueError(
            f"JailBreakV-28K fetch produced {missing_images} missing images. "
            f"Only {len(seed_prompts)} multimodal prompts were produced. "
            f"This is below the minimum required prompts of {min_prompts}. "
            f"Please ensure the zip_dir parameter is provided with the full image set or "
            f"check your backup image source."
        )
    elif missing_images > 0:
        logger.warning(f"Failed to resolve {missing_images} image paths in JailBreakV-28K dataset")
    if not seed_prompts:
        raise ValueError(
            "JailBreakV-28K fetch produced 0 prompts. "
            "Likely caused by all items returned after filtering having invalid image paths."
        )
    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset


def _normalize_policy(policy: str) -> str:
    """Create a machine-friendly variant alongside the human-readable policy."""
    return policy.strip().lower().replace(" ", "_").replace("-", "_")


def _resolve_image_path(
    rel_path: str,
    repo_id: str,
    data_home: Optional[str],
    backup_root: Optional[pathlib.Path] = None,
    call_cache: Dict[str, str] = {},
) -> str:
    """
    Resolve a repo-relative image path to a local absolute path using hf_hub_download.
    Uses a cache (module-level by default) to avoid re-downloading the same file.

    Args:
        rel_path: path relative to the dataset repository root (e.g., "images/0001.png").
        repo_id: HF dataset repo id, e.g., "JailbreakV-28K/JailBreakV-28k".
        data_home: optional cache directory.
        backup_root: optional path to a directory containing the zip file.
            If provided, will be used as a backup source for images not found in the HF dataset.
        cache: optional dict to use instead of the module-level cache.

    Returns:
        Absolute local path if resolved, else None (and caches the miss).
    """
    if not rel_path:
        return ""

    # check if image has already been cached
    if rel_path in call_cache:
        return call_cache[rel_path]
    path_root = "JailBreakV_28K"
    hf_path = f"{path_root}/{rel_path}"
    backup_path = None if not backup_root else backup_root / hf_path
    try:
        # first check if the path exists using HFApi()
        repo_file_list = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset")
        if hf_path in repo_file_list:
            # download the image
            abs_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=hf_path,
                cache_dir=data_home,
            )
        else:
            logger.debug(f"File {hf_path} not found in dataset {repo_id}, trying backup path {backup_path}")
            if backup_path and backup_path.exists():
                abs_path = str(backup_path)
            else:
                logger.debug(f"File {hf_path} not found in dataset {repo_id} or backup path {backup_path}")
                abs_path = ""

        call_cache[rel_path] = abs_path
        return abs_path
    except Exception as e:
        logger.error(f"Failed to download image {rel_path}: {str(e)}")
        call_cache[rel_path] = ""
        return ""
