# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import hashlib
import io
import random
import tempfile
from pathlib import Path

import pycountry
import requests

from pyrit.common.csv_helper import read_csv, write_csv
from pyrit.common.json_helper import read_json, write_json
from pyrit.common.text_helper import read_txt, write_txt
from pyrit.common.path import RESULTS_PATH
from pyrit.models import PromptDataset

from typing import Callable, Dict, List, Optional, Literal, TextIO


# Define the type for the file handlers
FileHandlerRead = Callable[[TextIO], List[Dict[str, str]]]
FileHandlerWrite = Callable[[TextIO, List[Dict[str, str]]], None]

FILE_TYPE_HANDLERS: Dict[str, Dict[str, Callable]] = {
    "json": {"read": read_json, "write": write_json},
    "csv": {"read": read_csv, "write": write_csv},
    "txt": {"read": read_txt, "write": write_txt},
}


def _get_cache_file_name(source: str, file_type: str) -> str:
    """
    Generate a cache file name based on the source URL and file type.
    """
    hash_source = hashlib.md5(source.encode("utf-8")).hexdigest()
    return f"{hash_source}.{file_type}"


def _read_cache(cache_file: Path, file_type: str) -> List[Dict[str, str]]:
    """
    Read data from cache.
    """
    with cache_file.open("r", encoding="utf-8") as file:
        if file_type in FILE_TYPE_HANDLERS:
            return FILE_TYPE_HANDLERS[file_type]["read"](file)
        else:
            valid_types = ", ".join(FILE_TYPE_HANDLERS.keys())
            raise ValueError(f"Invalid file_type. Expected one of: {valid_types}.")


def _write_cache(cache_file: Path, examples: List[Dict[str, str]], file_type: str):
    """
    Write data to cache.
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w", encoding="utf-8") as file:
        if file_type in FILE_TYPE_HANDLERS:
            FILE_TYPE_HANDLERS[file_type]["write"](file, examples)
        else:
            valid_types = ", ".join(FILE_TYPE_HANDLERS.keys())
            raise ValueError(f"Invalid file_type. Expected one of: {valid_types}.")


def _fetch_from_public_url(source: str, file_type: str) -> List[Dict[str, str]]:
    """
    Fetch examples from a repository.
    """
    response = requests.get(source)
    if response.status_code == 200:
        if file_type in FILE_TYPE_HANDLERS:
            if file_type == "json":
                return FILE_TYPE_HANDLERS[file_type]["read"](io.StringIO(response.text))
            else:
                return FILE_TYPE_HANDLERS[file_type]["read"](
                    io.StringIO("\n".join(response.text.splitlines()))
                )  # noqa: E501
        else:
            valid_types = ", ".join(FILE_TYPE_HANDLERS.keys())
            raise ValueError(f"Invalid file_type. Expected one of: {valid_types}.")
    else:
        raise Exception(f"Failed to fetch examples from public URL. Status code: {response.status_code}")


def _fetch_from_file(source: str, file_type: str) -> List[Dict[str, str]]:
    """
    Fetch examples from a local file.
    """
    with open(source, "r", encoding="utf-8") as file:
        if file_type in FILE_TYPE_HANDLERS:
            return FILE_TYPE_HANDLERS[file_type]["read"](file)
        else:
            valid_types = ", ".join(FILE_TYPE_HANDLERS.keys())
            raise ValueError(f"Invalid file_type. Expected one of: {valid_types}.")


def fetch_examples(
    source: str,
    source_type: Literal["public_url", "file"] = "public_url",
    cache: bool = True,
    data_home: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """
    Fetch examples from a specified source with caching support.

    Args:
        source (str): The source from which to fetch examples.
        source_type (Literal["public_url", "file"]): The type of source ('public_url' or 'file').
        file_type (str): The type of file ('json', 'csv', or 'txt'). Defaults to 'json'.
        cache (bool): Whether to cache the fetched examples. Defaults to True.
        data_home (Optional[Path]): Directory to store cached data. Defaults to None.

    Returns:
        List[Dict[str, str]]: A list of examples.

    Example usage:
        examples = fetch_examples(
            source='https://raw.githubusercontent.com/KutalVolkan/many-shot-jailbreaking-dataset/5eac855/examples.json',
            source_type='public_url'
        )
    """

    file_type = source.split(".")[-1]
    if file_type not in FILE_TYPE_HANDLERS:
        valid_types = ", ".join(FILE_TYPE_HANDLERS.keys())
        raise ValueError(f"Invalid file_type. Expected one of: {valid_types}.")

    if not data_home:
        data_home = RESULTS_PATH / "datasets"
    else:
        data_home = Path(data_home)

    cache_file = data_home / _get_cache_file_name(source, file_type)

    if cache and cache_file.exists():
        return _read_cache(cache_file, file_type)

    if source_type == "public_url":
        examples = _fetch_from_public_url(source, file_type)
    elif source_type == "file":
        examples = _fetch_from_file(source, file_type)

    if cache:
        _write_cache(cache_file, examples, file_type)
    else:
        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=f".{file_type}") as temp_file:
            FILE_TYPE_HANDLERS[file_type]["write"](temp_file, examples)

    return examples


def fetch_many_shot_jailbreaking_examples() -> List[Dict[str, str]]:
    """
    Fetch many-shot jailbreaking examples from a specified source.

    Returns:
        List[Dict[str, str]]: A list of many-shot jailbreaking examples.
    """

    source = "https://raw.githubusercontent.com/KutalVolkan/many-shot-jailbreaking-dataset/5eac855/examples.json"
    source_type: Literal["public_url"] = "public_url"

    return fetch_examples(source, source_type)


def fetch_seclists_bias_testing_examples(
    source: str = (
        "https://raw.githubusercontent.com/danielmiessler/SecLists/4e747a4/Ai/LLM_Testing/Bias_Testing/"
        "nationality_geographic_bias.txt"
    ),
    source_type: Literal["public_url"] = "public_url",
    cache: bool = True,
    data_home: Optional[Path] = None,
    random_seed: Optional[int] = None,
    country: Optional[str] = None,
    region: Optional[str] = None,
    nationality: Optional[str] = None,
    gender: Optional[str] = None,
    skin_color: Optional[str] = None,
) -> PromptDataset:
    """
    Fetch SecLists AI LLM Bias Testing examples from a specified source and create a PromptDataset.
    Args:
        source (str): The source from which to fetch examples. Defaults to the SecLists repository Bias_Testing.
        source_type (Literal["public_url"]): The type of source ('public_url').
        cache (bool): Whether to cache the fetched examples. Defaults to True.
        data_home (Optional[Path]): Directory to store cached data. Defaults to None.
        random_seed (Optional[int]): Seed for random number generation for reproducibility. Defaults to None.
        country (Optional[str]): Specific country to use for the placeholder. Defaults to None.
        region (Optional[str]): Specific region to use for the placeholder. Defaults to None.
        nationality (Optional[str]): Specific nationality to use for the placeholder. Defaults to None.
        gender (Optional[str]): Specific gender to use for the placeholder. Defaults to None.
        skin_color (Optional[str]): Specific skin color to use for the placeholder. Defaults to None.

    Returns:
        PromptDataset: A PromptDataset containing the examples with placeholders replaced.
    """

    if random_seed is not None:
        random.seed(random_seed)

    # Determine the file type from the source URL
    file_type = source.split(".")[-1]
    if file_type not in FILE_TYPE_HANDLERS:
        valid_types = ", ".join(FILE_TYPE_HANDLERS.keys())
        raise ValueError(f"Invalid file_type. Expected one of: {valid_types}.")

    # Fetch the examples using the provided `fetch_examples` function
    examples = fetch_examples(source, source_type, cache, data_home)

    # Define placeholder values generation functions using pycountry and random.choice
    countries = list(pycountry.countries)
    subdivisions = list(pycountry.subdivisions)

    placeholder_values = {
        "Country": lambda: country if country else random.choice(countries).name,
        "Region": lambda: region if region else random.choice(subdivisions).name,
        "Nationality": lambda: (
            nationality if nationality else random.choice(countries).name
        ),  # Same as Country, the target LLM should be able to understand the context
        "GENDER": lambda: gender if gender else random.choice(["male", "female", "non-binary"]),
        "SKIN_COLOR": lambda: skin_color if skin_color else random.choice(["white", "black", "brown", "yellow"]),
    }

    # Replace placeholders in the examples with generated values
    filled_examples = []
    for example in examples:
        prompt = example["prompt"]
        for placeholder, generator in placeholder_values.items():
            values_used = set()
            while f"[{placeholder}]" in prompt:
                value = generator()
                # Ensure the new value is not the same as the previous one
                while value in values_used:
                    value = generator()
                values_used.add(value)
                prompt = prompt.replace(f"[{placeholder}]", value, 1)

        filled_examples.append(prompt)

    # Create a PromptDataset object with the filled examples
    dataset = PromptDataset(
        name="SecLists Bias Testing Examples",
        description="A dataset of SecLists AI LLM Bias Testing examples with placeholders replaced.",
        harm_category="bias_testing",
        should_be_blocked=False,
        prompts=filled_examples,
    )

    return dataset


def fetch_xstest_examples(
    source: str = "https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/a3bb396/xstest_v2_prompts.csv",
    source_type: Literal["public_url"] = "public_url",
    cache: bool = True,
    data_home: Optional[Path] = None,
) -> PromptDataset:
    """
    Fetch XSTest examples and create a PromptDataset.

    Args:
        source (str): The source from which to fetch examples. Defaults to the exaggerated-safety repository.
        source_type (Literal["public_url"]): The type of source ('public_url').
        cache (bool): Whether to cache the fetched examples. Defaults to True.
        data_home (Optional[Path]): Directory to store cached data. Defaults to None.

    Returns:
        PromptDataset: A PromptDataset containing the examples.

    Note:
        For more information and access to the original dataset and related materials, visit:
        https://github.com/paul-rottger/exaggerated-safety
    """

    # Determine the file type from the source URL
    file_type = source.split(".")[-1]
    if file_type not in FILE_TYPE_HANDLERS:
        valid_types = ", ".join(FILE_TYPE_HANDLERS.keys())
        raise ValueError(f"Invalid file_type. Expected one of: {valid_types}.")

    # Fetch the examples using the provided `fetch_examples` function
    examples = fetch_examples(source, source_type, cache, data_home)

    # Extract prompts, harm categories, and other relevant data from the fetched examples
    prompts = [example["prompt"] for example in examples]
    harm_categories = [example["note"] for example in examples]

    # Join all categories into a single comma-separated string
    harm_category_str = ", ".join(filter(None, harm_categories))

    # Create a PromptDataset object with the fetched examples
    dataset = PromptDataset(
        name="XSTest Examples",
        description="A dataset of XSTest examples containing various categories such as violence, drugs, etc.",
        harm_category=harm_category_str,
        should_be_blocked=False,
        prompts=prompts,
    )

    return dataset


def fetch_harmbench_examples(
    source: str = (
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/c0423b9/data/behavior_datasets/"
        "harmbench_behaviors_text_all.csv"
    ),
    source_type: Literal["public_url"] = "public_url",
    cache: bool = True,
    data_home: Optional[Path] = None,
) -> PromptDataset:
    """
    Fetch HarmBench examples and create a PromptDataset.

    Args:
        source (str): The source from which to fetch examples. Defaults to the HarmBench repository.
        source_type (Literal["public_url"]): The type of source ('public_url').
        cache (bool): Whether to cache the fetched examples. Defaults to True.
        data_home (Optional[Path]): Directory to store cached data. Defaults to None.

    Returns:
        PromptDataset: A PromptDataset containing the examples.

    Note:
        For more information and access to the original dataset and related materials, visit:
        https://github.com/centerforaisafety/HarmBench
    """

    # Determine the file type from the source URL
    file_type = source.split(".")[-1]
    if file_type not in FILE_TYPE_HANDLERS:
        valid_types = ", ".join(FILE_TYPE_HANDLERS.keys())
        raise ValueError(f"Invalid file_type. Expected one of: {valid_types}.")

    # Required keys to validate each example
    required_keys = {"Behavior", "SemanticCategory"}

    # Initialize containers for prompts and semantic categories
    prompts = []
    semantic_categories = set()

    # Fetch the examples using the provided `fetch_examples` function
    examples = fetch_examples(source, source_type, cache, data_home)

    # Validate each example and extract data
    for example in examples:
        # Check for missing keys in the example
        missing_keys = required_keys - example.keys()
        if missing_keys:
            raise ValueError(f"Missing keys in example: {', '.join(missing_keys)}")

        # Extract and append the data to respective containers
        prompts.append(example["Behavior"])
        semantic_categories.add(example["SemanticCategory"])

        # Use the semantic categories to determine harm categories
        harm_category_str = ", ".join(set(semantic_categories))

    # Create a PromptDataset object with the fetched examples
    dataset = PromptDataset(
        name="HarmBench Examples",
        description=(
            "A dataset of HarmBench examples containing various categories such as chemical,"
            "biological, illegal activities, etc."
        ),
        harm_category=harm_category_str,
        should_be_blocked=True,
        prompts=prompts,
    )

    return dataset
