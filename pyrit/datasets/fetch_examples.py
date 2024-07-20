# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import requests


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    with cache_file.open("r") as file:
        if file_type == "json":
            return json.load(file)
        elif file_type == "csv":
            reader = csv.DictReader(file)
            return [row for row in reader]
        elif file_type == "txt":
            return [{"prompt": line.strip()} for line in file.readlines()]
        else:
            raise ValueError("Invalid file_type. Expected 'json', 'csv', or 'txt'.")


def _write_cache(cache_file: Path, examples: List[Dict[str, str]], file_type: str):
    """
    Write data to cache.
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w") as file:
        if file_type == "json":
            json.dump(examples, file)
        elif file_type == "csv":
            writer = csv.DictWriter(file, fieldnames=examples[0].keys())
            writer.writeheader()
            writer.writerows(examples)
        elif file_type == "txt":
            file.write("\n".join([ex["prompt"] for ex in examples]))


def _fetch_from_repository(source: str, file_type: str) -> List[Dict[str, str]]:
    """
    Fetch examples from a repository.
    """
    response = requests.get(source)
    if response.status_code == 200:
        if file_type == "json":
            return response.json()
        elif file_type == "csv":
            reader = csv.DictReader(response.text.splitlines())
            return [row for row in reader]
        elif file_type == "txt":
            return [{"prompt": line} for line in response.text.splitlines()]
        else:
            raise ValueError("Invalid file_type. Expected 'json', 'csv', or 'txt'.")
    else:
        raise Exception(f"Failed to fetch examples from repository. Status code: {response.status_code}")


def _fetch_from_file(source: str, file_type: str) -> List[Dict[str, str]]:
    """
    Fetch examples from a local file.
    """
    with open(source, "r") as file:
        if file_type == "json":
            return json.load(file)
        elif file_type == "csv":
            reader = csv.DictReader(file)
            return [row for row in reader]
        elif file_type == "txt":
            return [{"prompt": line.strip()} for line in file.readlines()]
        else:
            raise ValueError("Invalid file_type. Expected 'json', 'csv', or 'txt'.")


def fetch_examples(
    source: str,
    source_type: str = "repository",
    file_type: str = "json",
    cache: bool = True,
    data_home: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """
    Fetch examples from a specified source with caching support.

    Args:
        source (str): The source from which to fetch examples.
        source_type (str): The type of source ('repository' or 'file'). Defaults to 'repository'.
        file_type (str): The type of file ('json', 'csv', or 'txt'). Defaults to 'json'.
        cache (bool): Whether to cache the fetched examples. Defaults to True.
        data_home (Optional[Path]): Directory to store cached data. Defaults to None.

    Returns:
        List[Dict[str, str]]: A list of examples.

    Example usage:
        examples = fetch_examples(
            source='https://raw.githubusercontent.com/KutalVolkan/many-shot-jailbreaking-dataset/5eac855/examples.json,
            source_type='repository'
        )
    """

    if not data_home:
        data_home = Path().home() / ".pyrit"
    else:
        data_home = Path(data_home)

    cache_file = data_home / _get_cache_file_name(source, file_type)

    if cache and cache_file.exists():
        logger.info(f"Loading examples from cache: {cache_file}")
        return _read_cache(cache_file, file_type)

    if source_type == "repository":
        examples = _fetch_from_repository(source, file_type)
    elif source_type == "file":
        examples = _fetch_from_file(source, file_type)
    else:
        raise ValueError("Invalid source_type. Expected 'repository' or 'file'.")

    if cache:
        logger.info(f"Caching examples at: {cache_file}")
        _write_cache(cache_file, examples, file_type)
    else:
        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=f".{file_type}") as temp_file:
            if file_type == "json":
                json.dump(examples, temp_file)
            elif file_type == "csv":
                writer = csv.DictWriter(temp_file, fieldnames=examples[0].keys())
                writer.writeheader()
                writer.writerows(examples)
            elif file_type == "txt":
                temp_file.write("\n".join([ex["prompt"] for ex in examples]))
            temp_file_path = temp_file.name
            logger.info(f"Examples stored in temporary file: {temp_file_path}")

    return examples


def fetch_many_shot_jailbreaking_examples() -> List[Dict[str, str]]:
    """
    Fetch many-shot jailbreaking examples from a specified source.

    Returns:
        List[Dict[str, str]]: A list of many-shot jailbreaking examples.
    """

    source = "https://raw.githubusercontent.com/KutalVolkan/many-shot-jailbreaking-dataset/5eac855/examples.json"
    source_type = "repository"

    return fetch_examples(source, source_type, file_type="json")
