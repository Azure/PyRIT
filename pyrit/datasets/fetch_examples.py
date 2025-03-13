# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import hashlib
import io
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, TextIO

import requests

from pyrit.common.csv_helper import read_csv, write_csv
from pyrit.common.json_helper import read_json, read_jsonl, write_json, write_jsonl
from pyrit.common.path import DB_DATA_PATH
from pyrit.common.text_helper import read_txt, write_txt

# Define the type for the file handlers
FileHandlerRead = Callable[[TextIO], List[Dict[str, str]]]
FileHandlerWrite = Callable[[TextIO, List[Dict[str, str]]], None]

FILE_TYPE_HANDLERS: Dict[str, Dict[str, Callable]] = {
    "json": {"read": read_json, "write": write_json},
    "jsonl": {"read": read_jsonl, "write": write_jsonl},
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

    Example usage
    >>> examples = fetch_examples(
    >>>     source='https://raw.githubusercontent.com/KutalVolkan/many-shot-jailbreaking-dataset/5eac855/examples.json',
    >>>     source_type='public_url'
    >>> )

    Args:
        source (str): The source from which to fetch examples.
        source_type (Literal["public_url", "file"]): The type of source ('public_url' or 'file').
        cache (bool): Whether to cache the fetched examples. Defaults to True.
        data_home (Optional[Path]): Directory to store cached data. Defaults to None.

    Returns:
        List[Dict[str, str]]: A list of examples.
    """

    file_type = source.split(".")[-1]
    if file_type not in FILE_TYPE_HANDLERS:
        valid_types = ", ".join(FILE_TYPE_HANDLERS.keys())
        raise ValueError(f"Invalid file_type. Expected one of: {valid_types}.")

    if not data_home:
        data_home = DB_DATA_PATH / "seed-prompt-entries"
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
