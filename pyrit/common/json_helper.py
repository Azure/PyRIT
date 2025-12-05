# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from typing import Dict, List


def read_json(file) -> List[Dict[str, str]]:
    """
    Read a JSON file and return its content.

    Returns:
        List[Dict[str, str]]: Parsed JSON content.
    """
    return json.load(file)


def write_json(file, examples: List[Dict[str, str]]):
    """
    Write a list of dictionaries to a JSON file.

    Args:
        file: A file-like object opened for writing JSON data.
        examples (List[Dict[str, str]]): List of dictionaries to write as JSON.
    """
    json.dump(examples, file)


def read_jsonl(file) -> List[Dict[str, str]]:
    """
    Read a JSONL file and return its content.

    Returns:
        List[Dict[str, str]]: Parsed JSONL content.
    """
    return [json.loads(line) for line in file]


def write_jsonl(file, examples: List[Dict[str, str]]):
    """
    Write a list of dictionaries to a JSONL file.

    Args:
        file: A file-like object opened for writing JSONL data.
        examples (List[Dict[str, str]]): List of dictionaries to write as JSONL.
    """
    for example in examples:
        file.write(json.dumps(example) + "\n")
