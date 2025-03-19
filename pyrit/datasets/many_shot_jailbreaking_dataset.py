# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Literal

from pyrit.datasets.dataset_helper import fetch_examples


def fetch_many_shot_jailbreaking_dataset() -> List[Dict[str, str]]:
    """
    Fetch many-shot jailbreaking dataset from a specified source.

    Returns:
        List[Dict[str, str]]: A list of many-shot jailbreaking examples.
    """

    source = "https://raw.githubusercontent.com/KutalVolkan/many-shot-jailbreaking-dataset/5eac855/examples.json"
    source_type: Literal["public_url"] = "public_url"

    return fetch_examples(source, source_type)
