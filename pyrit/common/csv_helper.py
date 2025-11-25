# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
from typing import Dict, List


def read_csv(file) -> List[Dict[str, str]]:
    """
    Read a CSV file and return its rows as dictionaries.

    Returns:
        List[Dict[str, str]]: Parsed CSV rows as dictionaries.
    """
    reader = csv.DictReader(file)
    return [row for row in reader]


def write_csv(file, examples: List[Dict[str, str]]):
    """
    Write a list of dictionaries to a CSV file.

    Args:
        file: A file-like object opened for writing CSV data.
        examples (List[Dict[str, str]]): List of dictionaries to write as CSV rows.
    """
    writer = csv.DictWriter(file, fieldnames=examples[0].keys())
    writer.writeheader()
    writer.writerows(examples)
