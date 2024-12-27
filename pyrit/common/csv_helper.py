# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
from typing import Dict, List


def read_csv(file) -> List[Dict[str, str]]:
    reader = csv.DictReader(file)
    return [row for row in reader]


def write_csv(file, examples: List[Dict[str, str]]):
    writer = csv.DictWriter(file, fieldnames=examples[0].keys())
    writer.writeheader()
    writer.writerows(examples)
