# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from typing import Dict, List


def read_json(file) -> List[Dict[str, str]]:
    return json.load(file)


def write_json(file, examples: List[Dict[str, str]]):
    json.dump(examples, file)
