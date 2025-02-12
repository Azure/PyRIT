# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from typing import Dict, List


def read_json(file) -> List[Dict[str, str]]:
    return json.load(file)


def write_json(file, examples: List[Dict[str, str]]):
    json.dump(examples, file)


def read_jsonl(file) -> List[Dict[str, str]]:
    return [json.loads(line) for line in file]


def write_jsonl(file, examples: List[Dict[str, str]]):
    for example in examples:
        file.write(json.dumps(example) + "\n")
