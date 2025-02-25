# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List


def read_txt(file) -> List[Dict[str, str]]:
    return [{"prompt": line.strip()} for line in file.readlines()]


def write_txt(file, examples: List[Dict[str, str]]):
    file.write("\n".join([ex["prompt"] for ex in examples]))
