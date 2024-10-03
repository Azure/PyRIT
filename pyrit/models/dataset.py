# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass, field

from pyrit.common.yaml_loadable import YamlLoadable


@dataclass
class PromptDataset(YamlLoadable):
    name: str
    description: str
    harm_category: str
    should_be_blocked: bool
    author: str = ""
    group: str = ""
    source: str = ""
    prompts: list[str] = field(default_factory=list)
