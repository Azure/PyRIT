# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union
import uuid

from pyrit.common.yaml_loadable import YamlLoadable
from pyrit.models.literals import PromptDataType


@dataclass
class Prompt(YamlLoadable):
    id: Optional[Union[uuid.UUID, str]]
    value: str
    data_type: PromptDataType
    name: Optional[str]
    dataset_name: Optional[str]
    harm_categories: Optional[List[str]]
    description: Optional[str]
    authors: Optional[List[str]]
    groups: Optional[List[str]]
    source: Optional[str]
    date_added: Optional[datetime]
    added_by: Optional[str]
    metadata: Optional[Dict[str, str]]
    parameters: Optional[List[str]]
    prompt_group_id: Optional[Union[uuid.UUID, str]]
    sequence: Optional[int]

    def __init__(
        self,
        *,
        id: Optional[uuid.UUID] = None,
        value: str,
        data_type: PromptDataType,
        name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        harm_categories: Optional[List[str]] = None,
        description: Optional[str] = None,
        authors: Optional[List[str]] = None,
        groups: Optional[List[str]] = None,
        source: Optional[str] = None,
        date_added: Optional[datetime] = datetime.now(),
        added_by: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        parameters: Optional[List[str]] = None,
        prompt_group_id: Optional[uuid.UUID] = None,
        sequence: Optional[int] = None,
    ):
        self.id = id if id else uuid.uuid4()
        self.value = value
        self.data_type = data_type
        self.name = name
        self.dataset_name = dataset_name
        self.harm_categories = harm_categories or []
        self.description = description
        self.authors = authors or []
        self.groups = groups or []
        self.source = source
        self.date_added = date_added
        self.added_by = added_by
        self.metadata = metadata or {}
        self.parameters = parameters or []
        self.prompt_group_id = prompt_group_id
        self.sequence = sequence


class PromptGroup(YamlLoadable):
    id: Optional[Union[uuid.UUID, str]]
    name: str
    description: Optional[str]
    prompts: List[List[Prompt]]

    def __init__(
        self,
        *,
        id: Optional[uuid.UUID] = None,
        name: str,
        description: Optional[str] = None,
        prompts: List[Prompt],
    ):
        self.id = id if id else uuid.uuid4()
        self.name = name
        self.description = description
        self.prompts = prompts

# TODO: consider removing as it doesn't serve any purpose
class PromptDataset(YamlLoadable):
    prompts: List[Prompt]

    def __init__(self, prompts: List[Prompt]):
        if prompts and isinstance(prompts[0], dict):
            self.prompts = [Prompt(**prompt_args) for prompt_args in prompts]
        else:
            self.prompts = prompts
