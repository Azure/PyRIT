# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union
import uuid

from pyrit.common.apply_parameters_to_template import apply_parameters_to_template
from pyrit.common.yaml_loadable import YamlLoadable
from pyrit.models.literals import PromptDataType


@dataclass
class SeedPrompt(YamlLoadable):
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

    def to_prompt_template(self) -> SeedPromptTemplate:
        if not self.parameters:
            raise ValueError("SeedPrompt must have parameters to convert to a SeedPromptTemplate.")
        return SeedPromptTemplate(
            id=self.id,
            value=self.value,
            data_type=self.data_type,
            name=self.name,
            dataset_name=self.dataset_name,
            harm_categories=self.harm_categories,
            description=self.description,
            authors=self.authors,
            groups=self.groups,
            source=self.source,
            date_added=self.date_added,
            added_by=self.added_by,
            metadata=self.metadata,
            parameters=self.parameters,
            prompt_group_id=self.prompt_group_id,
            sequence=self.sequence,
        )


class SeedPromptTemplate(SeedPrompt):
    parameters: List[str]

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
        parameters: List[str],
        prompt_group_id: Optional[uuid.UUID] = None,
        sequence: Optional[int] = None,
    ):
        if data_type != PromptDataType.TEXT:
            raise ValueError("SeedPromptTemplate must have data_type 'text'.")
        if not parameters:
            raise ValueError("SeedPromptTemplate must have parameters. Please provide at least one.")
        super().__init__(
            id=id,
            value=value,
            data_type=data_type,
            name=name,
            dataset_name=dataset_name,
            harm_categories=harm_categories,
            description=description,
            authors=authors,
            groups=groups,
            source=source,
            date_added=date_added,
            added_by=added_by,
            metadata=metadata,
            parameters=parameters,
            prompt_group_id=prompt_group_id,
            sequence=sequence,
        )
    
    def apply_parameters(self, **kwargs) -> SeedPrompt:
        if not self.parameters:
            raise ValueError("SeedPromptTemplate must have parameters to apply.")
        if not all(param in kwargs for param in self.parameters):
            raise ValueError("Not all parameters were provided.")
        return apply_parameters_to_template(self.value, **kwargs)


class SeedPromptGroup(YamlLoadable):
    """
    A group of prompts that need to be sent together.

    For example, when using a target that requires multiple (multimodal) prompt pieces to be sent together,
    the prompt group enables grouping them together. Their prompt_group_id should match.
    """
    prompts: List[SeedPrompt]

    def __init__(
        self,
        *,
        prompts: List[SeedPrompt],
    ):
        self.prompts = prompts
        if len(prompts) > 1:
            if any([prompt.sequence is None for prompt in prompts]):
                raise ValueError("All prompts in a group must have a sequence number.")
            self.prompts.sort(key=lambda prompt: prompt.sequence)


class SeedPromptDataset(YamlLoadable):
    prompts: List[SeedPrompt]

    def __init__(self, prompts: List[SeedPrompt]):
        if prompts and isinstance(prompts[0], dict):
            self.prompts = [SeedPrompt(**prompt_args) for prompt_args in prompts]
        else:
            self.prompts = prompts
