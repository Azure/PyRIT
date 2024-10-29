# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import uuid

from collections import defaultdict

from pyrit.common.apply_parameters_to_template import apply_parameters_to_template
from pyrit.common.yaml_loadable import YamlLoadable
from pyrit.models.literals import PromptDataType


@dataclass
class SeedPrompt(YamlLoadable):
    """Represents a seed prompt with various attributes and metadata."""

    id: Optional[uuid.UUID]
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
    prompt_group_id: Optional[uuid.UUID]
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
        """Convert the SeedPrompt instance to a SeedPromptTemplate."""
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
    """Represents a template of a seed prompt with required parameters."""

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
        if data_type != "text":
            raise ValueError("SeedPromptTemplate must have data_type 'text'.")
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

    def apply_parameters(self, **kwargs) -> str:
        """Applies parameters to the seed prompt template and returns the formatted string."""
        if not self.parameters:
            raise ValueError("SeedPromptTemplate must have parameters to apply.")
        if not all(param in kwargs for param in self.parameters):
            raise ValueError("Not all parameters were provided.")

        return apply_parameters_to_template(self.value, self.parameters, **kwargs)


class SeedPromptGroup(YamlLoadable):
    """
    A group of prompts that need to be sent together.

    This class is useful when a target requires multiple (multimodal) prompt pieces to be grouped
    and sent together. All prompts in the group should share the same `prompt_group_id`.
    """

    prompts: List[SeedPrompt]

    def __init__(
        self,
        *,
        prompts: List[SeedPrompt],
    ):
        self.prompts = prompts
        if self.prompts and isinstance(self.prompts[0], dict):
            self.prompts = [SeedPrompt(**prompt_args) for prompt_args in self.prompts]  # type: ignore
        else:
            self.prompts = prompts

        # Check sequence and sort the prompts in the same loop
        if len(self.prompts) >= 1:
            self.prompts = sorted(self.prompts, key=lambda prompt: self._validate_and_get_sequence(prompt))

    def _validate_and_get_sequence(self, prompt: SeedPrompt) -> int:
        """
        Validates the sequence of a prompt and returns it.

        Args:
            prompt (SeedPrompt): The prompt whose sequence needs to be validated.

        Returns:
            int: The sequence number of the prompt.

        Raises:
            ValueError: If the prompt does not have a sequence number.
        """
        if prompt.sequence is None:
            raise ValueError("All prompts in a group must have a sequence number.")
        return prompt.sequence

    def __repr__(self):
        return f"<SeedPromptGroup(prompts={len(self.prompts)} prompts)>"


class SeedPromptDataset(YamlLoadable):
    """A dataset consisting of a list of seed prompts."""

    prompts: List[SeedPrompt]

    def __init__(self, prompts: List[SeedPrompt]):
        if prompts and isinstance(prompts[0], dict):
            self.prompts = [SeedPrompt(**prompt_args) for prompt_args in prompts]  # type: ignore
        else:
            self.prompts = prompts

    def __repr__(self):
        return f"<SeedPromptDataset(prompts={len(self.prompts)} prompts)>"


def group_seed_prompts_by_prompt_group_id(seed_prompts: list[SeedPrompt]) -> list[SeedPromptGroup]:
    """
    Groups the given list of SeedPrompts by their prompt_group_id and creates
    SeedPromptGroup instances.

    Args:
        seed_prompts: A list of SeedPrompt objects.

    Returns:
        A list of SeedPromptGroup objects, with prompts grouped by prompt_group_id.
    """
    # Group seed prompts by `prompt_group_id`
    grouped_prompts = defaultdict(list)
    for prompt in seed_prompts:
        if prompt.prompt_group_id:
            grouped_prompts[prompt.prompt_group_id].append(prompt)

    # Create SeedPromptGroup instances from grouped prompts
    seed_prompt_groups = []
    for group_prompts in grouped_prompts.values():
        if len(group_prompts) > 1:
            group_prompts.sort(key=lambda prompt: prompt.sequence)

        seed_prompt_group = SeedPromptGroup(prompts=group_prompts)
        seed_prompt_groups.append(seed_prompt_group)

    return seed_prompt_groups
