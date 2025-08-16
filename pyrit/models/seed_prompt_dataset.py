# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, Union

from pydantic.types import PositiveInt

from pyrit.common import utils
from pyrit.common.yaml_loadable import YamlLoadable
from pyrit.models.literals import PromptDataType
from pyrit.models.seed_prompt import SeedPrompt
from pyrit.models.seed_prompt_group import SeedPromptGroup

logger = logging.getLogger(__name__)


class SeedPromptDataset(YamlLoadable):
    """
    SeedPromptDataset manages seed prompts plus optional top-level defaults.
    Prompts are stored as a Sequence[SeedPrompt], so references to prompt properties
    are straightforward (e.g. ds.prompts[0].value).
    """

    data_type: Optional[str]
    name: Optional[str]
    dataset_name: Optional[str]
    harm_categories: Optional[Sequence[str]]
    description: Optional[str]
    authors: Optional[Sequence[str]]
    groups: Optional[Sequence[str]]
    source: Optional[str]
    date_added: Optional[datetime]
    added_by: Optional[str]

    # Now the actual prompts
    prompts: Sequence["SeedPrompt"]

    def __init__(
        self,
        *,
        prompts: Optional[Union[Sequence[Dict[str, Any]], Sequence[SeedPrompt]]] = None,
        data_type: Optional[PromptDataType] = "text",
        name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        harm_categories: Optional[Sequence[str]] = None,
        description: Optional[str] = None,
        authors: Optional[Sequence[str]] = None,
        groups: Optional[Sequence[str]] = None,
        source: Optional[str] = None,
        date_added: Optional[datetime] = None,
        added_by: Optional[str] = None,
    ):
        """
        Initialize the dataset.
        Typically, you'll call from_dict or from_yaml_file so that top-level defaults
        are merged into each prompt. If you're passing prompts directly, they can be
        either a list of SeedPrompt objects or prompt dictionaries (which then get
        converted to SeedPrompt objects).
        """
        if prompts is None:
            prompts = []
        if not prompts:
            raise ValueError("SeedPromptDataset cannot be empty.")

        # Store top-level fields
        self.data_type = data_type
        self.name = name
        self.dataset_name = dataset_name

        self.harm_categories = harm_categories
        self.description = description
        self.authors = authors or []
        self.groups = groups or []
        self.source = source
        self.date_added = date_added or datetime.now()
        self.added_by = added_by

        # Convert any dictionaries in `prompts` to SeedPrompt objects
        self.prompts = []
        for p in prompts:
            if isinstance(p, dict):
                self.prompts.append(SeedPrompt(**p))
            elif isinstance(p, SeedPrompt):
                self.prompts.append(p)
            else:
                raise ValueError("Prompts should be either dicts or SeedPrompt objects. Got something else.")

    def get_values(self, first: Optional[PositiveInt] = None, last: Optional[PositiveInt] = None) -> Sequence[str]:
        """
        Extracts and returns a list of prompt values from the dataset. By default, returns all of them.

        Args:
            first (Optional[int]): If provided, values from the first N prompts are included.
            last (Optional[int]): If provided, values from the last N prompts are included.

        Returns:
            Sequence[str]: A list of prompt values.
        """
        values = [prompt.value for prompt in self.prompts]

        if first is None and last is None:
            return values
        if first and last and first + last >= len(values):
            return values  # simply return all values in case of an overlap

        first_part = values[:first] if first is not None else []
        last_part = values[-last:] if last is not None else []

        return first_part + last_part

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeedPromptDataset":
        """
        Builds a SeedPromptDataset by merging top-level defaults into each item in 'prompts'.
        """
        # Pop out the prompts section
        prompts_data = data.pop("prompts", [])
        dataset_defaults = data  # everything else is top-level

        merged_prompts = []
        for p in prompts_data:
            # Merge dataset-level fields with the prompt-level fields
            merged = utils.combine_dict(dataset_defaults, p)

            merged["harm_categories"] = utils.combine_list(
                dataset_defaults.get("harm_categories", []),
                p.get("harm_categories", []),
            )

            merged["authors"] = utils.combine_list(
                dataset_defaults.get("authors", []),
                p.get("authors", []),
            )

            merged["groups"] = utils.combine_list(
                dataset_defaults.get("groups", []),
                p.get("groups", []),
            )

            if "data_type" not in merged:
                merged["data_type"] = dataset_defaults.get("data_type", "text")

            merged_prompts.append(merged)

        for prompt in merged_prompts:
            if "prompt_group_id" in prompt:
                raise ValueError("prompt_group_id should not be set in prompt data")

        SeedPromptDataset._set_seed_prompt_group_id_by_alias(seed_prompts=merged_prompts)

        # Now create the dataset with the newly merged prompt dicts
        return cls(prompts=merged_prompts, **dataset_defaults)

    def render_template_value(self, **kwargs):
        """Renders self.value as a template, applying provided parameters in kwargs

        Args:
            kwargs:Key-value pairs to replace in the SeedPromptDataset value.

        Returns:
            None

        Raises:
            ValueError: If parameters are missing or invalid in the template.
        """

        for prompt in self.prompts:
            prompt.value = prompt.render_template_value(**kwargs)

    @staticmethod
    def _set_seed_prompt_group_id_by_alias(seed_prompts: Sequence[dict]):
        """
        Sets all seed_prompt_group_ids based on prompt_group_alias matches

        This is important so the prompt_group_alias can be set in yaml to group prompts
        """
        alias_to_group_id = {}

        for prompt in seed_prompts:
            alias = prompt.get("prompt_group_alias")
            if alias:
                if alias not in alias_to_group_id:
                    alias_to_group_id[alias] = uuid.uuid4()
                prompt["prompt_group_id"] = alias_to_group_id[alias]
            else:
                prompt["prompt_group_id"] = uuid.uuid4()

    @staticmethod
    def group_seed_prompts_by_prompt_group_id(seed_prompts: Sequence[SeedPrompt]) -> Sequence[SeedPromptGroup]:
        """
        Groups the given list of SeedPrompts by their prompt_group_id and creates
        SeedPromptGroup instances. All seed prompts in a group must share the same prompt_group_id

        Args:
            seed_prompts: A list of SeedPrompt objects.

        Returns:
            A list of SeedPromptGroup objects, with prompts grouped by prompt_group_id. Each SeedPromptGroup
            will be ordered by the sequence number of the prompts, if available.

        """
        # Group seed prompts by `prompt_group_id`
        grouped_prompts = defaultdict(list)
        for prompt in seed_prompts:
            if prompt.prompt_group_id:
                grouped_prompts[prompt.prompt_group_id].append(prompt)
            else:
                grouped_prompts[uuid.uuid4()].append(prompt)

        # Create SeedPromptGroup instances from grouped prompts
        seed_prompt_groups = []
        for group_prompts in grouped_prompts.values():
            if len(group_prompts) > 1:
                group_prompts.sort(key=lambda prompt: prompt.sequence)

            seed_prompt_group = SeedPromptGroup(prompts=group_prompts)
            seed_prompt_groups.append(seed_prompt_group)

        return seed_prompt_groups

    def __repr__(self):
        return f"<SeedPromptDataset(prompts={len(self.prompts)} prompts)>"
