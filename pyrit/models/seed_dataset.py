# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import random
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, Union

from pydantic.types import PositiveInt

from pyrit.common import utils
from pyrit.common.yaml_loadable import YamlLoadable
from pyrit.models.literals import PromptDataType
from pyrit.models.seed import Seed
from pyrit.models.seed_group import SeedGroup
from pyrit.models.seed_objective import SeedObjective
from pyrit.models.seed_prompt import SeedPrompt

logger = logging.getLogger(__name__)


class SeedDataset(YamlLoadable):
    """
    SeedDataset manages seed prompts plus optional top-level defaults.
    Prompts are stored as a Sequence[Seed], so references to prompt properties
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
    objectives: Sequence["SeedObjective"]

    def __init__(
        self,
        *,
        prompts: Optional[Union[Sequence[Dict[str, Any]], Sequence[Seed]]] = None,
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
            raise ValueError("SeedDataset cannot be empty.")

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

        # Convert any dictionaries in `prompts` to SeedPrompt and/or SeedObjective objects
        self.prompts = []
        self.objectives = []
        for p in prompts:
            if isinstance(p, dict):
                if "is_objective" in p:
                    if p["is_objective"]:
                        self.objectives.append(
                            SeedObjective(
                                value=p["value"],
                                data_type="text",
                                value_sha256=p.get("value_sha256"),
                                id=uuid.uuid4(),
                                name=p.get("name"),
                                dataset_name=p.get("dataset_name"),
                                harm_categories=p.get("harm_categories", []),
                                description=p.get("description"),
                                authors=p.get("authors", []),
                                groups=p.get("groups", []),
                                source=p.get("source"),
                                date_added=p.get("date_added"),
                                added_by=p.get("added_by"),
                                metadata=p.get("metadata", {}),
                                prompt_group_id=p.get("prompt_group_id"),
                            )
                        )
                    del p["is_objective"]
                self.prompts.append(SeedPrompt(**p))
            elif isinstance(p, SeedPrompt):
                self.prompts.append(p)
            elif isinstance(p, SeedObjective):
                self.objectives.append(p)
            else:
                raise ValueError(
                    "Prompts should be either dicts, SeedPrompt objects, or SeedObjective objects. Got something else."
                )

    def get_values(
        self,
        *,
        first: Optional[PositiveInt] = None,
        last: Optional[PositiveInt] = None,
        harm_categories: Optional[Sequence[str]] = None,
    ) -> Sequence[str]:
        """
        Extracts and returns a list of prompt values from the dataset. By default, returns all of them.

        Args:
            first (Optional[int]): If provided, values from the first N prompts are included.
            last (Optional[int]): If provided, values from the last N prompts are included.
            harm_categories (Optional[Sequence[str]]): If provided, only prompts containing at least one of
                these harm categories are included.

        Returns:
            Sequence[str]: A list of prompt values.
        """
        # Filter by harm categories if specified
        prompts = self.prompts
        if harm_categories:
            prompts = [
                prompt
                for prompt in prompts
                if prompt.harm_categories and any(cat in prompt.harm_categories for cat in harm_categories)
            ]

        values = [prompt.value for prompt in prompts]

        if first is None and last is None:
            return values
        if first and last and first + last >= len(values):
            return values  # simply return all values in case of an overlap

        first_part = values[:first] if first is not None else []
        last_part = values[-last:] if last is not None else []

        return first_part + last_part

    def get_random_values(
        self, *, number: PositiveInt, harm_categories: Optional[Sequence[str]] = None
    ) -> Sequence[str]:
        """
        Extracts and returns a list of random prompt values from the dataset.

        Args:
            number (int): The number of random prompt values to return.
            harm_categories (Optional[Sequence[str]]): If provided, only prompts containing at least one of
                these harm categories are included.

        Returns:
            Sequence[str]: A list of prompt values.
        """
        prompts = self.get_values(harm_categories=harm_categories)
        return random.sample(prompts, min(len(prompts), number))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SeedDataset:
        """
        Builds a SeedDataset by merging top-level defaults into each item in 'prompts'.
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

        SeedDataset._set_seed_group_id_by_alias(seed_prompts=merged_prompts)

        # Now create the dataset with the newly merged prompt dicts
        return cls(prompts=merged_prompts, **dataset_defaults)

    def render_template_value(self, **kwargs):
        """
        Renders self.value as a template, applying provided parameters in kwargs.

        Args:
            kwargs:Key-value pairs to replace in the SeedDataset value.

        Returns:
            None

        Raises:
            ValueError: If parameters are missing or invalid in the template.
        """

        for prompt in self.prompts:
            prompt.value = prompt.render_template_value(**kwargs)

    @staticmethod
    def _set_seed_group_id_by_alias(seed_prompts: Sequence[dict]):
        """
        Sets all seed_group_ids based on prompt_group_alias matches.

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
    def group_seed_prompts_by_prompt_group_id(seed: Sequence[Seed]) -> Sequence[SeedGroup]:
        """
        Groups the given list of Seeds by their prompt_group_id and creates
        SeedGroup instances. All seed prompts in a group must share the same prompt_group_id.

        Args:
            seed: A list of Seed objects.

        Returns:
            A list of SeedGroup objects, with prompts grouped by prompt_group_id. Each SeedGroup
            will be ordered by the sequence number of the prompts, if available.

        """
        # Group seed prompts by `prompt_group_id`
        grouped_prompts = defaultdict(list)
        for prompt in seed:
            if prompt.prompt_group_id:
                grouped_prompts[prompt.prompt_group_id].append(prompt)
            else:
                grouped_prompts[uuid.uuid4()].append(prompt)

        # Create SeedGroup instances from grouped prompts
        seed_groups = []
        for group_prompts in grouped_prompts.values():
            if len(group_prompts) > 1:
                group_prompts.sort(key=lambda prompt: prompt.sequence if hasattr(prompt, "sequence") else 0)

            seed_group = SeedGroup(prompts=group_prompts)
            seed_groups.append(seed_group)

        return seed_groups

    def __repr__(self):
        return f"<SeedDataset(prompts={len(self.prompts)} prompts)>"
