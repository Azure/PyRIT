# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from typing import Any, Dict, Optional, Sequence, Union

from pyrit.common.yaml_loadable import YamlLoadable
from pyrit.models.seed import Seed
from pyrit.models.seed_objective import SeedObjective
from pyrit.models.seed_prompt import SeedPrompt

logger = logging.getLogger(__name__)


class SeedPromptGroup(YamlLoadable):
    """
    A group of prompts that need to be sent together, along with an objective. This can include multiturn and multimodal
    prompts.
    This class is useful when a target requires multiple message pieces to be grouped and sent together.
    All prompts in the group should share the same `prompt_group_id`.
    """

    objective: Optional[SeedObjective] = None
    prompts: Sequence[SeedPrompt]

    def __init__(
        self,
        *,
        prompts: Union[Sequence[Seed], Sequence[Dict[str, Any]]],
    ):
        if not prompts:
            raise ValueError("SeedPromptGroup cannot be empty.")
        self.prompts = []
        for prompt in prompts:
            if isinstance(prompt, SeedPrompt):
                self.prompts.append(prompt)
            elif isinstance(prompt, SeedObjective):
                self._set_objective_from_prompt(objective_prompt=prompt)
            elif isinstance(prompt, dict):
                # create an SeedObjective in addition to the SeedPrompt if is_objective is True
                is_objective = prompt.pop("is_objective", False)
                self.prompts.append(SeedPrompt(**prompt))
                if is_objective:
                    self._set_objective_from_prompt()
                    # if the only prompt is the objective, remove it from prompts
                    if len(prompts) == 1:
                        self.prompts = []

        self._enforce_consistent_group_id()
        self._enforce_consistent_role()

        # Check sequence and sort the prompts in the same loop
        if len(self.prompts) >= 1:
            self.prompts = sorted(
                self.prompts, key=lambda prompt: prompt.sequence if prompt.sequence is not None else 0
            )

    def render_template_value(self, **kwargs):
        """Renders self.value as a template, applying provided parameters in kwargs

        Args:
            kwargs:Key-value pairs to replace in the SeedPromptGroup value.

        Returns:
            None

        Raises:
            ValueError: If parameters are missing or invalid in the template.
        """

        for prompt in self.prompts:
            prompt.value = prompt.render_template_value(**kwargs)

    def _enforce_consistent_group_id(self):
        """
        Ensures that if any of the prompts already have a group ID set,
        they share the same ID. If none have a group ID set, assign a
        new UUID to all prompts.

        Raises:
            ValueError: If multiple different group IDs exist among the prompts.
        """
        existing_group_ids = {prompt.prompt_group_id for prompt in self.prompts if prompt.prompt_group_id is not None}

        if len(existing_group_ids) > 1:
            # More than one distinct group ID found among prompts.
            raise ValueError("Inconsistent group IDs found across prompts.")
        elif len(existing_group_ids) == 1:
            # Exactly one group ID is set; apply it to all.
            group_id = existing_group_ids.pop()
            for prompt in self.prompts:
                prompt.prompt_group_id = group_id
        else:
            # No group IDs set; generate a fresh one and assign it to all.
            new_group_id = uuid.uuid4()
            for prompt in self.prompts:
                prompt.prompt_group_id = new_group_id

    def _enforce_consistent_role(self):
        """
        Ensures that all prompts in the group that share a sequence have a consistent role.
        If no roles are set, all prompts will be assigned the default 'user' role.
        If only one prompt in a sequence has a role, all prompts will be assigned that role.
        Roles must be set if there is more than one sequence and within a sequence all
        roles must be consistent.

        Raises:
            ValueError: If multiple different roles are found across prompts in the group  or
            if no roles are set in a multi-sequence group.
        """
        # groups the prompts according to their sequence
        grouped_prompts = defaultdict(list)
        for prompt in self.prompts:
            if prompt.sequence not in grouped_prompts:
                grouped_prompts[prompt.sequence] = []
            grouped_prompts[prompt.sequence].append(prompt)

        num_sequences = len(grouped_prompts)
        for sequence, prompts in grouped_prompts.items():
            roles = {prompt.role for prompt in prompts if prompt.role is not None}
            if not len(roles) and num_sequences > 1:
                raise ValueError(
                    f"No roles set for sequence {sequence} in a multi-sequence group. "
                    "Please ensure at least one prompt within a sequence has an assigned role."
                )
            if len(roles) > 1:
                raise ValueError(f"Inconsistent roles found for sequence {sequence}: {roles}")
            role = roles.pop() if len(roles) else "user"
            for prompt in prompts:
                prompt.role = role

    def _set_objective_from_prompt(self, objective_prompt: Optional[SeedObjective] = None):
        """Sets the objective from the prompt marked as objective and removes it from prompts list."""
        prompt = objective_prompt if objective_prompt else self.prompts[-1]
        if self.objective is not None:
            raise ValueError("SeedPromptGroups can only have one objective.")
        self.objective = SeedObjective(
            value=prompt.value,
            value_sha256=prompt.value_sha256,
            # Note: all entries in database must have unique IDs
            # generate a new UUID for the objective
            id=uuid.uuid4(),
            data_type=prompt.data_type,
            name=prompt.name,
            dataset_name=prompt.dataset_name,
            authors=prompt.authors,
            groups=prompt.groups,
            source=prompt.source,
            added_by=prompt.added_by,
            harm_categories=prompt.harm_categories,
            metadata=prompt.metadata,
        )
        return

    def is_single_request(self) -> bool:
        unique_sequences = {prompt.sequence for prompt in self.prompts}
        return len(unique_sequences) <= 1

    def is_single_part_single_text_request(self) -> bool:
        return len(self.prompts) == 1 and self.prompts[0].data_type == "text"

    def __repr__(self):
        return f"<SeedPromptGroup(prompts={len(self.prompts)} prompts)>"
