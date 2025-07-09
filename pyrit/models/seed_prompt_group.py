# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Sequence, Union

from pyrit.common.yaml_loadable import YamlLoadable
from pyrit.models.seed_prompt import SeedPrompt

logger = logging.getLogger(__name__)


class SeedPromptGroup(YamlLoadable):
    """
    A group of prompts that need to be sent together, along with an objective.

    This class is useful when a target requires multiple prompts to be grouped
    and sent together. All prompts in the group should share the same `prompt_group_id`. Within the group,
    there can be multiple prompts with the same `prompt_seed_alias`, which will be sent together in a single turn.

    """

    prompts: list[SeedPrompt]

    def __init__(
        self,
        *,
        prompts: Union[Sequence[SeedPrompt], Sequence[Dict[str, Any]]],
    ):
        if not prompts:
            raise ValueError("SeedPromptGroup cannot be empty.")
        self.prompts = []
        for prompt in prompts:
            if isinstance(prompt, SeedPrompt):
                self.prompts.append(prompt)
            elif isinstance(prompt, dict):
                self.prompts.append(SeedPrompt(**prompt))

        self._enforce_consistent_group_id()
        self._enforce_consistent_role()

        # check seed_alias and group the seed prompts by sequence if seed_alias is set
        if any(prompt.prompt_seed_alias for prompt in self.prompts):
            self.prompts = sorted(
                self.prompts,
                key=lambda prompt: (prompt.prompt_seed_id, prompt.sequence if prompt.sequence is not None else 0),
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
        Ensures that all prompts with the same prompt_seed_id have the same role.
        SeedPrompts that have the same prompt_seed_id are part of the same turn which
        is why they should have the same role.
        If they do not, raises a ValueError.

        Raises:
            ValueError: If multiple different roles exist among the prompts.
        """
        prompt_seed_id_to_role = {}
        for prompt in self.prompts:
            role = prompt.role
            prompt_seed_id = prompt.prompt_seed_id
            if prompt_seed_id in prompt_seed_id_to_role:
                if prompt_seed_id_to_role[prompt_seed_id] != role:
                    raise ValueError(f"Inconsistent roles found across prompts: {role}")
            else:
                prompt_seed_id_to_role[prompt_seed_id] = role

    def is_single_request(self) -> bool:
        unique_sequences = {prompt.sequence for prompt in self.prompts}
        return len(unique_sequences) <= 1

    def is_single_part_single_text_request(self) -> bool:
        return len(self.prompts) == 1 and self.prompts[0].data_type == "text"

    def __repr__(self):
        return f"<SeedPromptGroup(prompts={len(self.prompts)} prompts)>"
