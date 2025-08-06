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
    A group of prompts that need to be sent together, along with an objective. This can include multiturn and multimodal
    prompts.
    This class is useful when a target requires multiple prompt pieces to be grouped and sent together.
    All prompts in the group should share the same `prompt_group_id`.
    """

    prompts: Sequence[SeedPrompt]

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

        self._validate()

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

    def _validate(self):
        """
        Validates the prompts in the group share a consistent group ID and role.

        """
        self._enforce_consistent_group_id()
        self._enforce_consistent_role()

    def _enforce_consistent_attribute(self, attribute_name: str, default_value: Any = None):
        """
        Generic function to enforce consistency of an attribute across all prompts in the group.

        Args:
            attribute_name: The name of the attribute to enforce consistency for.
            default_value: The default value to use if no values are set. If None and no values exist,
                         a UUID will be generated (useful for IDs).

        Raises:
            ValueError: If multiple different values exist for the attribute among the prompts.
        """
        existing_values = {
            getattr(prompt, attribute_name) for prompt in self.prompts if getattr(prompt, attribute_name) is not None
        }

        if len(existing_values) > 1:
            raise ValueError(f"Inconsistent {attribute_name}s found across prompts in the group.")
        elif len(existing_values) == 1:
            # Exactly one value is set; apply it to all
            value = existing_values.pop()
        else:
            # No values set; use default or generate UUID
            value = default_value if default_value is not None else uuid.uuid4()

        # Apply the consistent value to all prompts
        for prompt in self.prompts:
            setattr(prompt, attribute_name, value)

    def _enforce_consistent_role(self):
        """
        Ensures that all prompts in the group have the same role.
        Raises:
            ValueError: If multiple different roles exist among the prompts.
        """
        self._enforce_consistent_attribute(
            attribute_name="role",
            default_value="user",
        )

    def _enforce_consistent_group_id(self):
        """
        Ensures that if any of the prompts already have a group ID set,
        they share the same ID. If none have a group ID set, assign a
        new UUID to all prompts.

        Raises:
            ValueError: If multiple different group IDs exist among the prompts.
        """
        self._enforce_consistent_attribute(
            attribute_name="prompt_group_id",
        )

    def is_single_request(self) -> bool:
        unique_sequences = {prompt.sequence for prompt in self.prompts}
        return len(unique_sequences) <= 1

    def is_single_part_single_text_request(self) -> bool:
        return len(self.prompts) == 1 and self.prompts[0].data_type == "text"

    def __repr__(self):
        return f"<SeedPromptGroup(prompts={len(self.prompts)} prompts)>"
