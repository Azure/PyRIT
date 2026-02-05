# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Target registry for managing PyRIT target instances.

Targets are registered explicitly via initializers as pre-configured instances.

NOTE: This is a placeholder implementation. PR #1320 will add the full implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from pyrit.identifiers import Identifier
from pyrit.identifiers.class_name_utils import class_name_to_snake_case
from pyrit.registry.instance_registries.base_instance_registry import (
    BaseInstanceRegistry,
)

if TYPE_CHECKING:
    from pyrit.prompt_target import PromptTarget

logger = logging.getLogger(__name__)


# Placeholder identifier type until proper TargetIdentifier is defined
# TODO: Replace with TargetIdentifier when available
@dataclass(frozen=True)
class TargetIdentifier(Identifier):
    """Temporary identifier type for targets."""

    pass


class TargetRegistry(BaseInstanceRegistry["PromptTarget", TargetIdentifier]):
    """
    Registry for managing available target instances.

    This registry stores pre-configured PromptTarget instances (not classes).
    Targets are registered explicitly via initializers after being instantiated
    with their required parameters.

    NOTE: This is a placeholder. PR #1320 will add the full implementation.
    """

    @classmethod
    def get_registry_singleton(cls) -> "TargetRegistry":
        """
        Get the singleton instance of the TargetRegistry.

        Returns:
            The singleton TargetRegistry instance.
        """
        return super().get_registry_singleton()  # type: ignore[return-value]

    def register_instance(
        self,
        target: "PromptTarget",
        *,
        name: Optional[str] = None,
    ) -> None:
        """
        Register a target instance.

        Args:
            target: The pre-configured target instance (not a class).
            name: Optional custom registry name. If not provided,
                derived from class name (e.g., AzureOpenAIGPT4OChatTarget -> azure_openai_gpt4o_chat).
        """
        if name is None:
            name = class_name_to_snake_case(target.__class__.__name__, suffix="Target")

        self.register(target, name=name)
        logger.debug(f"Registered target instance: {name} ({target.__class__.__name__})")

    def get_instance_by_name(self, name: str) -> Optional["PromptTarget"]:
        """
        Get a registered target instance by name.

        Args:
            name: The registry name of the target.

        Returns:
            The target instance, or None if not found.
        """
        return self.get(name)

    def _build_metadata(self, name: str, instance: "PromptTarget") -> TargetIdentifier:
        """
        Build metadata for a target instance.

        Args:
            name: The registry name of the target.
            instance: The target instance.

        Returns:
            TargetIdentifier with basic info about the target.
        """
        return TargetIdentifier(
            class_name=instance.__class__.__name__,
            class_module=instance.__class__.__module__,
            class_description=f"Target: {name}",
            identifier_type="instance",
        )
