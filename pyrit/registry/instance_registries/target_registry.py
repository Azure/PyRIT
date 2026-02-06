# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Target registry for managing PyRIT target instances.

Targets are registered explicitly via initializers as pre-configured instances.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from pyrit.identifiers import TargetIdentifier
from pyrit.registry.instance_registries.base_instance_registry import (
    BaseInstanceRegistry,
)

if TYPE_CHECKING:
    from pyrit.prompt_target import PromptTarget

logger = logging.getLogger(__name__)


class TargetRegistry(BaseInstanceRegistry["PromptTarget", TargetIdentifier]):
    """
    Registry for managing available target instances.

    This registry stores pre-configured PromptTarget instances (not classes).
    Targets are registered explicitly via initializers after being instantiated
    with their required parameters.
    """

    @classmethod
    def get_registry_singleton(cls) -> TargetRegistry:
        """
        Get the singleton instance of the TargetRegistry.

        Returns:
            The singleton TargetRegistry instance.
        """
        return super().get_registry_singleton()  # type: ignore[return-value]

    def register_instance(
        self,
        target: PromptTarget,
        *,
        name: Optional[str] = None,
    ) -> None:
        """
        Register a target instance.

        Args:
            target: The pre-configured target instance (not a class).
            name: Optional custom registry name. If not provided,
                uses the target's identifier unique_name.
        """
        if name is None:
            name = target.get_identifier().unique_name

        self.register(target, name=name)
        logger.debug(f"Registered target instance: {name} ({target.__class__.__name__})")

    def get_instance_by_name(self, name: str) -> Optional[PromptTarget]:
        """
        Get a registered target instance by name.

        Args:
            name: The registry name of the target.

        Returns:
            The target instance, or None if not found.
        """
        return self.get(name)

    def _build_metadata(self, name: str, instance: PromptTarget) -> TargetIdentifier:
        """
        Build metadata for a target instance.

        Args:
            name: The registry name of the target.
            instance: The target instance.

        Returns:
            TargetIdentifier from the target's get_identifier() method.
        """
        return instance.get_identifier()
