# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Target registry for discovering and managing PyRIT prompt targets.

Targets are registered explicitly via initializers as pre-configured instances.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

from pyrit.registry.base import RegistryItemMetadata
from pyrit.registry.instance_registries.base_instance_registry import (
    BaseInstanceRegistry,
)
from pyrit.registry.name_utils import class_name_to_registry_name

if TYPE_CHECKING:
    from pyrit.prompt_target import PromptTarget

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetMetadata(RegistryItemMetadata):
    """
    Metadata describing a registered target instance.

    Unlike ScenarioMetadata/InitializerMetadata which describe classes,
    TargetMetadata describes an already-instantiated prompt target.

    Use get() to retrieve the actual target instance.
    """

    target_identifier: Dict[str, Any]


class TargetRegistry(BaseInstanceRegistry["PromptTarget", TargetMetadata]):
    """
    Registry for managing available prompt target instances.

    This registry stores pre-configured PromptTarget instances (not classes).
    Targets are registered explicitly via initializers after being instantiated
    with their required parameters (e.g., endpoint, API keys).

    Targets are identified by their snake_case name derived from the class name,
    or a custom name provided during registration.
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

        Note: Unlike ScenarioRegistry and InitializerRegistry which register classes,
        TargetRegistry registers pre-configured instances.

        Args:
            target: The pre-configured target instance (not a class).
            name: Optional custom registry name. If not provided,
                derived from class name with identifier hash appended
                (e.g., OpenAIChatTarget -> openai_chat_abc123).
        """
        if name is None:
            base_name = class_name_to_registry_name(target.__class__.__name__, suffix="Target")
            # Append identifier hash for uniqueness
            identifier_hash = self._compute_identifier_hash(target)[:8]
            name = f"{base_name}_{identifier_hash}"

        self.register(target, name=name)
        logger.debug(f"Registered target instance: {name} ({target.__class__.__name__})")

    def get_instance_by_name(self, name: str) -> Optional["PromptTarget"]:
        """
        Get a registered target instance by name.

        Note: This returns an already-instantiated target, not a class.

        Args:
            name: The registry name of the target.

        Returns:
            The target instance, or None if not found.
        """
        return self.get(name)

    def _build_metadata(self, name: str, instance: "PromptTarget") -> TargetMetadata:
        """
        Build metadata for a target instance.

        Args:
            name: The registry name of the target.
            instance: The target instance.

        Returns:
            TargetMetadata describing the target.
        """
        # Get description from docstring
        doc = instance.__class__.__doc__ or ""
        description = " ".join(doc.split()) if doc else "No description available"

        # Get identifier from the target
        target_identifier = instance.get_identifier()

        return TargetMetadata(
            name=name,
            class_name=instance.__class__.__name__,
            description=description,
            target_identifier=target_identifier,
        )

    @staticmethod
    def _compute_identifier_hash(target: "PromptTarget") -> str:
        """
        Compute a hash from the target's identifier for unique naming.

        Args:
            target: The target instance.

        Returns:
            A hex string hash of the identifier.
        """
        identifier = target.get_identifier()
        identifier_str = json.dumps(identifier, sort_keys=True)
        return hashlib.sha256(identifier_str.encode()).hexdigest()
