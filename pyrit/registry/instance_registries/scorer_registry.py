# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Scorer registry for discovering and managing PyRIT scorers.

scorers are registered explicitly via initializers as pre-configured instances.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from pyrit.registry.base import RegistryItemMetadata
from pyrit.registry.instance_registries.base_instance_registry import BaseInstanceRegistry
from pyrit.registry.name_utils import class_name_to_registry_name
from pyrit.score.float_scale.float_scale_scorer import FloatScaleScorer
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer

if TYPE_CHECKING:
    from pyrit.score.scorer import Scorer
    from pyrit.score.scorer_identifier import ScorerIdentifier

logger = logging.getLogger(__name__)


class ScorerMetadata(RegistryItemMetadata):
    """
    Metadata describing a registered scorer instance.

    Unlike ScenarioMetadata/InitializerMetadata which describe classes,
    ScorerMetadata describes an already-instantiated scorer.
    This TypedDict provides descriptive information, not the scorer itself.

    Use get() to retrieve the actual scorer instance.
    """

    scorer_type: str
    scorer_identifier: "ScorerIdentifier"


class ScorerRegistry(BaseInstanceRegistry["Scorer", ScorerMetadata]):
    """
    Registry for managing available scorer instances.

    This registry stores pre-configured Scorer instances (not classes).
    Scorers are registered explicitly via initializers after being instantiated
    with their required parameters (e.g., chat_target).

    Scorers are identified by their snake_case name derived from the class name,
    or a custom name provided during registration.
    """

    @classmethod
    def get_instance(cls) -> "ScorerRegistry":
        """Get the singleton instance of the ScorerRegistry."""
        return super().get_instance()  # type: ignore[return-value]

    def register_instance(
        self,
        scorer: "Scorer",
        *,
        name: Optional[str] = None,
    ) -> None:
        """
        Register a scorer instance.

        Note: Unlike ScenarioRegistry and InitializerRegistry which register classes,
        ScorerRegistry registers pre-configured instances.

        Args:
            scorer: The pre-configured scorer instance (not a class).
            name: Optional custom registry name. If not provided,
                derived from class name with scorer_identifier hash appended
                (e.g., SelfAskRefusalScorer -> self_ask_refusal_abc123).
        """
        if name is None:
            base_name = class_name_to_registry_name(scorer.__class__.__name__, suffix="Scorer")
            # Append scorer_identifier hash if available for uniqueness
            identifier_hash = scorer.scorer_identifier.compute_hash()[:8]
            name = f"{base_name}_{identifier_hash}"

        self.register(scorer, name=name)
        logger.debug(f"Registered scorer instance: {name} ({scorer.__class__.__name__})")

    def get_instance_by_name(self, name: str) -> Optional["Scorer"]:
        """
        Get a registered scorer instance by name.

        Note: This returns an already-instantiated scorer, not a class.

        Args:
            name: The registry name of the scorer.

        Returns:
            The scorer instance, or None if not found.
        """
        return self.get(name)

    def _build_metadata(self, name: str, instance: "Scorer") -> ScorerMetadata:
        """
        Build metadata for a scorer instance.

        Args:
            name: The registry name of the scorer.
            instance: The scorer instance.

        Returns:
            ScorerMetadata dictionary describing the scorer.
        """
        # Get description from docstring
        doc = instance.__class__.__doc__ or ""
        description = " ".join(doc.split()) if doc else "No description available"

        # Determine scorer_type from class hierarchy
        if isinstance(instance, TrueFalseScorer):
            scorer_type = "true_false"
        elif isinstance(instance, FloatScaleScorer):
            scorer_type = "float_scale"
        else:
            scorer_type = "unknown"

        return ScorerMetadata(
            name=name,
            class_name=instance.__class__.__name__,
            description=description,
            scorer_type=scorer_type,
            scorer_identifier=instance.scorer_identifier,
        )
