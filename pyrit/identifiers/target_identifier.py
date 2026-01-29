# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, cast
from urllib.parse import urlparse

from pyrit.identifiers.identifier import Identifier


@dataclass(frozen=True)
class TargetIdentifier(Identifier):
    """
    Identifier for PromptTarget instances.

    This frozen dataclass extends Identifier with target-specific fields.
    It provides a stable, hashable identifier for prompt targets that can be
    used for scorer evaluation, registry tracking, and memory storage.
    """

    endpoint: str = ""
    """The target endpoint URL."""

    model_name: str = ""
    """The model or deployment name."""

    temperature: Optional[float] = None
    """The temperature parameter for generation."""

    top_p: Optional[float] = None
    """The top_p parameter for generation."""

    max_requests_per_minute: Optional[int] = None
    """Maximum number of requests per minute."""

    target_specific_params: Optional[Dict[str, Any]] = None
    """Additional target-specific parameters."""

    def __post_init__(self) -> None:
        """
        Compute derived fields with target-specific unique_name format.

        Overrides the base Identifier to include model_name and endpoint in unique_name.
        Format: {snake_name}::{model_name}::{endpoint_host}::{hash[:8]}
        Only includes model_name and endpoint if they have values.
        """
        # Call parent to set up snake_class_name and hash
        super().__post_init__()

        # Build unique_name with model_name and endpoint if available
        parts = [self.snake_class_name]

        if self.model_name:
            parts.append(self.model_name)

        if self.endpoint:
            # Simplify endpoint to just the host for readability
            try:
                parsed = urlparse(self.endpoint)
                host = parsed.netloc or self.endpoint
                parts.append(host)
            except Exception:
                # Fallback: truncate if parsing fails
                parts.append(self.endpoint[:20] if len(self.endpoint) > 20 else self.endpoint)

        parts.append(self.hash[:8])

        object.__setattr__(self, "unique_name", "::".join(parts))

    @classmethod
    def from_dict(cls: Type["TargetIdentifier"], data: dict[str, Any]) -> "TargetIdentifier":
        """
        Create a TargetIdentifier from a dictionary (e.g., retrieved from database).

        Extends the base Identifier.from_dict() to handle legacy key mappings.

        Args:
            data: The dictionary representation.

        Returns:
            TargetIdentifier: A new TargetIdentifier instance.
        """
        # Delegate to parent class for standard processing
        result = Identifier.from_dict.__func__(cls, data)  # type: ignore[attr-defined]
        return cast(TargetIdentifier, result)
