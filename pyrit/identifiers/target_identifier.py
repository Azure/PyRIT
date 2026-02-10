# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, cast

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

    supports_conversation_history: bool = False
    """Whether the target supports explicit setting of conversation history (is a PromptChatTarget)."""

    target_specific_params: Optional[Dict[str, Any]] = None
    """Additional target-specific parameters."""

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
