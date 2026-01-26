# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union, cast

from pyrit.common.deprecation import print_deprecation_message
from pyrit.identifiers.identifier import MAX_STORAGE_LENGTH, Identifier
from pyrit.models.score import ScoreType


@dataclass(frozen=True)
class ScorerIdentifier(Identifier):
    """
    Identifier for Scorer instances.

    This frozen dataclass extends Identifier with scorer-specific fields.
    Long prompt templates are automatically truncated for storage display.

    Attributes:
        scorer_type: The type of scorer ("true_false", "float_scale", or "unknown").
        system_prompt_template: The system prompt template used by the scorer.
            Truncated for storage if > 100 characters.
        user_prompt_template: The user prompt template used by the scorer.
            Truncated for storage if > 100 characters.
        sub_identifier: List of sub-scorer identifiers for composite scorers.
        target_info: Information about the prompt target used by the scorer.
        score_aggregator: The name of the score aggregator function.
        scorer_specific_params: Additional scorer-specific parameters.
    """

    scorer_type: ScoreType = "unknown"
    system_prompt_template: Optional[str] = field(default=None, metadata={MAX_STORAGE_LENGTH: 100})
    user_prompt_template: Optional[str] = field(default=None, metadata={MAX_STORAGE_LENGTH: 100})
    sub_identifier: Optional[List["ScorerIdentifier"]] = None
    target_info: Optional[Dict[str, Any]] = None
    score_aggregator: Optional[str] = None
    scorer_specific_params: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls: Type["ScorerIdentifier"], data: dict[str, Any]) -> "ScorerIdentifier":
        """
        Create a ScorerIdentifier from a dictionary (e.g., retrieved from database).

        Extends the base Identifier.from_dict() to recursively reconstruct
        nested ScorerIdentifier objects in sub_identifier.

        Args:
            data: The dictionary representation.

        Returns:
            ScorerIdentifier: A new ScorerIdentifier instance.
        """
        # Create a mutable copy
        data = dict(data)

        # Recursively reconstruct sub_identifier if present
        if "sub_identifier" in data and data["sub_identifier"] is not None:
            data["sub_identifier"] = [
                ScorerIdentifier.from_dict(sub) if isinstance(sub, dict) else sub for sub in data["sub_identifier"]
            ]

        # Delegate to parent class for standard processing
        result = Identifier.from_dict.__func__(cls, data)  # type: ignore[attr-defined]
        return cast(ScorerIdentifier, result)

    @classmethod
    def normalize(
        cls,
        value: Union["ScorerIdentifier", Dict[str, Any], None],
    ) -> "ScorerIdentifier":
        """
        Normalize a value to a ScorerIdentifier.

        This method handles conversion from legacy dict format to ScorerIdentifier,
        emitting a deprecation warning when a dict is passed.

        Args:
            value: A ScorerIdentifier, a dict (legacy format), or None.

        Returns:
            ScorerIdentifier: The normalized ScorerIdentifier. Returns a default
                ScorerIdentifier with "Unknown" class_name if value is None or empty.

        Raises:
            TypeError: If value is not a ScorerIdentifier, dict, or None.
        """
        if value is None or (isinstance(value, dict) and not value):
            # Create default ScorerIdentifier for None/empty values
            return cls(
                class_name="Unknown",
                class_module="unknown",
                class_description="",
                identifier_type="instance",
            )

        if isinstance(value, dict):
            print_deprecation_message(
                old_item="dict for scorer_class_identifier",
                new_item="ScorerIdentifier",
                removed_in="0.13.0",
            )
            return cls.from_dict(value)

        if isinstance(value, cls):
            return value

        raise TypeError(f"Expected ScorerIdentifier, dict, or None, got {type(value).__name__}")
