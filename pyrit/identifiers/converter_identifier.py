# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, cast

from pyrit.identifiers.identifier import Identifier


@dataclass(frozen=True)
class ConverterIdentifier(Identifier):
    """
    Identifier for PromptConverter instances.

    This frozen dataclass extends Identifier with converter-specific fields.
    It provides a structured way to identify and track converters used in
    prompt transformations.
    """

    supported_input_types: Tuple[str, ...] = field(kw_only=True)
    """The input data types supported by this converter (e.g., ('text',), ('image', 'text'))."""

    supported_output_types: Tuple[str, ...] = field(kw_only=True)
    """The output data types produced by this converter."""

    sub_identifier: Optional[List["ConverterIdentifier"]] = None
    """List of sub-converter identifiers for composite converters like ConverterPipeline."""

    target_info: Optional[Dict[str, Any]] = None
    """Information about the prompt target used by the converter (for LLM-based converters)."""

    converter_specific_params: Optional[Dict[str, Any]] = None
    """Additional converter-specific parameters."""

    @classmethod
    def from_dict(cls: Type["ConverterIdentifier"], data: dict[str, Any]) -> "ConverterIdentifier":
        """
        Create a ConverterIdentifier from a dictionary (e.g., retrieved from database).

        Extends the base Identifier.from_dict() to recursively reconstruct
        nested ConverterIdentifier objects in sub_identifier.

        Args:
            data: The dictionary representation.

        Returns:
            ConverterIdentifier: A new ConverterIdentifier instance.
        """
        # Create a mutable copy
        data = dict(data)

        # Recursively reconstruct sub_identifier if present
        if "sub_identifier" in data and data["sub_identifier"] is not None:
            data["sub_identifier"] = [
                ConverterIdentifier.from_dict(sub) if isinstance(sub, dict) else sub for sub in data["sub_identifier"]
            ]

        # Convert supported_input_types and supported_output_types from list to tuple if needed
        if "supported_input_types" in data and data["supported_input_types"] is not None:
            if isinstance(data["supported_input_types"], list):
                data["supported_input_types"] = tuple(data["supported_input_types"])
        else:
            # Provide default for legacy dicts that don't have this field
            data["supported_input_types"] = ()

        if "supported_output_types" in data and data["supported_output_types"] is not None:
            if isinstance(data["supported_output_types"], list):
                data["supported_output_types"] = tuple(data["supported_output_types"])
        else:
            # Provide default for legacy dicts that don't have this field
            data["supported_output_types"] = ()

        # Delegate to parent class for standard processing
        result = Identifier.from_dict.__func__(cls, data)  # type: ignore[attr-defined]
        return cast(ConverterIdentifier, result)
