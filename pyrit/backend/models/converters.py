# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Converter-related request and response models.

This module defines the Instance models and preview functionality.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from pyrit.models import PromptDataType

__all__ = [
    "ConverterInstance",
    "ConverterInstanceListResponse",
    "ConverterParameterMetadata",
    "ConverterTypeMetadata",
    "ConverterTypeListResponse",
    "CreateConverterRequest",
    "CreateConverterResponse",
    "ConverterPreviewRequest",
    "ConverterPreviewResponse",
    "ConverterTypePreviewRequest",
    "PreviewStep",
]


# ============================================================================
# Converter Instances (Runtime Objects)
# ============================================================================


class ConverterInstance(BaseModel):
    """A registered converter instance."""

    converter_id: str = Field(..., description="Unique converter instance identifier")
    converter_type: str = Field(..., description="Converter class name (e.g., 'Base64Converter')")
    display_name: Optional[str] = Field(None, description="Human-readable display name")
    supported_input_types: list[str] = Field(
        default_factory=list, description="Input data types supported by this converter"
    )
    supported_output_types: list[str] = Field(
        default_factory=list, description="Output data types produced by this converter"
    )
    converter_specific_params: Optional[dict[str, Any]] = Field(
        None, description="Additional converter-specific parameters"
    )
    sub_converter_ids: Optional[list[str]] = Field(
        None, description="Converter IDs of sub-converters (for pipelines/composites)"
    )


class ConverterInstanceListResponse(BaseModel):
    """Response for listing converter instances."""

    items: list[ConverterInstance] = Field(..., description="List of converter instances")


class ConverterParameterMetadata(BaseModel):
    """Lightweight metadata for a converter constructor parameter."""

    name: str = Field(..., description="Parameter name")
    display_name: str = Field(..., description="Friendly parameter label")
    type_label: str = Field(..., description="Readable type label")
    required: bool = Field(..., description="Whether the parameter must be provided")
    default_value: Optional[str] = Field(None, description="String form of the default value")
    input_kind: Literal["text", "number", "boolean", "select", "list", "unsupported"] = Field(
        ..., description="Suggested UI control type"
    )
    options: Optional[list[str]] = Field(None, description="Allowed values for select-style parameters")


class ConverterTypeMetadata(BaseModel):
    """Metadata describing an available converter type."""

    converter_type: str = Field(..., description="Converter class name (e.g., 'Base64Converter')")
    display_name: str = Field(..., description="Friendly converter name")
    description: str = Field(..., description="Plain-language summary of the converter")
    supported_input_types: list[str] = Field(..., description="Supported input data types")
    supported_output_types: list[str] = Field(..., description="Supported output data types")
    parameters: list[ConverterParameterMetadata] = Field(..., description="Constructor parameters")
    preview_supported: bool = Field(..., description="Whether the UI can preview this converter directly")
    preview_unavailable_reason: Optional[str] = Field(
        None, description="Why direct preview is not available for this converter"
    )


class ConverterTypeListResponse(BaseModel):
    """Response listing all available converter types."""

    items: list[ConverterTypeMetadata] = Field(..., description="List of available converter types")


class CreateConverterRequest(BaseModel):
    """Request to create a new converter instance."""

    type: str = Field(..., description="Converter type (e.g., 'Base64Converter')")
    display_name: Optional[str] = Field(None, description="Human-readable display name")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Converter constructor parameters",
    )


class CreateConverterResponse(BaseModel):
    """Response after creating a converter instance."""

    converter_id: str = Field(..., description="Unique converter instance identifier")
    converter_type: str = Field(..., description="Converter class name")
    display_name: Optional[str] = Field(None, description="Human-readable display name")


# ============================================================================
# Converter Preview
# ============================================================================


class PreviewStep(BaseModel):
    """A single step in the conversion preview."""

    converter_id: str = Field(..., description="Converter instance ID")
    converter_type: str = Field(..., description="Converter type")
    input_value: str = Field(..., description="Input to this converter")
    input_data_type: PromptDataType = Field(..., description="Input data type")
    output_value: str = Field(..., description="Output from this converter")
    output_data_type: PromptDataType = Field(..., description="Output data type")


class ConverterPreviewRequest(BaseModel):
    """Request to preview converter transformation."""

    original_value: str = Field(..., description="Text to convert")
    original_value_data_type: PromptDataType = Field(default="text", description="Data type of original value")
    converter_ids: list[str] = Field(..., description="Converter instance IDs to apply")


class ConverterTypePreviewRequest(BaseModel):
    """Request to preview a converter directly from its type and params."""

    type: str = Field(..., description="Converter type (e.g., 'Base64Converter')")
    params: dict[str, Any] = Field(default_factory=dict, description="Converter constructor parameters")
    original_value: str = Field(..., description="Text to convert")
    original_value_data_type: PromptDataType = Field(default="text", description="Data type of original value")


class ConverterPreviewResponse(BaseModel):
    """Response from converter preview."""

    original_value: str = Field(..., description="Original input text")
    original_value_data_type: PromptDataType = Field(..., description="Data type of original value")
    converted_value: str = Field(..., description="Final converted text")
    converted_value_data_type: PromptDataType = Field(..., description="Data type of converted value")
    steps: list[PreviewStep] = Field(..., description="Step-by-step conversion results")
