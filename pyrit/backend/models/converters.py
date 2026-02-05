# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Converter-related request and response models.

This module defines the Instance models and preview functionality.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from pyrit.models import PromptDataType

__all__ = [
    "ConverterInstance",
    "ConverterInstanceListResponse",
    "CreateConverterRequest",
    "CreateConverterResponse",
    "ConverterPreviewRequest",
    "ConverterPreviewResponse",
    "PreviewStep",
]


# ============================================================================
# Converter Instances (Runtime Objects)
# ============================================================================


class ConverterInstance(BaseModel):
    """A registered converter instance."""

    converter_id: str = Field(..., description="Unique converter instance identifier")
    type: str = Field(..., description="Converter type (e.g., 'base64', 'translation')")
    display_name: Optional[str] = Field(None, description="Human-readable display name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Converter parameters (resolved)")


class ConverterInstanceListResponse(BaseModel):
    """Response for listing converter instances."""

    items: List[ConverterInstance] = Field(..., description="List of converter instances")


class CreateConverterRequest(BaseModel):
    """Request to create a new converter instance."""

    type: str = Field(..., description="Converter type (e.g., 'base64', 'translation')")
    display_name: Optional[str] = Field(None, description="Human-readable display name")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Converter parameters",
    )


class CreateConverterResponse(BaseModel):
    """Response after creating a converter instance."""

    converter_id: str = Field(..., description="Unique converter instance identifier")
    type: str = Field(..., description="Converter type")
    display_name: Optional[str] = Field(None, description="Human-readable display name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Converter parameters")


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
    converter_ids: List[str] = Field(..., description="Converter instance IDs to apply")


class ConverterPreviewResponse(BaseModel):
    """Response from converter preview."""

    original_value: str = Field(..., description="Original input text")
    original_value_data_type: PromptDataType = Field(..., description="Data type of original value")
    converted_value: str = Field(..., description="Final converted text")
    converted_value_data_type: PromptDataType = Field(..., description="Data type of converted value")
    steps: List[PreviewStep] = Field(..., description="Step-by-step conversion results")
