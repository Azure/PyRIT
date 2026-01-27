# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Converter-related request and response models.

Converters have two concepts:
- Types: Static metadata bundled with frontend (from registry)
- Instances: Runtime objects created via API with specific configuration

This module defines both the Instance models and preview functionality.
Nested converters (e.g., SelectiveTextConverter wrapping Base64Converter) are supported.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from pyrit.backend.models.registry import ConverterMetadataResponse
from pyrit.models import PromptDataType

__all__ = [
    "ConverterMetadataResponse",
    "ConverterInstance",
    "ConverterInstanceListResponse",
    "CreateConverterRequest",
    "CreateConverterResponse",
    "InlineConverterConfig",
    "NestedConverterConfig",
    "ConverterPreviewRequest",
    "ConverterPreviewResponse",
    "PreviewStep",
]


# ============================================================================
# Converter Instances (Runtime Objects)
# ============================================================================


class InlineConverterConfig(BaseModel):
    """Inline converter configuration (type + params)."""

    type: str = Field(..., description="Converter type (e.g., 'base64', 'translation')")
    params: Dict[str, Any] = Field(default_factory=dict, description="Converter parameters")


class NestedConverterConfig(BaseModel):
    """
    Converter config that may contain nested converters.

    Used for composite converters like SelectiveTextConverter that wrap other converters.
    The 'converter' param can contain another NestedConverterConfig.
    """

    type: str = Field(..., description="Converter type")
    params: Dict[str, Any] = Field(default_factory=dict, description="Converter parameters")


class ConverterInstance(BaseModel):
    """A registered converter instance."""

    converter_id: str = Field(..., description="Unique converter instance identifier")
    type: str = Field(..., description="Converter type (e.g., 'base64', 'translation')")
    display_name: Optional[str] = Field(None, description="Human-readable display name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Converter parameters (resolved)")
    created_at: datetime = Field(..., description="Creation timestamp")
    source: Literal["initializer", "user"] = Field(..., description="How the converter was created")


class ConverterInstanceListResponse(BaseModel):
    """Response for listing converter instances."""

    items: List[ConverterInstance] = Field(..., description="List of converter instances")


class CreateConverterRequest(BaseModel):
    """
    Request to create a new converter instance.

    Supports nested converters - if params contains a 'converter' key with
    an InlineConverterConfig, the backend will create both and link them.
    """

    type: str = Field(..., description="Converter type (e.g., 'base64', 'translation')")
    display_name: Optional[str] = Field(None, description="Human-readable display name")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Converter parameters (may include nested 'converter' config)",
    )


class CreateConverterResponse(BaseModel):
    """Response after creating a converter instance."""

    converter_id: str = Field(..., description="Unique converter instance identifier")
    type: str = Field(..., description="Converter type")
    display_name: Optional[str] = Field(None, description="Human-readable display name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Resolved parameters (nested converters have IDs)")
    created_converters: Optional[List[ConverterInstance]] = Field(
        None, description="All converters created (including nested), ordered inner-to-outer"
    )
    created_at: datetime = Field(..., description="Creation timestamp")
    source: Literal["user"] = Field(default="user", description="Source is always 'user' for API-created")


# ============================================================================
# Converter Preview
# ============================================================================


class PreviewStep(BaseModel):
    """A single step in the conversion preview."""

    converter_id: Optional[str] = Field(None, description="Converter instance ID (if using ID)")
    converter_type: str = Field(..., description="Converter type")
    input_value: str = Field(..., description="Input to this converter")
    input_data_type: PromptDataType = Field(..., description="Input data type")
    output_value: str = Field(..., description="Output from this converter")
    output_data_type: PromptDataType = Field(..., description="Output data type")


class ConverterPreviewRequest(BaseModel):
    """Request to preview converter transformation."""

    original_value: str = Field(..., description="Text to convert")
    original_value_data_type: PromptDataType = Field(default="text", description="Data type of original value")
    converter_ids: Optional[List[str]] = Field(None, description="Converter instance IDs to apply")
    converters: Optional[List[InlineConverterConfig]] = Field(None, description="Inline converter definitions")


class ConverterPreviewResponse(BaseModel):
    """Response from converter preview."""

    original_value: str = Field(..., description="Original input text")
    original_value_data_type: PromptDataType = Field(..., description="Data type of original value")
    converted_value: str = Field(..., description="Final converted text")
    converted_value_data_type: PromptDataType = Field(..., description="Data type of converted value")
    steps: List[PreviewStep] = Field(..., description="Step-by-step conversion results")
