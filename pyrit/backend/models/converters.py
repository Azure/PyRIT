# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Converter-related request and response models.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from pyrit.backend.models.conversations import ConverterConfig
from pyrit.backend.models.registry import ConverterMetadataResponse
from pyrit.models import PromptDataType

# Re-export for convenience
__all__ = [
    "ConverterMetadataResponse",
    "ConverterListResponse",
    "ConverterConfig",
    "ConversionStep",
    "PreviewConverterRequest",
    "PreviewConverterResponse",
]


class ConverterListResponse(BaseModel):
    """Response containing list of available converters."""

    converters: List[ConverterMetadataResponse] = Field(..., description="Available converter types")


class ConversionStep(BaseModel):
    """Single step in a conversion chain."""

    converter_class: str = Field(..., description="Converter class that was applied")
    input: str = Field(..., description="Input to this converter")
    input_data_type: PromptDataType = Field(..., description="Input data type")
    output: str = Field(..., description="Output from this converter")
    output_data_type: PromptDataType = Field(..., description="Output data type")


class PreviewConverterRequest(BaseModel):
    """Request to preview converter output."""

    content: str = Field(..., description="Original content to convert")
    data_type: PromptDataType = Field("text", description="Content data type")
    converters: List[ConverterConfig] = Field(..., description="Ordered list of converters to apply")


class PreviewConverterResponse(BaseModel):
    """Response with converter preview results."""

    original_content: str = Field(..., description="Original input content")
    converted_content: str = Field(..., description="Final converted content")
    converted_data_type: PromptDataType = Field(..., description="Final data type")
    conversion_chain: List[ConversionStep] = Field(..., description="Step-by-step conversion results")
    converter_identifiers: List[Dict[str, Any]] = Field(
        ..., description="Converter identifiers for use in pre_converted requests"
    )
