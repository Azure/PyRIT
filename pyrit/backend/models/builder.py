# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Builder-specific API models.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from pyrit.models import PromptDataType


class BuilderPresetField(BaseModel):
    """Field definition for a curated prompt-bank preset."""

    name: str = Field(..., description="Stable preset field name")
    label: str = Field(..., description="Human-readable label")
    placeholder: Optional[str] = Field(None, description="Suggested UI placeholder")
    required: bool = Field(True, description="Whether the field is required")
    default_value: Optional[str] = Field(None, description="Default value for the field")


class BuilderPreset(BaseModel):
    """Curated prompt-bank preset."""

    preset_id: str = Field(..., description="Stable preset identifier")
    family_id: str = Field(..., description="Family identifier for grouping")
    title: str = Field(..., description="Preset title")
    summary: str = Field(..., description="Short preset description")
    template: str = Field(..., description="Template string with named placeholders")
    fields: list[BuilderPresetField] = Field(default_factory=list, description="Fields required for the preset")


class BuilderPromptFamily(BaseModel):
    """Display metadata for a prompt-bank family."""

    family_id: str = Field(..., description="Stable family identifier")
    title: str = Field(..., description="Family title")
    summary: str = Field(..., description="Family description")
    preset_ids: list[str] = Field(default_factory=list, description="Preset ids belonging to the family")


class BuilderCapabilities(BaseModel):
    """Feature flags describing available builder capabilities."""

    reference_image_available: bool = Field(..., description="Whether reference-image generation is available")
    reference_image_target_name: Optional[str] = Field(None, description="Display name of the image helper target")


class BuilderDefaults(BaseModel):
    """Builder defaults returned to the frontend."""

    default_blocked_words: list[str] = Field(default_factory=list, description="Curated blocked-word list")
    max_variant_count: int = Field(..., description="Maximum number of prompt variants allowed")
    multi_variant_converter_types: list[str] = Field(
        default_factory=list,
        description="Converter types that support multiple text variants",
    )


class BuilderConfigResponse(BaseModel):
    """Builder configuration payload."""

    families: list[BuilderPromptFamily] = Field(default_factory=list, description="Prompt-bank families")
    presets: list[BuilderPreset] = Field(default_factory=list, description="Curated prompt-bank presets")
    defaults: BuilderDefaults = Field(..., description="Builder defaults")
    capabilities: BuilderCapabilities = Field(..., description="Capability flags")


class BuilderPipelineStep(BaseModel):
    """A single builder workflow step."""

    stage: Literal["preset", "blocked_words", "converter", "variants"] = Field(..., description="Workflow stage")
    title: str = Field(..., description="Short stage title")
    input_value: str = Field(..., description="Stage input")
    input_data_type: PromptDataType = Field(..., description="Stage input type")
    output_value: str = Field(..., description="Stage output")
    output_data_type: PromptDataType = Field(..., description="Stage output type")
    detail: Optional[str] = Field(None, description="Optional operator-facing detail")


class BuilderVariant(BaseModel):
    """One prompt version returned by the builder."""

    variant_id: str = Field(..., description="Stable id for client-side selection")
    label: str = Field(..., description="Display label")
    value: str = Field(..., description="Variant content")
    data_type: PromptDataType = Field(..., description="Variant data type")
    kind: Literal["base", "variation"] = Field(..., description="Whether this is the base output or a variation")


class BuilderBuildRequest(BaseModel):
    """Request to build prompt outputs for the current builder state."""

    source_content: str = Field(default="", description="Current source content shown in the builder")
    source_content_data_type: PromptDataType = Field(default="text", description="Data type of the source content")
    converter_type: str = Field(..., description="Selected converter type")
    converter_params: dict[str, Any] = Field(default_factory=dict, description="Selected converter parameters")
    preset_id: Optional[str] = Field(None, description="Optional curated preset identifier")
    preset_values: dict[str, str] = Field(default_factory=dict, description="Preset field values")
    avoid_blocked_words: bool = Field(False, description="Whether blocked-word avoidance should run")
    blocked_words: list[str] = Field(default_factory=list, description="Custom blocked-word list")
    variant_count: int = Field(1, ge=1, le=5, description="Total number of text variants requested")


class BuilderBuildResponse(BaseModel):
    """Response returned by the builder workflow."""

    resolved_source_value: str = Field(..., description="Effective source value used as input")
    resolved_source_data_type: PromptDataType = Field(..., description="Effective source data type")
    converted_value: str = Field(..., description="Primary converter output")
    converted_value_data_type: PromptDataType = Field(..., description="Primary converter output type")
    variants: list[BuilderVariant] = Field(default_factory=list, description="Built prompt variants")
    steps: list[BuilderPipelineStep] = Field(default_factory=list, description="Workflow trace")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal warnings")


class ReferenceImageRequest(BaseModel):
    """Request to generate a reference image from a text prompt."""

    prompt: str = Field(..., description="Prompt to use for image generation")


class ReferenceImageResponse(BaseModel):
    """Response from reference-image generation."""

    prompt: str = Field(..., description="Source prompt used for generation")
    image_path: str = Field(..., description="Stored local image path or remote URL")
    image_url: str = Field(..., description="Browser-safe URL for preview")
    data_type: Literal["image_path"] = Field("image_path", description="Returned media type")
    target_name: Optional[str] = Field(None, description="Display name of the image helper target")
