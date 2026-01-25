# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Registry response models.

Models for targets, scenarios, scorers, converters, and initializers.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from pyrit.models import PromptDataType

# ============================================================================
# Common
# ============================================================================


class ParameterInfo(BaseModel):
    """Information about a constructor parameter."""

    name: str = Field(..., description="Parameter name")
    type_hint: Optional[str] = Field(None, description="Type hint as string")
    required: bool = Field(..., description="Whether parameter is required")
    default: Optional[Any] = Field(None, description="Default value if not required")


# ============================================================================
# Targets
# ============================================================================


class TargetMetadataResponse(BaseModel):
    """Metadata for a target type."""

    name: str = Field(..., description="Registry name")
    class_name: str = Field(..., description="Python class name")
    description: str = Field(..., description="Target description")
    is_chat_target: bool = Field(..., description="Whether target supports chat/system prompts")
    supports_json_response: bool = Field(..., description="Whether target supports JSON response format")
    supported_data_types: List[PromptDataType] = Field(..., description="Supported input data types")
    params_schema: Optional[Dict[str, Any]] = Field(None, description="Parameter schema")


class TargetListResponse(BaseModel):
    """Response containing list of available targets."""

    targets: List[TargetMetadataResponse] = Field(..., description="Available target types")


# ============================================================================
# Scenarios
# ============================================================================


class ScenarioMetadataResponse(BaseModel):
    """Metadata for a scenario type."""

    name: str = Field(..., description="Registry name")
    class_name: str = Field(..., description="Python class name")
    description: str = Field(..., description="Scenario description")
    default_strategy: str = Field(..., description="Default strategy name")
    all_strategies: List[str] = Field(..., description="All available strategies")
    aggregate_strategies: List[str] = Field(..., description="Composite/aggregate strategies")
    default_datasets: List[str] = Field(..., description="Default dataset names")
    max_dataset_size: Optional[int] = Field(None, description="Maximum dataset size limit")


class ScenarioListResponse(BaseModel):
    """Response containing list of available scenarios."""

    scenarios: List[ScenarioMetadataResponse] = Field(..., description="Available scenarios")


# ============================================================================
# Scorers
# ============================================================================


class ScorerMetadataResponse(BaseModel):
    """Metadata for a registered scorer instance."""

    name: str = Field(..., description="Registry name")
    class_name: str = Field(..., description="Python class name")
    description: str = Field(..., description="Scorer description")
    scorer_type: str = Field(..., description="Score type (true_false or float_scale)")
    scorer_identifier: Dict[str, Any] = Field(..., description="Scorer identifier (filtered)")


class ScorerListResponse(BaseModel):
    """Response containing list of registered scorers."""

    scorers: List[ScorerMetadataResponse] = Field(..., description="Registered scorer instances")


# ============================================================================
# Initializers
# ============================================================================


class InitializerMetadataResponse(BaseModel):
    """Metadata for an initializer."""

    name: str = Field(..., description="Registry name")
    class_name: str = Field(..., description="Python class name")
    description: str = Field(..., description="Initializer description")
    required_env_vars: List[str] = Field(..., description="Required environment variables")
    execution_order: int = Field(..., description="Execution order priority")


class InitializerListResponse(BaseModel):
    """Response containing list of available initializers."""

    initializers: List[InitializerMetadataResponse] = Field(..., description="Available initializers")


# ============================================================================
# Converters
# ============================================================================


class ConverterMetadataResponse(BaseModel):
    """Metadata for a converter type."""

    name: str = Field(..., description="Registry name (snake_case)")
    class_name: str = Field(..., description="Python class name")
    description: str = Field(..., description="Converter description")
    supported_input_types: List[PromptDataType] = Field(..., description="Supported input data types")
    supported_output_types: List[PromptDataType] = Field(..., description="Supported output data types")
    is_llm_based: bool = Field(..., description="Whether converter requires LLM calls")
    is_deterministic: bool = Field(..., description="Whether same input produces same output")
    params_schema: Optional[Dict[str, Any]] = Field(None, description="Parameter schema")


class ConverterListResponse(BaseModel):
    """Response containing list of available converters."""

    converters: List[ConverterMetadataResponse] = Field(..., description="Available converter types")
