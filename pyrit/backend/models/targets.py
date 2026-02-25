# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Target instance models.

Targets have two concepts:
- Types: Static metadata bundled with frontend (from registry)
- Instances: Runtime objects created via API with specific configuration

This module defines the Instance models for runtime target management.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from pyrit.backend.models.common import PaginationInfo


class TargetInstance(BaseModel):
    """
    A runtime target instance.

    Created either by an initializer (at startup) or by user (via API).
    Also used as the create-target response (same shape as GET).
    """

    target_unique_name: str = Field(
        ..., description="Unique target instance identifier (ComponentIdentifier.unique_name)"
    )
    target_type: str = Field(..., description="Target class name (e.g., 'OpenAIChatTarget')")
    endpoint: Optional[str] = Field(None, description="Target endpoint URL")
    model_name: Optional[str] = Field(None, description="Model or deployment name")
    temperature: Optional[float] = Field(None, description="Temperature parameter for generation")
    top_p: Optional[float] = Field(None, description="Top-p parameter for generation")
    max_requests_per_minute: Optional[int] = Field(None, description="Maximum requests per minute")
    target_specific_params: Optional[Dict[str, Any]] = Field(None, description="Additional target-specific parameters")


class TargetListResponse(BaseModel):
    """Response for listing target instances."""

    items: List[TargetInstance] = Field(..., description="List of target instances")
    pagination: PaginationInfo = Field(..., description="Pagination metadata")


class CreateTargetRequest(BaseModel):
    """Request to create a new target instance."""

    type: str = Field(..., description="Target type (e.g., 'OpenAIChatTarget')")
    params: Dict[str, Any] = Field(default_factory=dict, description="Target constructor parameters")
