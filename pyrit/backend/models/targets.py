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


class TargetInstance(BaseModel):
    """
    A runtime target instance.

    Created either by an initializer (at startup) or by user (via API).
    """

    target_id: str = Field(..., description="Unique target instance identifier")
    type: str = Field(..., description="Target type (e.g., 'azure_openai', 'text_target')")
    display_name: Optional[str] = Field(None, description="Human-readable display name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Target configuration (sensitive fields filtered)")


class TargetListResponse(BaseModel):
    """Response for listing target instances."""

    items: List[TargetInstance] = Field(..., description="List of target instances")


class CreateTargetRequest(BaseModel):
    """Request to create a new target instance."""

    type: str = Field(..., description="Target type (e.g., 'OpenAIChatTarget')")
    display_name: Optional[str] = Field(None, description="Human-readable display name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Target constructor parameters")


class CreateTargetResponse(BaseModel):
    """Response after creating a target instance."""

    target_id: str = Field(..., description="Unique target instance identifier")
    type: str = Field(..., description="Target type")
    display_name: Optional[str] = Field(None, description="Human-readable display name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Filtered configuration (no secrets)")
