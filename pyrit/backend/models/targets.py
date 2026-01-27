# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Target instance models.

Targets have two concepts:
- Types: Static metadata bundled with frontend (from registry)
- Instances: Runtime objects created via API with specific configuration

This module defines the Instance models for runtime target management.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

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
    created_at: datetime = Field(..., description="Instance creation timestamp")
    source: Literal["initializer", "user"] = Field(..., description="How the target was created")


class TargetListResponse(BaseModel):
    """Response for listing target instances."""

    items: List[TargetInstance] = Field(..., description="List of target instances")


class CreateTargetRequest(BaseModel):
    """Request to create a new target instance."""

    type: str = Field(..., description="Target type (e.g., 'azure_openai', 'text_target')")
    display_name: Optional[str] = Field(None, description="Human-readable display name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Target constructor parameters")


class CreateTargetResponse(BaseModel):
    """Response after creating a target instance."""

    target_id: str = Field(..., description="Unique target instance identifier")
    type: str = Field(..., description="Target type")
    display_name: Optional[str] = Field(None, description="Human-readable display name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Filtered configuration (no secrets)")
    created_at: datetime = Field(..., description="Instance creation timestamp")
    source: Literal["user"] = Field(default="user", description="Source is always 'user' for API-created targets")
