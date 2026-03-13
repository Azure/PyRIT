# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Builder API routes.
"""

from fastapi import APIRouter, HTTPException, status

from pyrit.backend.models.builder import (
    BuilderBuildRequest,
    BuilderBuildResponse,
    BuilderConfigResponse,
    ReferenceImageRequest,
    ReferenceImageResponse,
)
from pyrit.backend.models.common import ProblemDetail
from pyrit.backend.services.builder_service import get_builder_service

router = APIRouter(prefix="/builder", tags=["builder"])


@router.get(
    "/config",
    response_model=BuilderConfigResponse,
)
async def get_builder_config() -> BuilderConfigResponse:
    """Return prompt-bank and capability metadata for the builder UI."""
    return await get_builder_service().get_config_async()


@router.post(
    "/build",
    response_model=BuilderBuildResponse,
    responses={
        400: {"model": ProblemDetail, "description": "Invalid builder request"},
    },
)
async def build_builder_output(request: BuilderBuildRequest) -> BuilderBuildResponse:
    """Build prompt output for the current builder state."""
    try:
        return await get_builder_service().build_async(request=request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/reference-image",
    response_model=ReferenceImageResponse,
    responses={
        400: {"model": ProblemDetail, "description": "Reference image generation unavailable or invalid"},
    },
)
async def generate_reference_image(request: ReferenceImageRequest) -> ReferenceImageResponse:
    """Generate a reference image for the currently selected prompt variant."""
    try:
        return await get_builder_service().generate_reference_image_async(request=request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
