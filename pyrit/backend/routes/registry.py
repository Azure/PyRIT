# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Registry API routes.

Provides endpoints for querying available components.
"""

from typing import List, Optional

from fastapi import APIRouter, Query

from pyrit.backend.models.registry import (
    ConverterMetadataResponse,
    InitializerMetadataResponse,
    ScenarioMetadataResponse,
    ScorerMetadataResponse,
    TargetMetadataResponse,
)
from pyrit.backend.services import get_registry_service

router = APIRouter(prefix="/registry", tags=["registry"])


@router.get(
    "/targets",
    response_model=List[TargetMetadataResponse],
)
async def list_targets(
    is_chat_target: Optional[bool] = Query(None, description="Filter by chat target support"),
) -> List[TargetMetadataResponse]:
    """
    List available targets.

    Returns metadata about all available prompt targets, optionally
    filtered by chat target support.

    Returns:
        List[TargetMetadataResponse]: List of target metadata.
    """
    service = get_registry_service()

    return service.get_targets(is_chat_target=is_chat_target)


@router.get(
    "/scenarios",
    response_model=List[ScenarioMetadataResponse],
)
async def list_scenarios() -> List[ScenarioMetadataResponse]:
    """
    List available scenarios.

    Returns metadata about all registered scenarios.

    Returns:
        List[ScenarioMetadataResponse]: List of scenario metadata.
    """
    service = get_registry_service()

    return service.get_scenarios()


@router.get(
    "/scorers",
    response_model=List[ScorerMetadataResponse],
)
async def list_scorers(
    scorer_type: Optional[str] = Query(None, description="Filter by scorer type (true_false or float_scale)"),
) -> List[ScorerMetadataResponse]:
    """
    List registered scorers.

    Returns metadata about all registered scorer instances.

    Returns:
        List[ScorerMetadataResponse]: List of scorer metadata.
    """
    service = get_registry_service()

    return service.get_scorers(scorer_type=scorer_type)


@router.get(
    "/converters",
    response_model=List[ConverterMetadataResponse],
)
async def list_converters(
    is_llm_based: Optional[bool] = Query(None, description="Filter by LLM-based converters"),
    is_deterministic: Optional[bool] = Query(None, description="Filter by deterministic converters"),
) -> List[ConverterMetadataResponse]:
    """
    List available converters.

    Returns metadata about all available prompt converters.
    Note: Also available at /converters endpoint.

    Returns:
        List[ConverterMetadataResponse]: List of converter metadata.
    """
    service = get_registry_service()

    return service.get_converters(
        is_llm_based=is_llm_based,
        is_deterministic=is_deterministic,
    )


@router.get(
    "/initializers",
    response_model=List[InitializerMetadataResponse],
)
async def list_initializers() -> List[InitializerMetadataResponse]:
    """
    List available initializers.

    Returns metadata about all registered initializers.

    Returns:
        List[InitializerMetadataResponse]: List of initializer metadata.
    """
    service = get_registry_service()

    return service.get_initializers()
