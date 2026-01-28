# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Scorers API routes.

Provides endpoints for listing available scorers.
"""

from typing import List, Optional

from fastapi import APIRouter, Query

from pyrit.backend.models.registry import ScorerMetadataResponse
from pyrit.backend.services import get_registry_service

router = APIRouter(prefix="/scorers", tags=["scorers"])


@router.get(
    "",
    response_model=List[ScorerMetadataResponse],
)
async def list_scorers(
    scorer_type: Optional[str] = Query(
        None, description="Filter by scorer type (true_false or float_scale)"
    ),
) -> List[ScorerMetadataResponse]:
    """
    List available scorers.

    Returns metadata about all registered scorer types.

    Returns:
        List[ScorerMetadataResponse]: List of scorer metadata.
    """
    service = get_registry_service()
    return service.get_scorers(scorer_type=scorer_type)
