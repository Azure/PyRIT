# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Scenarios API routes.

Provides endpoints for listing available scenarios.
"""

from typing import List

from fastapi import APIRouter

from pyrit.backend.models.registry import ScenarioMetadataResponse
from pyrit.backend.services import get_registry_service

router = APIRouter(prefix="/scenarios", tags=["scenarios"])


@router.get(
    "",
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
