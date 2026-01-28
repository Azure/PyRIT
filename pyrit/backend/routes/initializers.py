# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Initializers API routes.

Provides endpoints for listing available initializers.
"""

from typing import List

from fastapi import APIRouter

from pyrit.backend.models.registry import InitializerMetadataResponse
from pyrit.backend.services import get_registry_service

router = APIRouter(prefix="/initializers", tags=["initializers"])


@router.get(
    "",
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
