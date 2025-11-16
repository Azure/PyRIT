# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Target configuration endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import List

from pyrit.backend.models.responses import TargetInfo
from pyrit.backend.services.target_registry import TargetRegistry

router = APIRouter()


@router.get("/targets", response_model=List[TargetInfo])
async def list_targets():
    """
    List all available prompt targets from environment configuration
    """
    targets = TargetRegistry.get_available_targets()
    return [
        TargetInfo(
            id=t["id"],
            name=t["name"],
            type=t["type"],
            description=t["description"],
            status=t["status"],
        )
        for t in targets
    ]


@router.get("/targets/{target_id}", response_model=TargetInfo)
async def get_target(target_id: str):
    """
    Get information about a specific target
    """
    targets = await list_targets()
    target = next((t for t in targets if t.id == target_id), None)
    
    if not target:
        raise HTTPException(status_code=404, detail="Target not found")
    
    return target
