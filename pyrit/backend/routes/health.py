# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Health check endpoints
"""

from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Basic health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "pyrit-backend",
    }


@router.get("/version")
async def version():
    """
    Get API version information
    """
    return {
        "version": "0.10.0",
        "api_version": "v1",
    }
