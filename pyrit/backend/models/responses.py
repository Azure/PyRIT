# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Response models for API endpoints
"""

from pydantic import BaseModel


class TargetInfo(BaseModel):
    """Information about a prompt target"""
    id: str
    name: str
    type: str
    description: str
    status: str
