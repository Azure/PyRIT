# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from fastapi import APIRouter
from typing import List, Dict, Any
import os

router = APIRouter()


@router.get("/env-vars")
async def get_env_vars() -> Dict[str, List[str]]:
    """Get available environment variables for target configuration"""
    
    # Get all environment variables
    all_vars = list(os.environ.keys())
    
    # Filter by type
    endpoint_vars = sorted([v for v in all_vars if "ENDPOINT" in v])
    key_vars = sorted([v for v in all_vars if "KEY" in v or "API" in v])
    model_vars = sorted([v for v in all_vars if "MODEL" in v])
    
    return {
        "all": sorted(all_vars),
        "endpoints": endpoint_vars,
        "keys": key_vars,
        "models": model_vars,
    }


@router.get("/env-vars/{var_name}")
async def get_env_var_value(var_name: str) -> Dict[str, Any]:
    """Get the value of a specific environment variable (not API keys)"""
    
    # Don't expose API keys
    if 'key' in var_name.lower() or 'api' in var_name.lower() or 'secret' in var_name.lower():
        return {
            "name": var_name,
            "value": None,
            "masked": True,
            "exists": var_name in os.environ
        }
    
    value = os.getenv(var_name)
    return {
        "name": var_name,
        "value": value,
        "masked": False,
        "exists": value is not None
    }


@router.get("/target-types")
async def get_target_types() -> List[Dict[str, Any]]:
    """Get available PyRIT target types"""
    
    # Return OpenAI-compatible targets with their default environment variable names
    target_types = [
        {
            "id": "OpenAIChatTarget",
            "name": "OpenAI Chat",
            "description": "OpenAI-compatible chat endpoint (works with Azure OpenAI, OpenAI, and compatible APIs)",
            "default_env_vars": {
                "endpoint": "OPENAI_CHAT_ENDPOINT",
                "api_key": "OPENAI_CHAT_KEY",
                "model": "OPENAI_CHAT_MODEL"
            }
        },
        {
            "id": "OpenAIImageTarget",
            "name": "OpenAI Image (DALL-E)",
            "description": "OpenAI image generation endpoint (DALL-E)",
            "default_env_vars": {
                "endpoint": "OPENAI_IMAGE_ENDPOINT",
                "api_key": "OPENAI_IMAGE_API_KEY",
                "model": "OPENAI_IMAGE_MODEL"
            }
        },
        {
            "id": "OpenAITTSTarget",
            "name": "OpenAI Text-to-Speech",
            "description": "OpenAI text-to-speech endpoint",
            "default_env_vars": {
                "endpoint": "OPENAI_TTS_ENDPOINT",
                "api_key": "OPENAI_TTS_KEY",
                "model": "OPENAI_TTS_MODEL"
            }
        },
        {
            "id": "OpenAIVideoTarget",
            "name": "OpenAI Video",
            "description": "OpenAI video generation endpoint",
            "default_env_vars": {
                "endpoint": "OPENAI_VIDEO_ENDPOINT",
                "api_key": "OPENAI_VIDEO_KEY",
                "model": "OPENAI_VIDEO_MODEL"
            }
        },
        {
            "id": "RealtimeTarget",
            "name": "OpenAI Realtime",
            "description": "OpenAI realtime API endpoint",
            "default_env_vars": {
                "endpoint": "OPENAI_REALTIME_ENDPOINT",
                "api_key": "OPENAI_REALTIME_API_KEY",
                "model": "OPENAI_REALTIME_MODEL"
            }
        },
    ]
    
    return target_types
