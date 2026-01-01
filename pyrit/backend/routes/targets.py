# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Target configuration endpoints.
"""

from typing import Any, Dict, List

from fastapi import APIRouter

router = APIRouter()


@router.get("/types")
async def get_target_types() -> List[Dict[str, Any]]:
    """
    Get available PyRIT target types with their default environment variables.

    Returns:
        List[Dict[str, Any]]: List of target type configurations with default env vars.
    """
    # Return OpenAI-compatible targets with their default environment variable names
    target_types = [
        {
            "id": "OpenAIChatTarget",
            "name": "OpenAI Chat",
            "description": "OpenAI API-compatible chat endpoint (works with Azure OpenAI, OpenAI, and compatible APIs)",
            "default_env_vars": {
                "endpoint": "OPENAI_CHAT_ENDPOINT",
                "api_key": "OPENAI_CHAT_KEY",
                "model": "OPENAI_CHAT_MODEL",
            },
        },
        {
            "id": "OpenAIImageTarget",
            "name": "OpenAI Image",
            "description": "OpenAI API-compatible image generation endpoint",
            "default_env_vars": {
                "endpoint": "OPENAI_IMAGE_ENDPOINT",
                "api_key": "OPENAI_IMAGE_API_KEY",
                "model": "OPENAI_IMAGE_MODEL",
            },
        },
        {
            "id": "OpenAITTSTarget",
            "name": "OpenAI Text-to-Speech",
            "description": "OpenAI API-compatible text-to-speech endpoint",
            "default_env_vars": {
                "endpoint": "OPENAI_TTS_ENDPOINT",
                "api_key": "OPENAI_TTS_KEY",
                "model": "OPENAI_TTS_MODEL",
            },
        },
        {
            "id": "OpenAIVideoTarget",
            "name": "OpenAI Video",
            "description": "OpenAI API-compatible video generation endpoint",
            "default_env_vars": {
                "endpoint": "OPENAI_VIDEO_ENDPOINT",
                "api_key": "OPENAI_VIDEO_KEY",
                "model": "OPENAI_VIDEO_MODEL",
            },
        },
        {
            "id": "RealtimeTarget",
            "name": "OpenAI Realtime",
            "description": "OpenAI API-compatible realtime API endpoint",
            "default_env_vars": {
                "endpoint": "OPENAI_REALTIME_ENDPOINT",
                "api_key": "OPENAI_REALTIME_API_KEY",
                "model": "OPENAI_REALTIME_MODEL",
            },
        },
        {
            "id": "OpenAIResponseTarget",
            "name": "OpenAI Response",
            "description": "OpenAI API-compatible response endpoint for structured outputs",
            "default_env_vars": {
                "endpoint": "OPENAI_RESPONSES_ENDPOINT",
                "api_key": "OPENAI_RESPONSES_KEY",
                "model": "OPENAI_RESPONSES_MODEL",
            },
        },
    ]

    return target_types
