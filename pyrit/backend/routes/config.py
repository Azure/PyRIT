# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Any, Dict, List

from fastapi import APIRouter

router = APIRouter()


@router.get("/env-vars")
async def get_env_vars() -> Dict[str, List[str]]:
    """
    Get available environment variables for target configuration.

    Returns:
        Dict[str, List[str]]: Dictionary with categorized environment variable names:
            - keys: Variables containing KEY or SECRET (names only, never values)
            - endpoints: Variables containing ENDPOINT
            - models: Variables containing MODEL
    """
    # Get all environment variables
    all_vars = list(os.environ.keys())

    # Categorize variables
    # Key vars: contain KEY or SECRET (only names, never values)
    key_vars = sorted([v for v in all_vars if "KEY" in v or "SECRET" in v])

    # Non-sensitive vars: don't contain KEY or SECRET
    non_sensitive = [v for v in all_vars if "KEY" not in v and "SECRET" not in v]

    # Specific categories for endpoints and models
    endpoint_vars = sorted([v for v in non_sensitive if "ENDPOINT" in v])
    model_vars = sorted([v for v in non_sensitive if "MODEL" in v])

    return {
        "keys": key_vars,  # For API key field (names only)
        "endpoints": endpoint_vars,  # For endpoint field
        "models": model_vars,  # For model field
    }


@router.get("/env-vars/{var_name}")
async def get_env_var_value(var_name: str) -> Dict[str, Any]:
    """
    Get the value of a specific environment variable (not API keys).

    Args:
        var_name: Name of the environment variable to retrieve.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - name: Variable name
            - value: Variable value (None for sensitive variables)
            - masked: Whether the value was masked for security
            - exists: Whether the variable exists in environment
    """
    # Don't expose API keys
    if "key" in var_name.lower() or "api" in var_name.lower() or "secret" in var_name.lower():
        return {"name": var_name, "value": None, "masked": True, "exists": var_name in os.environ}

    value = os.getenv(var_name)
    return {"name": var_name, "value": value, "masked": False, "exists": value is not None}
