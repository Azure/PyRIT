# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from fastapi import APIRouter
from typing import List, Dict, Any
import os

router = APIRouter()


@router.get("/env-vars")
async def get_env_vars() -> Dict[str, List[str]]:
    """Get available environment variables for target configuration"""
    
    # Get all environment variables, sorted alphabetically
    all_vars = sorted(list(os.environ.keys()))
    
    return {
        "all": all_vars,
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
    """Get available PyRIT target types with their default env vars"""
    
    # Import here to avoid circular dependencies
    import inspect
    import pyrit.prompt_target as pt
    from pyrit.backend.services.target_registry import TargetRegistry
    
    target_types = []
    
    # Get all classes from pyrit.prompt_target
    for name, obj in inspect.getmembers(pt):
        if inspect.isclass(obj) and name.endswith('Target'):
            # Try to find matching env var mapping
            default_env_vars = {}
            for target_id, config in TargetRegistry.TARGET_ENV_MAPPINGS.items():
                if target_id == name or target_id in name:
                    default_env_vars = {
                        "api_key": config.get("key_var", ""),
                        "endpoint": config.get("endpoint_var", ""),
                        "model": config.get("model_var", ""),
                    }
                    break
            
            target_types.append({
                "name": name,
                "module": obj.__module__,
                "description": (obj.__doc__ or "").split('\n')[0].strip() if obj.__doc__ else "",
                "default_env_vars": default_env_vars,
            })
    
    return sorted(target_types, key=lambda x: x['name'])
