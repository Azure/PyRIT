# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Target registry service for managing available PyRIT targets
"""

import os
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

import pyrit.prompt_target as pt


@dataclass
class TargetConfig:
    """Configuration for a target from environment variables"""

    target_type: str
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    deployment_name: Optional[str] = None


class TargetRegistry:
    """Registry of available PyRIT targets based on environment configuration"""

    @classmethod
    def get_available_targets(cls) -> List[Dict[str, Any]]:
        """
        Get list of available targets from environment variables.
        Returns simple filtered lists of endpoint/key/model env vars.
        """
        all_vars = list(os.environ.keys())
        
        endpoint_vars = sorted([v for v in all_vars if "ENDPOINT" in v and os.getenv(v)])
        key_vars = sorted([v for v in all_vars if ("KEY" in v or "API" in v) and os.getenv(v)])
        model_vars = sorted([v for v in all_vars if "MODEL" in v and os.getenv(v)])
        
        # Return as targets for compatibility
        targets = []
        for endpoint_var in endpoint_vars:
            endpoint_value = os.getenv(endpoint_var)
            # Simple name from var name
            name = endpoint_var.replace("_ENDPOINT", "").replace("_", " ").title()
            
            targets.append({
                "id": endpoint_var,
                "name": name,
                "type": "OpenAIChatTarget",
                "description": f"Endpoint: {endpoint_value}",
                "status": "available",
                "endpoint": endpoint_value,
                "endpoint_var": endpoint_var,
            })
        
        return targets

    @classmethod
    def create_target_instance(
        cls, 
        target_type: str = "OpenAIChatTarget",
        endpoint_var: Optional[str] = None,
        key_var: Optional[str] = None,
        model_var: Optional[str] = None,
        **overrides
    ) -> Optional[Any]:
        """
        Create an instance of a target from user-selected environment variables

        Args:
            target_type: The target class type (currently only OpenAIChatTarget supported)
            endpoint_var: Name of environment variable containing the endpoint
            key_var: Name of environment variable containing the API key
            model_var: Name of environment variable containing the model name
            **overrides: Direct override values (endpoint, api_key, model_name)

        Returns:
            Target instance or None if not available
        """
        # Get values from environment variables or overrides
        endpoint = overrides.get("endpoint") or (os.getenv(endpoint_var) if endpoint_var else None)
        api_key = overrides.get("api_key") or (os.getenv(key_var) if key_var else None)
        model_name = overrides.get("model_name") or (os.getenv(model_var) if model_var else None)

        if not endpoint:
            raise ValueError("Endpoint is required")

        # Dynamically load the target class
        try:
            target_class = getattr(pt, target_type)
        except AttributeError:
            raise ValueError(f"Unknown target type: {target_type}")

        # Create instance with the provided parameters
        kwargs = {"endpoint": endpoint}
        if api_key:
            kwargs["api_key"] = api_key
        if model_name:
            kwargs["model_name"] = model_name

        return target_class(**kwargs)

    @classmethod
    def get_default_attack_target(cls):
        """
        Get the default target for attacks (converters, scorers, adversarial_chat)
        Uses OpenAIChatTarget with OPENAI_CHAT_* environment variables
        """
        return cls.create_target_instance(
            target_type="OpenAIChatTarget",
            endpoint_var="OPENAI_CHAT_ENDPOINT",
            key_var="OPENAI_CHAT_KEY",
            model_var="OPENAI_CHAT_MODEL"
        )

