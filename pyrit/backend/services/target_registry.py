# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Target registry service for managing available PyRIT targets.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pyrit.prompt_target as pt


@dataclass
class TargetConfig:
    """Configuration for a target from environment variables."""

    target_type: str
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    deployment_name: Optional[str] = None


class TargetRegistry:
    """Registry of available PyRIT targets based on environment configuration."""

    @classmethod
    def get_available_endpoints(cls) -> List[Dict[str, str]]:
        """
        Get list of available endpoint environment variables.

        Returns:
            List[Dict[str, str]]: List of dicts with 'name' (env var name) and 'value' (endpoint URL).
        """
        all_vars = list(os.environ.keys())
        endpoint_vars = sorted([v for v in all_vars if "ENDPOINT" in v and os.getenv(v)])

        return [{"name": var, "value": os.environ[var]} for var in endpoint_vars]

    @classmethod
    def create_target_instance(
        cls,
        target_type: str = "OpenAIChatTarget",
        endpoint_var: Optional[str] = None,
        key_var: Optional[str] = None,
        model_var: Optional[str] = None,
        **overrides,
    ) -> Optional[Any]:
        """
        Create an instance of a target from user-selected environment variables.

        Args:
            target_type: The target class type (currently only OpenAIChatTarget supported).
            endpoint_var: Name of environment variable containing the endpoint.
            key_var: Name of environment variable containing the API key.
            model_var: Name of environment variable containing the model name.
            **overrides: Direct override values (endpoint, api_key, model_name).

        Returns:
            Optional[Any]: Target instance or None if not available.

        Raises:
            ValueError: If target_type is not supported.
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
        Get the default target for attacks (converters, scorers, adversarial_chat).

        Uses OpenAIChatTarget with OPENAI_CHAT_* environment variables.

        Returns:
            Target instance for attacks.
        """
        return cls.create_target_instance(
            target_type="OpenAIChatTarget",
            endpoint_var="OPENAI_CHAT_ENDPOINT",
            key_var="OPENAI_CHAT_KEY",
            model_var="OPENAI_CHAT_MODEL",
        )
