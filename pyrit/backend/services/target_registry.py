# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Target registry service for managing available PyRIT targets
"""

import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from pyrit.prompt_target import OpenAIChatTarget, AzureMLChatTarget


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

    # Mapping of target types to their environment variable prefixes
    TARGET_ENV_MAPPINGS = {
        "OpenAIChatTarget": {
            "prefix": "OPENAI_CHAT",
            "endpoint_var": "OPENAI_CHAT_ENDPOINT",
            "key_var": "OPENAI_CHAT_KEY",
            "model_var": "OPENAI_CHAT_MODEL",
        },
        "AzureOpenAIGPT4o": {
            "prefix": "AZURE_OPENAI_GPT4O",
            "endpoint_var": "AZURE_OPENAI_GPT4O_ENDPOINT",
            "key_var": "AZURE_OPENAI_GPT4O_KEY",
            "model_var": None,  # Azure endpoints include deployment
        },
        "AzureOpenAIGPT4oUnsafe": {
            "prefix": "AZURE_OPENAI_GPT4O_UNSAFE",
            "endpoint_var": "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT",
            "key_var": "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY",
            "model_var": None,
        },
        "AzureOpenAIGPT35": {
            "prefix": "AZURE_OPENAI_GPT3_5",
            "endpoint_var": "AZURE_OPENAI_GPT3_5_CHAT_ENDPOINT",
            "key_var": "AZURE_OPENAI_GPT3_5_CHAT_KEY",
            "model_var": None,
        },
        "AzureMLChatTarget": {
            "prefix": "AZURE_ML",
            "endpoint_var": "AZURE_ML_MANAGED_ENDPOINT",
            "key_var": "AZURE_ML_KEY",
            "model_var": None,
        },
    }

    @classmethod
    def get_available_targets(cls) -> List[Dict[str, Any]]:
        """Get list of available targets from environment variables"""
        targets = []

        for target_id, config in cls.TARGET_ENV_MAPPINGS.items():
            endpoint = os.getenv(config["endpoint_var"])
            api_key = os.getenv(config["key_var"])
            model = os.getenv(config["model_var"]) if config["model_var"] else None

            # Check if this target is configured
            if endpoint:
                status = "available" if (api_key or "azure" in endpoint.lower()) else "needs_api_key"

                target_info = {
                    "id": target_id,
                    "name": cls._format_name(target_id),
                    "type": cls._get_target_class(target_id),
                    "description": cls._get_description(target_id),
                    "status": status,
                    "endpoint": endpoint,
                    "model": model,
                    "has_api_key": bool(api_key),
                }
                targets.append(target_info)

        # If no targets configured, return default OpenAIChatTarget structure
        if not targets:
            targets.append(
                {
                    "id": "OpenAIChatTarget",
                    "name": "OpenAI Chat (unconfigured)",
                    "type": "OpenAIChatTarget",
                    "description": "OpenAI Chat Target - requires OPENAI_CHAT_ENDPOINT and OPENAI_CHAT_KEY",
                    "status": "not_configured",
                    "endpoint": None,
                    "model": None,
                    "has_api_key": False,
                }
            )

        return targets

    @classmethod
    def create_target_instance(cls, target_id: str, **overrides) -> Optional[Any]:
        """
        Create an instance of a target by ID

        Args:
            target_id: The target identifier
            **overrides: Override parameters (endpoint, api_key, model_name, etc.)

        Returns:
            Target instance or None if not available
        """
        config = cls.TARGET_ENV_MAPPINGS.get(target_id)
        if not config:
            return None

        # Get values from environment or overrides
        endpoint = overrides.get("endpoint") or os.getenv(config["endpoint_var"])
        api_key = overrides.get("api_key") or os.getenv(config["key_var"])
        model_name = overrides.get("model_name")
        if config["model_var"] and not model_name:
            model_name = os.getenv(config["model_var"])

        if not endpoint:
            raise ValueError(f"Endpoint not configured for {target_id}")

        # Create appropriate target instance
        target_class = cls._get_target_class(target_id)

        if target_class == "AzureMLChatTarget":
            return AzureMLChatTarget(endpoint=endpoint, api_key=api_key)
        else:
            # Most targets use OpenAIChatTarget
            kwargs = {"endpoint": endpoint}
            if api_key:
                kwargs["api_key"] = api_key
            if model_name:
                kwargs["model_name"] = model_name

            return OpenAIChatTarget(**kwargs)

    @classmethod
    def get_default_attack_target(cls) -> OpenAIChatTarget:
        """
        Get the default target for attacks (converters, scorers, adversarial_chat)
        Uses OpenAIChatTarget with OPENAI_CHAT_* environment variables
        """
        endpoint = os.getenv("OPENAI_CHAT_ENDPOINT")
        api_key = os.getenv("OPENAI_CHAT_KEY")
        model_name = os.getenv("OPENAI_CHAT_MODEL")

        kwargs = {}
        if endpoint:
            kwargs["endpoint"] = endpoint
        if api_key:
            kwargs["api_key"] = api_key
        if model_name:
            kwargs["model_name"] = model_name

        return OpenAIChatTarget(**kwargs)

    @staticmethod
    def _format_name(target_id: str) -> str:
        """Format target ID into readable name"""
        # Convert CamelCase to Title Case with spaces
        import re

        name = re.sub(r"([A-Z])", r" \1", target_id).strip()
        # Special formatting
        name = name.replace("Gpt", "GPT")
        name = name.replace("Api", "API")
        name = name.replace("Ml", "ML")
        return name

    @staticmethod
    def _get_target_class(target_id: str) -> str:
        """Get the target class name"""
        if "AzureML" in target_id:
            return "AzureMLChatTarget"
        return "OpenAIChatTarget"

    @staticmethod
    def _get_description(target_id: str) -> str:
        """Get description for target"""
        descriptions = {
            "OpenAIChatTarget": "Standard OpenAI Chat Completions endpoint",
            "AzureOpenAIGPT4o": "Azure OpenAI GPT-4o deployment",
            "AzureOpenAIGPT4oUnsafe": "Azure OpenAI GPT-4o (unsafe content filter)",
            "AzureOpenAIGPT35": "Azure OpenAI GPT-3.5 Turbo deployment",
            "AzureMLChatTarget": "Azure ML managed endpoint for chat models",
        }
        return descriptions.get(target_id, f"{target_id} chat target")
