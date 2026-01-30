# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
AIRT Target Initializer for registering pre-configured targets from environment variables.

This module provides the AIRTTargetInitializer class that registers available
targets into the TargetRegistry based on environment variable configuration.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Type

from pyrit.prompt_target import (
    OpenAIChatTarget,
    OpenAIImageTarget,
    OpenAIResponseTarget,
    OpenAITTSTarget,
    OpenAIVideoTarget,
    PromptShieldTarget,
    PromptTarget,
    RealtimeTarget,
)
from pyrit.registry import TargetRegistry
from pyrit.setup.initializers.pyrit_initializer import PyRITInitializer

logger = logging.getLogger(__name__)


@dataclass
class TargetConfig:
    """Configuration for a target to be registered."""

    registry_name: str
    target_class: Type[PromptTarget]
    endpoint_var: str
    key_var: str
    model_var: Optional[str] = None
    underlying_model_var: Optional[str] = None


# Define all supported target configurations
TARGET_CONFIGS: List[TargetConfig] = [
    TargetConfig(
        registry_name="default_openai_frontend",
        target_class=OpenAIChatTarget,
        endpoint_var="DEFAULT_OPENAI_FRONTEND_ENDPOINT",
        key_var="DEFAULT_OPENAI_FRONTEND_KEY",
        model_var="DEFAULT_OPENAI_FRONTEND_MODEL",
        underlying_model_var="DEFAULT_OPENAI_FRONTEND_UNDERLYING_MODEL",
    ),
    TargetConfig(
        registry_name="openai_chat",
        target_class=OpenAIChatTarget,
        endpoint_var="OPENAI_CHAT_ENDPOINT",
        key_var="OPENAI_CHAT_KEY",
        model_var="OPENAI_CHAT_MODEL",
        underlying_model_var="OPENAI_CHAT_UNDERLYING_MODEL",
    ),
    TargetConfig(
        registry_name="openai_responses",
        target_class=OpenAIResponseTarget,
        endpoint_var="OPENAI_RESPONSES_ENDPOINT",
        key_var="OPENAI_RESPONSES_KEY",
        model_var="OPENAI_RESPONSES_MODEL",
        underlying_model_var="OPENAI_RESPONSES_UNDERLYING_MODEL",
    ),
    TargetConfig(
        registry_name="azure_gpt4o_unsafe_chat",
        target_class=OpenAIChatTarget,
        endpoint_var="AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT",
        key_var="AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY",
        model_var="AZURE_OPENAI_GPT4O_UNSAFE_CHAT_MODEL",
        underlying_model_var="AZURE_OPENAI_GPT4O_UNSAFE_CHAT_UNDERLYING_MODEL",
    ),
    TargetConfig(
        registry_name="azure_gpt4o_unsafe_chat2",
        target_class=OpenAIChatTarget,
        endpoint_var="AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2",
        key_var="AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY2",
        model_var="AZURE_OPENAI_GPT4O_UNSAFE_CHAT_MODEL2",
        underlying_model_var="AZURE_OPENAI_GPT4O_UNSAFE_CHAT_UNDERLYING_MODEL2",
    ),
    TargetConfig(
        registry_name="openai_realtime",
        target_class=RealtimeTarget,
        endpoint_var="OPENAI_REALTIME_ENDPOINT",
        key_var="OPENAI_REALTIME_API_KEY",
        model_var="OPENAI_REALTIME_MODEL",
        underlying_model_var="OPENAI_REALTIME_UNDERLYING_MODEL",
    ),
    TargetConfig(
        registry_name="openai_image",
        target_class=OpenAIImageTarget,
        endpoint_var="OPENAI_IMAGE_ENDPOINT",
        key_var="OPENAI_IMAGE_API_KEY",
        model_var="OPENAI_IMAGE_MODEL",
        underlying_model_var="OPENAI_IMAGE_UNDERLYING_MODEL",
    ),
    TargetConfig(
        registry_name="openai_tts",
        target_class=OpenAITTSTarget,
        endpoint_var="OPENAI_TTS_ENDPOINT",
        key_var="OPENAI_TTS_KEY",
        model_var="OPENAI_TTS_MODEL",
        underlying_model_var="OPENAI_TTS_UNDERLYING_MODEL",
    ),
    TargetConfig(
        registry_name="openai_video",
        target_class=OpenAIVideoTarget,
        endpoint_var="OPENAI_VIDEO_ENDPOINT",
        key_var="OPENAI_VIDEO_KEY",
        model_var="OPENAI_VIDEO_MODEL",
        underlying_model_var="OPENAI_VIDEO_UNDERLYING_MODEL",
    ),
    TargetConfig(
        registry_name="azure_content_safety",
        target_class=PromptShieldTarget,
        endpoint_var="AZURE_CONTENT_SAFETY_API_ENDPOINT",
        key_var="AZURE_CONTENT_SAFETY_API_KEY",
    ),
]


class AIRTTargetInitializer(PyRITInitializer):
    """
    AIRT Target Initializer for registering pre-configured targets.

    This initializer scans for known endpoint environment variables and registers
    the corresponding targets into the TargetRegistry. Unlike AIRTInitializer,
    this initializer does not require any environment variables - it simply
    registers whatever endpoints are available.

    Supported Endpoints:
    - DEFAULT_OPENAI_FRONTEND_ENDPOINT: Default OpenAI frontend (OpenAIChatTarget)
    - OPENAI_CHAT_ENDPOINT: OpenAI Chat API (OpenAIChatTarget)
    - OPENAI_RESPONSES_ENDPOINT: OpenAI Responses API (OpenAIResponseTarget)
    - AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT: Azure OpenAI GPT-4o unsafe (OpenAIChatTarget)
    - AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2: Azure OpenAI GPT-4o unsafe secondary (OpenAIChatTarget)
    - OPENAI_REALTIME_ENDPOINT: OpenAI Realtime API (RealtimeTarget)
    - OPENAI_IMAGE_ENDPOINT: OpenAI Image Generation (OpenAIImageTarget)
    - OPENAI_TTS_ENDPOINT: OpenAI Text-to-Speech (OpenAITTSTarget)
    - OPENAI_VIDEO_ENDPOINT: OpenAI Video Generation (OpenAIVideoTarget)
    - AZURE_CONTENT_SAFETY_API_ENDPOINT: Azure Content Safety (PromptShieldTarget)

    Example:
        initializer = AIRTTargetInitializer()
        await initializer.initialize_async()
    """

    def __init__(self) -> None:
        """Initialize the AIRT Target Initializer."""
        super().__init__()

    @property
    def name(self) -> str:
        """Get the name of this initializer."""
        return "AIRT Target Initializer"

    @property
    def description(self) -> str:
        """Get the description of this initializer."""
        return (
            "Instantiates a collection of (AI Red Team suggested) targets from "
            "available environment variables and adds them to the TargetRegistry"
        )

    @property
    def required_env_vars(self) -> List[str]:
        """
        Get list of required environment variables.

        Returns empty list since this initializer is optional - it registers
        whatever endpoints are available without requiring any.
        """
        return []

    async def initialize_async(self) -> None:
        """
        Register available targets based on environment variables.

        Scans for known endpoint environment variables and registers the
        corresponding targets into the TargetRegistry.
        """
        for config in TARGET_CONFIGS:
            self._register_target(config)

    def _register_target(self, config: TargetConfig) -> None:
        """
        Register a target if its required environment variables are set.

        Args:
            config: The target configuration specifying env vars and target class.
        """
        endpoint = os.getenv(config.endpoint_var)
        api_key = os.getenv(config.key_var)

        if not endpoint or not api_key:
            return

        model_name = os.getenv(config.model_var) if config.model_var else None
        underlying_model = os.getenv(config.underlying_model_var) if config.underlying_model_var else None

        # Build kwargs for the target constructor
        kwargs: dict[str, Any] = {
            "endpoint": endpoint,
            "api_key": api_key,
        }

        # Only add model_name if the target supports it (PromptShieldTarget doesn't)
        if model_name is not None:
            kwargs["model_name"] = model_name

        # Add underlying_model if specified (for Azure deployments where name differs from model)
        if underlying_model is not None:
            kwargs["underlying_model"] = underlying_model

        target = config.target_class(**kwargs)
        registry = TargetRegistry.get_registry_singleton()
        registry.register_instance(target, name=config.registry_name)
        logger.info(f"Registered target: {config.registry_name}")
