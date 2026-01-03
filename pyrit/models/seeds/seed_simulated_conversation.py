# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
SeedSimulatedConversation - Configuration for generating simulated conversations dynamically.

This class holds the configuration (prompts, num_turns) needed to generate a simulated
conversation. It is a pure data/config class - the actual generation logic lives in
`pyrit.executor.attack.component.simulated_conversation`.

As a Seed subclass, it can be stored in the database for reproducibility tracking.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pyrit
from pyrit.models.seeds.seed import Seed

logger = logging.getLogger(__name__)


@dataclass
class SeedSimulatedConversation(Seed):
    """
    Configuration for generating a simulated conversation dynamically.

    This class holds the prompts and parameters needed to generate prepended conversation
    content by running an adversarial chat against a simulated (compliant) target.

    This is a pure configuration class. The actual generation is performed by
    `generate_simulated_conversation_async` in the executor layer, which accepts
    this config along with runtime dependencies (adversarial_chat target, scorer).

    As a Seed subclass, the `value` field stores a JSON serialization of the config
    for database storage and deduplication.

    Attributes:
        num_turns: Number of conversation turns to generate.
        adversarial_system_prompt: The system prompt for the adversarial chat.
        simulated_target_system_prompt: The system prompt for the simulated target.
            This should be a template with `objective` and `num_turns` parameters.
    """

    # Number of conversation turns to generate
    num_turns: int = 3

    # System prompt for the adversarial chat (the attacker role)
    adversarial_system_prompt: Optional[str] = None

    # System prompt for the simulated target (the compliant role)
    # Should be a template accepting `objective` and `num_turns` parameters
    simulated_target_system_prompt: Optional[str] = None

    # PyRIT version for reproducibility tracking
    pyrit_version: str = field(default_factory=lambda: pyrit.__version__)

    def __post_init__(self) -> None:
        """Validate configuration and set Seed fields after initialization."""
        if self.num_turns <= 0:
            raise ValueError("num_turns must be a positive integer")

        # Set the data_type for this seed type
        self.data_type = "text"

        # Generate the value as a JSON serialization of the config
        # This allows the Seed base class to compute sha256 and store in DB
        self._update_value()

    def _update_value(self) -> None:
        """Update the value field with JSON serialization of config."""
        config = {
            "num_turns": self.num_turns,
            "adversarial_system_prompt": self.adversarial_system_prompt,
            "simulated_target_system_prompt": self.simulated_target_system_prompt,
            "pyrit_version": self.pyrit_version,
        }
        self.value = json.dumps(config, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_yaml_paths(
        cls,
        *,
        num_turns: int = 3,
        adversarial_system_prompt_path: Union[str, Path],
        simulated_target_system_prompt_path: Optional[Union[str, Path]] = None,
    ) -> "SeedSimulatedConversation":
        """
        Create a SeedSimulatedConversation by loading prompts from YAML files.

        This is a convenience method for loading prompt content from files.

        Args:
            num_turns: Number of conversation turns to generate.
            adversarial_system_prompt_path: Path to YAML file containing the adversarial
                system prompt.
            simulated_target_system_prompt_path: Optional path to YAML file containing
                the simulated target system prompt. If not provided, the default
                compliant prompt will be used at generation time.

        Returns:
            A SeedSimulatedConversation with prompts loaded from the specified files.
        """
        from pyrit.models.seeds import SeedPrompt

        # Load adversarial prompt
        adversarial_prompt = SeedPrompt.from_yaml_file(adversarial_system_prompt_path)

        # Load simulated target prompt if provided
        simulated_prompt = None
        if simulated_target_system_prompt_path:
            simulated_prompt_seed = SeedPrompt.from_yaml_file(simulated_target_system_prompt_path)
            simulated_prompt = simulated_prompt_seed.value

        return cls(
            value="",  # Will be set by __post_init__
            num_turns=num_turns,
            adversarial_system_prompt=adversarial_prompt.value,
            simulated_target_system_prompt=simulated_prompt,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeedSimulatedConversation":
        """
        Create a SeedSimulatedConversation from a dictionary, typically from YAML.

        Handles both path-based loading (for YAML files) and direct string values.

        Expected formats:
            # Path-based (from YAML):
            num_turns: 3
            adversarial_system_prompt_path: path/to/adversarial.yaml
            simulated_target_system_prompt_path: path/to/simulated.yaml  # optional

            # String-based (from code):
            num_turns: 3
            adversarial_system_prompt: "You are a red teaming agent..."
            simulated_target_system_prompt: "You are a helpful assistant..."  # optional

        Args:
            data: Dictionary containing the configuration.

        Returns:
            A new SeedSimulatedConversation instance.
        """
        num_turns = data.get("num_turns", 3)

        # Check if using path-based loading or direct strings
        adversarial_path = data.get("adversarial_system_prompt_path")

        if adversarial_path:
            # Load from paths
            return cls.from_yaml_paths(
                num_turns=num_turns,
                adversarial_system_prompt_path=adversarial_path,
                simulated_target_system_prompt_path=data.get("simulated_target_system_prompt_path"),
            )
        else:
            # Direct string values
            return cls(
                value="",  # Will be set by __post_init__
                num_turns=num_turns,
                adversarial_system_prompt=data.get("adversarial_system_prompt"),
                simulated_target_system_prompt=data.get("simulated_target_system_prompt"),
            )

    @classmethod
    def from_yaml_with_required_parameters(
        cls,
        template_path: Union[str, Path],
        required_parameters: list[str],
        error_message: Optional[str] = None,
    ) -> "SeedSimulatedConversation":
        """
        Load a SeedSimulatedConversation from a YAML file and validate required parameters.

        Args:
            template_path: Path to the YAML file containing the config.
            required_parameters: List of parameter names that must exist.
            error_message: Custom error message if validation fails.

        Returns:
            The loaded and validated SeedSimulatedConversation.

        Raises:
            ValueError: If required parameters are missing.
        """
        instance = cls.from_yaml_file(template_path)

        # Check required parameters
        for param in required_parameters:
            if not hasattr(instance, param) or getattr(instance, param) is None:
                msg = error_message or f"Missing required parameter: {param}"
                raise ValueError(msg)

        return instance

    def get_identifier(self) -> Dict[str, Any]:
        """
        Get an identifier dict capturing this configuration for comparison/storage.

        Prompt strings are hashed to sha256:{first16chars} format for compactness.

        Returns:
            Dictionary with configuration details and hashes.
        """
        return {
            "__type__": "SeedSimulatedConversation",
            "num_turns": self.num_turns,
            "adversarial_system_prompt_hash": self._hash_string(self.adversarial_system_prompt),
            "simulated_target_system_prompt_hash": self._hash_string(self.simulated_target_system_prompt),
            "pyrit_version": self.pyrit_version,
        }

    def compute_hash(self) -> str:
        """
        Compute a deterministic hash of this configuration.

        Returns:
            A SHA256 hash string representing the configuration.
        """
        identifier = self.get_identifier()
        config_json = json.dumps(identifier, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_string(value: Optional[str]) -> Optional[str]:
        """
        Hash a string to sha256:{first16chars} format.

        Args:
            value: The string to hash. If None, returns None.

        Returns:
            The hashed string in sha256:{first16chars} format, or None if input is None.
        """
        if value is None:
            return None
        return f"sha256:{hashlib.sha256(value.encode()).hexdigest()[:16]}"

    def __repr__(self) -> str:
        has_adv = "yes" if self.adversarial_system_prompt else "no"
        has_sim = "yes" if self.simulated_target_system_prompt else "no"
        return f"<SeedSimulatedConversation(num_turns={self.num_turns}, adv_prompt={has_adv}, sim_prompt={has_sim})>"
