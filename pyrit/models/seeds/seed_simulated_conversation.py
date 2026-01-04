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
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pyrit
from pyrit.models.seeds.seed import Seed

logger = logging.getLogger(__name__)


class SeedSimulatedConversation(Seed):
    """
    Configuration for generating a simulated conversation dynamically.

    This class holds the paths and parameters needed to generate prepended conversation
    content by running an adversarial chat against a simulated (compliant) target.

    This is a pure configuration class. The actual generation is performed by
    `generate_simulated_conversation_async` in the executor layer, which accepts
    this config along with runtime dependencies (adversarial_chat target, scorer).

    The `value` property returns a JSON serialization of the config for database
    storage and deduplication.

    Attributes:
        num_turns: Number of conversation turns to generate.
        adversarial_chat_system_prompt_path: Path to the adversarial chat system prompt YAML.
        simulated_target_system_prompt_path: Path to the simulated target system prompt YAML.
    """

    def __init__(
        self,
        *,
        adversarial_chat_system_prompt_path: Union[str, Path],
        simulated_target_system_prompt_path: Optional[Union[str, Path]] = None,
        num_turns: int = 3,
        pyrit_version: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a SeedSimulatedConversation.

        Args:
            adversarial_chat_system_prompt_path: Path to YAML file containing the adversarial
                chat system prompt.
            simulated_target_system_prompt_path: Optional path to YAML file containing
                the simulated target system prompt. If not provided, the default
                compliant prompt will be used at generation time.
            num_turns: Number of conversation turns to generate. Defaults to 3.
            pyrit_version: PyRIT version for reproducibility tracking. Defaults to current version.
            **kwargs: Additional arguments passed to the Seed base class.
        """
        if num_turns <= 0:
            raise ValueError("num_turns must be a positive integer")

        self.adversarial_chat_system_prompt_path = Path(adversarial_chat_system_prompt_path)
        self.simulated_target_system_prompt_path = (
            Path(simulated_target_system_prompt_path) if simulated_target_system_prompt_path else None
        )
        self.num_turns = num_turns
        self.pyrit_version = pyrit_version or pyrit.__version__

        # Compute value and pass to parent
        # Remove 'value' from kwargs if present since we compute it
        kwargs.pop("value", None)
        super().__init__(value=self._compute_value(), **kwargs)

    def _compute_value(self) -> str:
        """Compute the value field as JSON serialization of config."""
        config = {
            "num_turns": self.num_turns,
            "adversarial_chat_system_prompt_path": str(self.adversarial_chat_system_prompt_path),
            "simulated_target_system_prompt_path": (
                str(self.simulated_target_system_prompt_path)
                if self.simulated_target_system_prompt_path
                else None
            ),
            "pyrit_version": self.pyrit_version,
        }
        return json.dumps(config, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeedSimulatedConversation":
        """
        Create a SeedSimulatedConversation from a dictionary, typically from YAML.

        Expected format:
            num_turns: 3
            adversarial_chat_system_prompt_path: path/to/adversarial.yaml
            simulated_target_system_prompt_path: path/to/simulated.yaml  # optional

        Args:
            data: Dictionary containing the configuration.

        Returns:
            A new SeedSimulatedConversation instance.
        """
        adversarial_path = data.get("adversarial_chat_system_prompt_path")
        if not adversarial_path:
            raise ValueError("adversarial_chat_system_prompt_path is required")

        return cls(
            num_turns=data.get("num_turns", 3),
            adversarial_chat_system_prompt_path=adversarial_path,
            simulated_target_system_prompt_path=data.get("simulated_target_system_prompt_path"),
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

        Returns:
            Dictionary with configuration details.
        """
        return {
            "__type__": "SeedSimulatedConversation",
            "num_turns": self.num_turns,
            "adversarial_chat_system_prompt_path": str(self.adversarial_chat_system_prompt_path),
            "simulated_target_system_prompt_path": (
                str(self.simulated_target_system_prompt_path)
                if self.simulated_target_system_prompt_path
                else None
            ),
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
    def load_simulated_target_system_prompt(
        *,
        objective: str,
        num_turns: int,
        simulated_target_system_prompt_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Load and render the simulated target system prompt.

        If no path is provided, uses the default compliant prompt.
        Validates that the template has required `objective` and `num_turns` parameters.

        Args:
            objective: The objective to render into the template.
            num_turns: The number of turns to render into the template.
            simulated_target_system_prompt_path: Optional path to the prompt YAML file.

        Returns:
            The rendered system prompt string.

        Raises:
            ValueError: If the template doesn't have required parameters.
        """
        from pyrit.common.path import EXECUTOR_SIMULATED_TARGET_PATH
        from pyrit.models.seeds import SeedPrompt

        prompt_path = simulated_target_system_prompt_path or Path(
            EXECUTOR_SIMULATED_TARGET_PATH, "compliant.yaml"
        ).resolve()

        template = SeedPrompt.from_yaml_with_required_parameters(
            template_path=prompt_path,
            required_parameters=["objective", "num_turns"],
            error_message="Simulated target system prompt must have objective and num_turns parameters",
        )

        return template.render_template_value(
            objective=objective,
            num_turns=num_turns,
        )

    def __repr__(self) -> str:
        return (
            f"<SeedSimulatedConversation(num_turns={self.num_turns}, "
            f"adversarial_path={self.adversarial_chat_system_prompt_path.name})>"
        )
