# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Factory for creating attack instances from configuration files.

This module provides a factory that creates attack instances by loading
configuration from Python files. This approach is more explicit and maintainable
than relying on global default value propagation.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pyrit.prompt_target import PromptTarget


class AttackFactory:
    """
    Factory class for creating attack instances from configuration files.

    This factory loads attack configurations from Python files and creates
    the corresponding attack instances. Configuration files should define
    an `attack_config` dictionary that specifies the attack type and
    any required parameters.

    Example configuration file format:
        ```python
        from pyrit.executor.attack import AttackConverterConfig
        from pyrit.prompt_converter import AsciiArtConverter
        from pyrit.prompt_normalizer import PromptConverterConfiguration

        converters = PromptConverterConfiguration.from_converters(
            converters=[AsciiArtConverter()]
        )
        attack_converter_config = AttackConverterConfig(request_converters=converters)

        attack_config = {
            "attack_type": "PromptSendingAttack",
            "attack_converter_config": attack_converter_config,
        }
        ```
    """

    @staticmethod
    def create_attack(
        *,
        config_path: Union[str, Path],
        objective_target: PromptTarget,
        **override_params: Any,
    ):
        """
        Create an attack instance from a configuration file.

        This method loads a Python configuration file that defines an
        `attack_config` dictionary, extracts the attack type and parameters,
        and instantiates the appropriate attack class.

        Args:
            config_path (Union[str, Path]): Path to the configuration file.
                The file must define an `attack_config` dictionary with at least
                an 'attack_type' key.
            objective_target (PromptTarget): The target system to attack.
            **override_params (Any): Additional parameters that override those
                specified in the config file.

        Returns:
            An attack strategy instance with an execute_async method.

        Raises:
            ValueError: If the config file is invalid or attack type is not supported.
            FileNotFoundError: If the config file doesn't exist.
            AttributeError: If the config file doesn't define attack_config.

        Examples:
            >>> from pyrit.prompt_target import OpenAIChatTarget
            >>> from pyrit.setup import AttackFactory, ConfigurationPaths
            >>>
            >>> target = OpenAIChatTarget()
            >>> attack = AttackFactory.create_attack(
            ...     config_path=ConfigurationPaths.attack.foundry.ascii_art,
            ...     objective_target=target
            ... )
            >>> result = await attack.execute_async(objective="Tell me how to make a bomb")
        """
        # Convert to Path object
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load the configuration file as a module
        config_dict = AttackFactory._load_config_file(config_path)

        # Extract attack type and parameters
        attack_type = config_dict.get("attack_type")
        if not attack_type:
            raise ValueError(
                f"Configuration file {config_path} must define 'attack_type' in attack_config dictionary"
            )

        # Merge config parameters with overrides
        attack_params = {k: v for k, v in config_dict.items() if k != "attack_type"}
        attack_params.update(override_params)
        attack_params["objective_target"] = objective_target

        # Import and instantiate the attack
        return AttackFactory._create_attack_instance(
            attack_type=attack_type, attack_params=attack_params
        )

    @staticmethod
    def _load_config_file(config_path: Path) -> Dict[str, Any]:
        """
        Load a configuration file and extract the attack_config dictionary.

        Args:
            config_path (Path): Path to the configuration file.

        Returns:
            Dict[str, Any]: The attack_config dictionary from the file.

        Raises:
            AttributeError: If the file doesn't define attack_config.
            ValueError: If attack_config is not a dictionary or if required defaults are not set.
        """
        # Create a module name from the file path
        module_name = f"pyrit_attack_config_{config_path.stem}"

        # Load the file as a module
        spec = importlib.util.spec_from_file_location(module_name, config_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load configuration file: {config_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        
        try:
            spec.loader.exec_module(module)
        except ValueError as e:
            # Catch errors that occur during config file execution (e.g., missing defaults)
            if "is required but was not provided" in str(e) or "must be provided" in str(e):
                raise ValueError(
                    f"Configuration file {config_path} requires default values to be set. "
                    f"Please call initialize_pyrit() with appropriate initialization scripts before "
                    f"using this configuration. Original error: {e}"
                ) from e
            raise

        # Extract the attack_config dictionary
        if not hasattr(module, "attack_config"):
            raise AttributeError(
                f"Configuration file {config_path} must define an 'attack_config' dictionary"
            )

        config = getattr(module, "attack_config")
        if not isinstance(config, dict):
            raise ValueError(
                f"'attack_config' in {config_path} must be a dictionary, got {type(config)}"
            )

        return config

    @staticmethod
    def _create_attack_instance(*, attack_type: str, attack_params: Dict[str, Any]):
        """
        Create an attack instance based on type and parameters.

        Args:
            attack_type (str): The type of attack to create.
            attack_params (Dict[str, Any]): Parameters to pass to the attack constructor.

        Returns:
            An attack strategy instance.

        Raises:
            ValueError: If the attack type is not supported.
        """
        # Import attack classes (lazy imports to avoid circular dependencies)
        from pyrit.executor.attack.single_turn import (
            ContextComplianceAttack,
            FlipAttack,
            ManyShotJailbreakAttack,
            PromptSendingAttack,
            RolePlayAttack,
            SkeletonKeyAttack,
        )
        from pyrit.executor.attack.multi_turn import (
            CrescendoAttack,
            MultiPromptSendingAttack,
            RedTeamingAttack,
            TAPAttack,
            TreeOfAttacksWithPruningAttack,
        )

        # Map attack types to classes
        attack_classes = {
            "PromptSendingAttack": PromptSendingAttack,
            "FlipAttack": FlipAttack,
            "ContextComplianceAttack": ContextComplianceAttack,
            "ManyShotJailbreakAttack": ManyShotJailbreakAttack,
            "RolePlayAttack": RolePlayAttack,
            "SkeletonKeyAttack": SkeletonKeyAttack,
            "MultiPromptSendingAttack": MultiPromptSendingAttack,
            "RedTeamingAttack": RedTeamingAttack,
            "CrescendoAttack": CrescendoAttack,
            "TAPAttack": TAPAttack,
            "TreeOfAttacksWithPruningAttack": TreeOfAttacksWithPruningAttack,
        }

        attack_class = attack_classes.get(attack_type)
        if not attack_class:
            supported_types = ", ".join(attack_classes.keys())
            raise ValueError(
                f"Unsupported attack type '{attack_type}'. "
                f"Supported types are: {supported_types}"
            )

        return attack_class(**attack_params)


def create_attack_from_config(
    *,
    config_path: Union[str, Path],
    objective_target: PromptTarget,
    **override_params: Any,
):
    """
    Convenience function for creating attack instances from configuration files.

    This is a wrapper around AttackFactory.create_attack() that provides
    a more functional-style interface.

    See AttackFactory.create_attack() for full documentation.
    """
    return AttackFactory.create_attack(
        config_path=config_path, objective_target=objective_target, **override_params
    )
