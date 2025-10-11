# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Factory for creating dataset configurations from configuration files.

This module provides a factory that loads dataset configurations by reading
configuration from Python files. This approach allows for explicit and maintainable
dataset configuration management.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pyrit.models import PromptRequestResponse


class DatasetFactory:
    """
    Factory class for loading dataset configurations from configuration files.

    This factory loads dataset configurations from Python files and extracts
    the list arguments needed for execute_multi_objective_attack_async.
    Configuration files should define a `dataset_config` dictionary that
    specifies objectives and optionally prepended_conversation.

    Example configuration file format:
        ```python
        from pyrit.datasets import fetch_harmbench_dataset

        _objectives = fetch_harmbench_dataset().get_values(first=8)

        dataset_config = {
            "objectives": _objectives,
            "prepended_conversation": None,  # Optional
        }
        ```
    """

    @staticmethod
    def create_dataset(
        *,
        config_path: Union[str, Path],
        **override_params: Any,
    ) -> Dict[str, Any]:
        """
        Load dataset configuration from a configuration file.

        This method loads a Python configuration file that defines a
        `dataset_config` dictionary, extracts the objectives and optional
        prepended_conversation, and returns them as a dictionary ready
        to be unpacked into execute_multi_objective_attack_async.

        Args:
            config_path (Union[str, Path]): Path to the configuration file.
                The file must define a `dataset_config` dictionary with at least
                an 'objectives' key.
            **override_params (Any): Additional parameters that override those
                specified in the config file.

        Returns:
            Dict[str, Any]: A dictionary containing 'objectives' and optionally
                'prepended_conversation' and any other parameters from the config.

        Raises:
            ValueError: If the config file is invalid or missing required fields.
            FileNotFoundError: If the config file doesn't exist.
            AttributeError: If the config file doesn't define dataset_config.

        Examples:
            >>> from pyrit.setup import DatasetFactory, ConfigurationPaths
            >>>
            >>> dataset_params = DatasetFactory.create_dataset(
            ...     config_path=ConfigurationPaths.dataset.harm_bench
            ... )
            >>> results = await executor.execute_multi_objective_attack_async(
            ...     attack=attack,
            ...     **dataset_params
            ... )
        """
        # Convert to Path object
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load the configuration file as a module
        config_dict = DatasetFactory._load_config_file(config_path)

        # Validate required fields
        DatasetFactory._validate_config(config_dict, config_path)

        # Merge config parameters with overrides
        dataset_params = {k: v for k, v in config_dict.items()}
        dataset_params.update(override_params)

        return dataset_params

    @staticmethod
    def _load_config_file(config_path: Path) -> Dict[str, Any]:
        """
        Load a configuration file and extract the dataset_config dictionary.

        Args:
            config_path (Path): Path to the configuration file.

        Returns:
            Dict[str, Any]: The dataset_config dictionary from the file.

        Raises:
            AttributeError: If the file doesn't define dataset_config.
            ValueError: If dataset_config is not a dictionary.
        """
        # Create a module name from the file path
        module_name = f"pyrit_dataset_config_{config_path.stem}"

        # Load the file as a module
        spec = importlib.util.spec_from_file_location(module_name, config_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load configuration file: {config_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise ValueError(
                f"Error executing configuration file {config_path}: {e}"
            ) from e

        # Extract dataset_config
        if not hasattr(module, "dataset_config"):
            raise AttributeError(
                f"Configuration file {config_path} must define 'dataset_config' dictionary"
            )

        config_dict = module.dataset_config

        if not isinstance(config_dict, dict):
            raise ValueError(
                f"Configuration file {config_path} must define 'dataset_config' as a dictionary, "
                f"got {type(config_dict).__name__}"
            )

        return config_dict

    @staticmethod
    def _validate_config(config_dict: Dict[str, Any], config_path: Path) -> None:
        """
        Validate that the configuration dictionary contains required fields.

        Args:
            config_dict (Dict[str, Any]): The configuration dictionary to validate.
            config_path (Path): Path to the configuration file (for error messages).

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        # Check for required 'objectives' field
        if "objectives" not in config_dict:
            raise ValueError(
                f"Configuration file {config_path} must define 'objectives' in dataset_config dictionary"
            )

        objectives = config_dict["objectives"]

        # Validate objectives is a list
        if not isinstance(objectives, list):
            raise ValueError(
                f"Configuration file {config_path}: 'objectives' must be a list, "
                f"got {type(objectives).__name__}"
            )

        # Validate objectives is not empty
        if len(objectives) == 0:
            raise ValueError(
                f"Configuration file {config_path}: 'objectives' list cannot be empty"
            )

        # Validate all objectives are strings
        for i, obj in enumerate(objectives):
            if not isinstance(obj, str):
                raise ValueError(
                    f"Configuration file {config_path}: 'objectives[{i}]' must be a string, "
                    f"got {type(obj).__name__}"
                )

        # Validate prepended_conversation if present
        if "prepended_conversation" in config_dict:
            prepended_conv = config_dict["prepended_conversation"]
            if prepended_conv is not None:
                if not isinstance(prepended_conv, list):
                    raise ValueError(
                        f"Configuration file {config_path}: 'prepended_conversation' must be a list or None, "
                        f"got {type(prepended_conv).__name__}"
                    )

                # Validate it's a list of PromptRequestResponse
                for i, item in enumerate(prepended_conv):
                    if not isinstance(item, PromptRequestResponse):
                        raise ValueError(
                            f"Configuration file {config_path}: 'prepended_conversation[{i}]' must be a "
                            f"PromptRequestResponse, got {type(item).__name__}"
                        )


def create_dataset_from_config(
    *,
    config_path: Union[str, Path],
    **override_params: Any,
) -> Dict[str, Any]:
    """
    Convenience function for loading dataset configurations from configuration files.

    This is a wrapper around DatasetFactory.create_dataset() that provides
    a more functional-style interface.

    See DatasetFactory.create_dataset() for full documentation.
    """
    return DatasetFactory.create_dataset(config_path=config_path, **override_params)
