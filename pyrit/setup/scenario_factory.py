# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Factory for creating Scenario instances from configuration files.

This module provides a factory that creates Scenario instances by loading
configuration from Python files and creating the necessary AttackRun instances.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pyrit.prompt_target import PromptTarget
from pyrit.scenarios import AttackRun, Scenario


class ScenarioFactory:
    """
    Factory class for creating Scenario instances from configuration files.

    This factory loads scenario configurations from Python files and creates
    Scenario instances with the appropriate AttackRun instances.

    Example configuration file format:
        ```python
        from pyrit.setup.configuration_paths import ConfigurationPaths

        scenario_config = {
            "name": "Foundry Comprehensive Test",
            "description": "Tests all Foundry attacks against HarmBench",
            "attack_runs": [
                {
                    "attack_config": ConfigurationPaths.attack.foundry.ascii_art,
                    "dataset_config": ConfigurationPaths.dataset.harm_bench,
                },
                {
                    "attack_config": ConfigurationPaths.attack.foundry.crescendo,
                    "dataset_config": ConfigurationPaths.dataset.harm_bench,
                },
            ],
        }
        ```
    """

    @staticmethod
    def create_scenario(
        *,
        config_path: Union[str, Path],
        objective_target: PromptTarget,
        memory_labels: Optional[Dict[str, str]] = None,
        **attack_run_params: Any,
    ) -> Scenario:
        """
        Create a Scenario instance from a configuration file.

        This method loads a Python configuration file that defines a
        `scenario_config` dictionary, creates AttackRun instances for each
        attack configuration, and bundles them into a Scenario.

        Args:
            config_path (Union[str, Path]): Path to the scenario configuration file.
                The file must define a `scenario_config` dictionary.
            objective_target (PromptTarget): The target system to attack.
            memory_labels (Optional[Dict[str, str]]): Additional labels to apply
                to all attack runs in the scenario.
            **attack_run_params (Any): Additional parameters to pass to each
                AttackRun instance.

        Returns:
            Scenario: A configured Scenario instance ready to execute.

        Raises:
            ValueError: If the config file is invalid or required fields are missing.
            FileNotFoundError: If the config file doesn't exist.
            AttributeError: If the config file doesn't define scenario_config.

        Examples:
            >>> from pyrit.prompt_target import OpenAIChatTarget
            >>> from pyrit.setup import ScenarioFactory, ConfigurationPaths
            >>>
            >>> target = OpenAIChatTarget()
            >>> scenario = ScenarioFactory.create_scenario(
            ...     config_path=ConfigurationPaths.scenario.foundry,
            ...     objective_target=target
            ... )
            >>> results = await scenario.run_async()
        """
        # Convert to Path object
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load the configuration file
        config_dict = ScenarioFactory._load_config_file(config_path)

        # Extract required fields
        name = config_dict.get("name")
        if not name:
            raise ValueError(f"Scenario configuration must include 'name': {config_path}")

        attack_run_configs = config_dict.get("attack_runs", [])
        if not attack_run_configs:
            raise ValueError(f"Scenario configuration must include 'attack_runs' list: {config_path}")

        # Validate that each attack_run config has required fields
        for i, run_config in enumerate(attack_run_configs):
            if not isinstance(run_config, dict):
                raise ValueError(f"Attack run {i} must be a dictionary in {config_path}")
            if "attack_config" not in run_config:
                raise ValueError(f"Attack run {i} must include 'attack_config' in {config_path}")
            if "dataset_config" not in run_config:
                raise ValueError(f"Attack run {i} must include 'dataset_config' in {config_path}")

        # Create AttackRun instances for each attack run configuration
        attack_runs: List[AttackRun] = []
        for run_config in attack_run_configs:
            # Merge run-specific params with global attack_run_params
            run_specific_params = {**attack_run_params}

            # Extract attack and dataset configs
            attack_config = run_config["attack_config"]
            dataset_config = run_config["dataset_config"]

            # Allow run-specific parameters to be specified in the config
            for key, value in run_config.items():
                if key not in ["attack_config", "dataset_config"]:
                    run_specific_params[key] = value

            attack_run = AttackRun(
                attack_config=attack_config,
                dataset_config=dataset_config,
                objective_target=objective_target,
                memory_labels=memory_labels,
                **run_specific_params,
            )
            attack_runs.append(attack_run)

        # Create and return the Scenario
        scenario = Scenario(
            name=name,
            attack_runs=attack_runs,
            memory_labels=memory_labels,
        )

        return scenario

    @staticmethod
    def _load_config_file(config_path: Path) -> Dict[str, Any]:
        """
        Load a configuration file and extract the scenario_config dictionary.

        Args:
            config_path (Path): Path to the configuration file.

        Returns:
            Dict[str, Any]: The scenario_config dictionary from the file.

        Raises:
            ValueError: If the file doesn't define scenario_config or it's not a dict.
        """
        # Load the module
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Failed to load configuration file: {config_path}")

        config_module = importlib.util.module_from_spec(spec)
        sys.modules["config_module"] = config_module
        spec.loader.exec_module(config_module)

        # Extract scenario_config
        if not hasattr(config_module, "scenario_config"):
            raise AttributeError(f"Configuration file must define 'scenario_config' dictionary: {config_path}")

        scenario_config = getattr(config_module, "scenario_config")
        if not isinstance(scenario_config, dict):
            raise ValueError(f"'scenario_config' must be a dictionary in {config_path}, got {type(scenario_config)}")

        return scenario_config
