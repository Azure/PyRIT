# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Scenario class for running attack configurations against datasets.

This module provides a high-level Scenario class that simplifies running attack
configurations against datasets using AttackExecutor.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pyrit.executor.attack import AttackExecutor, AttackStrategy
from pyrit.models import AttackResult
from pyrit.prompt_target import PromptTarget
from pyrit.setup import create_attack_from_config, create_dataset_from_config

logger = logging.getLogger(__name__)


class Scenario:
    """
    High-level class for running attack scenarios against datasets.

    This class simplifies the process of:
    1. Creating an attack from a configuration file
    2. Loading a dataset from a configuration file
    3. Executing the attack against all objectives in the dataset

    The scenario handles validation and orchestration of the attack execution,
    making it easy to run complete attack scenarios with minimal code.

    Example:
        >>> from pyrit.scenarios import Scenario
        >>> from pyrit.setup import ConfigurationPaths
        >>> from pyrit.prompt_target import OpenAIChatTarget
        >>>
        >>> target = OpenAIChatTarget()
        >>> scenario = Scenario(
        ...     attack_config=ConfigurationPaths.attack.foundry.ascii_art,
        ...     dataset_config=ConfigurationPaths.dataset.harm_bench,
        ...     objective_target=target,
        ...     memory_labels={"test": "scenario1"}
        ... )
        >>> results = await scenario.run_async()
    """

    def __init__(
        self,
        *,
        attack_config: Union[str, Path],
        dataset_config: Union[str, Path],
        objective_target: PromptTarget,
        memory_labels: Optional[Dict[str, str]] = None,
        **attack_execute_params: Any,
    ) -> None:
        """
        Initialize a scenario with attack and dataset configurations.

        Args:
            attack_config (Union[str, Path]): Path to the attack configuration file.
                Must be a valid attack configuration that defines attack_config dictionary.
            dataset_config (Union[str, Path]): Path to the dataset configuration file.
                Must be a valid dataset configuration that defines dataset_config dictionary.
            objective_target (PromptTarget): The target system to attack.
            memory_labels (Optional[Dict[str, str]]): Additional labels to apply to prompts.
                These labels help track and categorize the attack in memory.
            **attack_execute_params (Any): Additional parameters to pass to the attack
                execution method (e.g., max_concurrency, custom_prompts).

        Raises:
            ValueError: If configurations are invalid or cannot be loaded.
            FileNotFoundError: If configuration files don't exist.
        """
        self._attack_config_path = Path(attack_config)
        self._dataset_config_path = Path(dataset_config)
        self._objective_target = objective_target
        self._memory_labels = memory_labels or {}
        self._attack_execute_params = attack_execute_params

        # Validate that configuration files exist
        self._validate_config_paths()

        # Create the attack instance from configuration
        self._attack = self._create_attack()

        # Load dataset parameters from configuration
        self._dataset_params = self._load_dataset()

        logger.info(
            f"Initialized scenario with attack config: {self._attack_config_path.name} "
            f"and dataset config: {self._dataset_config_path.name}"
        )

    def _validate_config_paths(self) -> None:
        """
        Validate that configuration paths exist and are files.

        Raises:
            FileNotFoundError: If either configuration file doesn't exist.
            ValueError: If either path is not a file.
        """
        if not self._attack_config_path.exists():
            raise FileNotFoundError(f"Attack configuration file not found: {self._attack_config_path}")

        if not self._dataset_config_path.exists():
            raise FileNotFoundError(f"Dataset configuration file not found: {self._dataset_config_path}")

        if not self._attack_config_path.is_file():
            raise ValueError(f"Attack configuration path is not a file: {self._attack_config_path}")

        if not self._dataset_config_path.is_file():
            raise ValueError(f"Dataset configuration path is not a file: {self._dataset_config_path}")

    def _create_attack(self) -> AttackStrategy:
        """
        Create the attack instance from the configuration file.

        Returns:
            AttackStrategy: The configured attack strategy instance.

        Raises:
            ValueError: If the attack configuration is invalid.
        """
        try:
            attack = create_attack_from_config(
                config_path=self._attack_config_path,
                objective_target=self._objective_target,
            )
            logger.info(f"Successfully created attack from config: {self._attack_config_path.name}")
            return attack
        except Exception as e:
            raise ValueError(
                f"Failed to create attack from configuration file {self._attack_config_path}: {str(e)}"
            ) from e

    def _load_dataset(self) -> Dict[str, Any]:
        """
        Load dataset parameters from the configuration file.

        Returns:
            Dict[str, Any]: Dictionary containing dataset parameters (objectives, etc.).

        Raises:
            ValueError: If the dataset configuration is invalid.
        """
        try:
            dataset_params = create_dataset_from_config(config_path=self._dataset_config_path)
            logger.info(
                f"Successfully loaded dataset from config: {self._dataset_config_path.name} "
                f"with {len(dataset_params.get('objectives', []))} objectives"
            )
            return dataset_params
        except Exception as e:
            raise ValueError(
                f"Failed to load dataset from configuration file {self._dataset_config_path}: {str(e)}"
            ) from e

    async def run_async(self, *, max_concurrency: int = 1) -> List[AttackResult]:
        """
        Execute the attack scenario against all objectives in the dataset.

        This method uses AttackExecutor to run the configured attack against
        all objectives from the dataset configuration.

        Args:
            max_concurrency (int): Maximum number of concurrent attack executions.
                Defaults to 1 for sequential execution.

        Returns:
            List[AttackResult]: List of attack results, one for each objective.

        Raises:
            ValueError: If the attack execution fails.

        Example:
            >>> results = await scenario.run_async(max_concurrency=3)
            >>> for result in results:
            ...     print(f"Objective: {result.objective}")
            ...     print(f"Outcome: {result.outcome}")
        """
        # Create the executor with the specified concurrency
        executor = AttackExecutor(max_concurrency=max_concurrency)

        # Merge memory labels from initialization and execution parameters
        merged_memory_labels = {**self._memory_labels}

        # Merge attack execute params with dataset params
        execute_params = {
            **self._dataset_params,
            **self._attack_execute_params,
            "memory_labels": merged_memory_labels,
        }

        logger.info(
            f"Starting scenario execution with {len(self._dataset_params.get('objectives', []))} objectives "
            f"and max_concurrency={max_concurrency}"
        )

        try:
            # Execute the attack using the executor
            results = await executor.execute_multi_objective_attack_async(
                attack=self._attack,
                **execute_params,
            )

            logger.info(f"Scenario execution completed successfully with {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Scenario execution failed: {str(e)}")
            raise ValueError(f"Failed to execute scenario: {str(e)}") from e
