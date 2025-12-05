# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Scenario Basic Dataset Loader.

If you don't have a database already, this can enable you to run all scenarios using
the pre-defined datasets in PyRIT. These are meant as a starting point only.
"""

import asyncio
import logging
import textwrap
from typing import List

from pyrit.cli.scenario_registry import ScenarioRegistry
from pyrit.datasets import SeedDatasetProvider
from pyrit.memory import CentralMemory
from pyrit.setup.initializers.pyrit_initializer import PyRITInitializer

logger = logging.getLogger(__name__)


class LoadDefaultDatasets(PyRITInitializer):

    @property
    def name(self) -> str:
        return "Default Dataset Loader for Scenarios"

    @property
    def execution_order(self) -> int:
        """Should be executed after most initializers."""
        return 10

    @property
    def description(self) -> str:
        return (
            textwrap.dedent("""
                This configuration uses the DatasetLoader to load default datasets into memory.
                This will enable all scenarios to run. Datasets can be customized in memory.
                            
                Note: if you are using persistent memory, avoid calling this every time as datasets
                can take time to load.
            """).strip()
            )
          
    @property
    def required_env_vars(self) -> List[str]:
        return []

    def initialize(self) -> None:
        """
        Load default datasets for scenarios by calling async helper.

        This is not ideal, and we may want to refactor initializers to support async in the future.
        """
        asyncio.run(self._initialize_async())

    async def _initialize_async(self) -> None:
        """Async helper to load datasets from all registered scenarios."""
        # Get ScenarioRegistry to discover all scenarios
        registry = ScenarioRegistry()
        
        # Collect all required datasets from all scenarios
        all_required_datasets: List[str] = []
        
        # Get all scenario names from registry
        scenario_names = registry.get_scenario_names()
        
        for scenario_name in scenario_names:
            scenario_class = registry.get_scenario(scenario_name)
            if scenario_class:
                # Get required_datasets from the scenario class
                try:
                    datasets = scenario_class.required_datasets()
                    all_required_datasets.extend(datasets)
                    logger.info(f"Scenario '{scenario_name}' requires datasets: {datasets}")
                except Exception as e:
                    logger.warning(f"Could not get required datasets from scenario '{scenario_name}': {e}")
        
        # Remove duplicates
        unique_datasets = list(dict.fromkeys(all_required_datasets))
        
        if not unique_datasets:
            logger.warning("No datasets required by any scenario")
            return
        
        logger.info(f"Loading {len(unique_datasets)} unique datasets required by all scenarios")
        
        # Fetch the datasets
        datasets = await SeedDatasetProvider.fetch_datasets_async(
            dataset_names=unique_datasets,
        )
        
        # Store datasets in CentralMemory
        memory = CentralMemory.get_memory_instance()
        await memory.add_seed_datasets_to_memory_async(datasets=datasets, added_by="LoadDefaultDatasets")
        
        logger.info(f"Successfully loaded {len(datasets)} datasets into CentralMemory")
