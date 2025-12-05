# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Scenario Basic Dataset Loader.

If you don't have a database already, this can enable you to run all scenarios using
the pre-defined datasets in PyRIT. These are meant as a starting point only.
"""

import asyncio
import os
from typing import List

from pyrit.common.apply_defaults import set_default_value
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.scenario import Scenario
from pyrit.setup.initializers.pyrit_initializer import PyRITInitializer
from pyrit.datasets import SeedDatasetProvider


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
            "This configuration uses the DatasetLoader to load default datasets into memory. "
            "This will enable all scenarios to run. Datasets can be customized in memory."
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
        """Async helper to load datasets."""
        required_datasets: List[str] = [
            # Add dataset names here as needed
        ]

        if required_datasets:
            datasets = await SeedDatasetProvider.fetch_datasets_async(
                dataset_names=required_datasets,
            )
            # TODO: Store datasets or make them available to scenarios
