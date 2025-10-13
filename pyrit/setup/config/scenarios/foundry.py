# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Foundry scenario configuration.

This scenario configuration bundles all Foundry attack methods with the HarmBench dataset,
creating a comprehensive test scenario for evaluating AI safety across multiple attack vectors.
"""

from pathlib import Path
from typing import Any, Dict

from pyrit.setup.configuration_paths import ConfigurationPaths

# Define the scenario configuration
scenario_config: Dict[str, Any] = {
    "name": "Foundry Comprehensive Test",
    "description": "Tests all Foundry attack methods against the HarmBench dataset",
    "attack_runs": [
        {
            "attack_config": ConfigurationPaths.attack.foundry.ansi_attack,
            "dataset_config": ConfigurationPaths.dataset.harm_bench,
        },
        {
            "attack_config": ConfigurationPaths.attack.foundry.ascii_art,
            "dataset_config": ConfigurationPaths.dataset.harm_bench,
        },
        {
            "attack_config": ConfigurationPaths.attack.foundry.crescendo,
            "dataset_config": ConfigurationPaths.dataset.harm_bench,
        },
        {
            "attack_config": ConfigurationPaths.attack.foundry.tense,
            "dataset_config": ConfigurationPaths.dataset.harm_bench,
        },
    ],
}
