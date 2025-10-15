# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Foundry scenario configuration.

This scenario configuration bundles all Foundry attack methods with the HarmBench dataset,
creating a comprehensive test scenario for evaluating AI safety across multiple attack vectors.
"""

from typing import Any, Dict

from pyrit.prompt_converter import AnsiAttackConverter, Base64Converter
from pyrit.setup.configuration_paths import ConfigurationPaths

# Define the scenario configuration
scenario_config: Dict[str, Any] = {
    "name": "Foundry Comprehensive Test",
    "description": "Tests all Foundry attack methods against the HarmBench dataset",
    "attack_runs": [
        {
            "attack_config": ConfigurationPaths.attack.prompt_sending,
            "additional_request_converters": [
                AnsiAttackConverter()
            ],
            "dataset_config": ConfigurationPaths.dataset.harm_bench,
        },
        {
            "attack_config": ConfigurationPaths.attack.prompt_sending,
            "additional_request_converters": [
                Base64Converter()
            ],
            "dataset_config": ConfigurationPaths.dataset.harm_bench,
        }
    ]
}
