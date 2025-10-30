# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Unit tests for scenario strategy parameter consistency.

This test file validates that all built-in scenarios expose the scenario_strategies
parameter in their __init__ method, which is required for CLI integration.
"""

import inspect

import pytest

from pyrit.cli.scenario_registry import ScenarioRegistry


class TestScenarioStrategiesParameter:
    """Test that all built-in scenarios expose the scenario_strategies parameter."""

    @pytest.fixture
    def scenario_registry(self) -> ScenarioRegistry:
        """Create a ScenarioRegistry with built-in scenarios."""
        registry = ScenarioRegistry()
        return registry

    def test_all_scenarios_have_strategy_parameter(self, scenario_registry: ScenarioRegistry):
        """
        Test that all built-in scenarios have a 'scenario_strategies' parameter in __init__.

        This ensures consistency across all scenarios and enables users to specify
        which strategies to run from the CLI or in code.
        """
        scenario_names = scenario_registry.get_scenario_names()
        assert len(scenario_names) > 0, "No scenarios found in registry"

        for scenario_name in scenario_names:
            scenario_class = scenario_registry.get_scenario(scenario_name)
            assert scenario_class is not None, f"Could not load scenario: {scenario_name}"

            # Get the __init__ signature
            init_signature = inspect.signature(scenario_class.__init__)
            parameters = init_signature.parameters

            # Check that scenario_strategies parameter exists
            assert (
                "scenario_strategies" in parameters
            ), f"{scenario_name} missing 'scenario_strategies' parameter in __init__"

            # Verify it's a keyword-only parameter (comes after *)
            param = parameters["scenario_strategies"]
            assert param.kind in (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.VAR_KEYWORD,
            ), f"{scenario_name} 'scenario_strategies' should be keyword-only"

            # Check that it has a default value (not Parameter.empty)
            assert (
                param.default is not inspect.Parameter.empty
            ), f"{scenario_name} 'scenario_strategies' parameter should have a default value"

            # Verify the default is None or a list
            assert param.default is None or isinstance(
                param.default, list
            ), f"{scenario_name} 'scenario_strategies' default should be None or a list"

            # If it's a list, verify it's not empty
            if isinstance(param.default, list):
                assert len(param.default) > 0, f"{scenario_name} 'scenario_strategies' default list should not be empty"
