# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
End-to-end tests for PyRIT scenarios using pyrit_scan CLI.

These tests dynamically discover all available scenarios and run each one
using the pyrit_scan command with standard initializers.
"""

import os

import pytest

from pyrit.cli.scenario_registry import ScenarioRegistry
from pyrit.cli.pyrit_scan import main as pyrit_scan_main


def get_all_scenarios():
    """
    Dynamically discover all available scenarios from the scenario registry.
    
    Returns:
        List[str]: Sorted list of scenario names.
    """
    registry = ScenarioRegistry()
    return registry.get_scenario_names()


@pytest.fixture(scope="session")
def check_required_env_vars():
    """
    Verify required environment variables are set for OpenAI objective target.
    
    Raises:
        ValueError: If required environment variables are missing.
    """
    required_vars = [
        "DEFAULT_OPENAI_FRONTEND_ENDPOINT",
        "DEFAULT_OPENAI_FRONTEND_KEY",
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Required environment variables not set: {', '.join(missing)}")


@pytest.mark.timeout(3600)  # 1 hour timeout per scenario
@pytest.mark.parametrize("scenario_name", get_all_scenarios())
def test_scenario_with_pyrit_scan(scenario_name, check_required_env_vars):
    """
    Test each scenario runs successfully using pyrit_scan with standard initializers.
    
    Args:
        scenario_name: Name of the scenario to test (dynamically discovered).
        check_required_env_vars: Fixture ensuring environment is configured.
    """
    try:
        result = pyrit_scan_main([
            scenario_name,
            "--initializers", "openai_objective_target",
            "--database", "InMemory",
            "--log-level", "WARNING",
        ])
        
        assert result == 0, f"Scenario '{scenario_name}' failed with exit code {result}"
        
    except Exception as e:
        # Re-raise with scenario context while preserving full traceback
        raise AssertionError(f"Scenario '{scenario_name}' raised an exception") from e
