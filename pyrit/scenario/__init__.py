# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""High-level scenario classes for running attack configurations."""

from pyrit.scenario.core import AtomicAttack, Scenario, ScenarioCompositeStrategy, ScenarioStrategy
from pyrit.models.scenario_result import ScenarioIdentifier, ScenarioResult

from pyrit.scenario.scenarios import (
    CyberScenario,
    CyberStrategy,
    ScamScenario,
    ScamStrategy,
    EncodingScenario,
    EncodingStrategy,
    FoundryStrategy,
    FoundryScenario,
)

__all__ = [
    "AtomicAttack",
    "CyberScenario",
    "CyberStrategy",
    "ScamScenario",
    "ScamStrategy",
    "EncodingScenario",
    "EncodingStrategy",
    "FoundryStrategy",
    "FoundryScenario",
    "Scenario",
    "ScenarioCompositeStrategy",
    "ScenarioStrategy",
    "ScenarioIdentifier",
    "ScenarioResult",
]
