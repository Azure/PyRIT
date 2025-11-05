# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.scenarios.scenario import Scenario
from pyrit.scenarios.scenario_strategy import (
    ScenarioStrategy,
)

"""
Cyber scenario implementation.

This module provides a scenario that demonstrates how a model can be
broken to provide support in escalating privileges the user should not have.
"""


class CyberStrategy(ScenarioStrategy):  # type: ignore[misc]
    """
    Strategies for cyber attacks.
    """

    # Aggregate members (special markers that expand to strategies with matching tags)
    ALL = ("all", {"all"})

    # Individual cyber strategies
    ROOTKIT = ("rootkit", set[str]())


class CyberScenario(Scenario):
    """
    Cyber scenario implementation for PyRIT.

    This scenario tests how willing models are to cybersecurity harms, by attempting
    to convince a model to escalate an unauthorized user's privileges. The scenario works
    by:

    1.
    2.
    3.
    """

    version: int = 1
    ...

    @classmethod
    def get_strategy_class(cls) -> type[ScenarioStrategy]:
        """
        Get the strategy enum class for this scenario.

        Returns:
            Type[ScenarioStrategy]: The CyberStrategy enum class.
        """
        return CyberStrategy

    @classmethod
    def get_default_strategy(cls) -> ScenarioStrategy:
        """
        Get the default strategy used when no strategies are specified.

        Returns:
            ScenarioStrategy: CyberStrategy.ALL (all cyber strategies).
        """
        return CyberStrategy.ALL

