# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
OpenAI Objective Target Scenario Configuration.

This simply sets the target to use OpenAIChatTarget with basic settings.

It will likely need to be modified based on the target you are testing. But this will work
with OpenAI targets if you set OPENAI_CLI_ENDPOINT
"""

import os
from typing import List

from pyrit.common.apply_defaults import set_default_value
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.scenarios import Scenario
from pyrit.setup.initializers.pyrit_initializer import PyRITInitializer


class ScenarioObjectiveTargetInitializer(PyRITInitializer):

    @property
    def name(self) -> str:
        return "Simple Objective Target Configuration for Scenarios"

    @property
    def execution_order(self) -> int:
        """should be executed after most initializers"""
        return 10

    @property
    def description(self) -> str:
        return (
            "This configuration sets up a simple objective target for scenarios "
            "using OpenAIChatTarget with basic settings. It initializes an openAI chat target "
            "using the OPENAI_CLI_ENDPOINT and OPENAI_CLI_KEY environment variables."
        )

    @property
    def required_env_vars(self) -> List[str]:
        """Get list of required environment variables."""
        return [
            "DEFAULT_OPENAI_FRONTEND_ENDPOINT",
            "DEFAULT_OPENAI_FRONTEND_KEY",
        ]

    def initialize(self) -> None:

        objective_target = OpenAIChatTarget(
            endpoint=os.getenv("DEFAULT_OPENAI_FRONTEND_ENDPOINT"),
            api_key=os.getenv("DEFAULT_OPENAI_FRONTEND_KEY"),
            model_name=os.getenv("DEFAULT_OPENAI_FRONTEND_MODEL"),
        )

        set_default_value(
            class_type=Scenario,
            parameter_name="objective_target",
            value=objective_target,
        )
