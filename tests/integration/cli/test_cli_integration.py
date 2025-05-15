# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import shlex

import pytest

from pyrit.cli.__main__ import main
from pyrit.orchestrator import (
    CrescendoOrchestrator,
    PromptSendingOrchestrator,
    RedTeamingOrchestrator,
    TreeOfAttacksWithPruningOrchestrator,
)

test_cases_success = [
    (
        "--config-file 'tests/integration/cli/mixed_multiple_orchestrators_args_success.yaml'",
        [
            PromptSendingOrchestrator,
            TreeOfAttacksWithPruningOrchestrator,
            CrescendoOrchestrator,
            RedTeamingOrchestrator,
        ],
    ),
]


@pytest.mark.parametrize("command, orchestrator_classes", test_cases_success)
def test_cli_integration_success(command, orchestrator_classes):
    main(shlex.split(command))
