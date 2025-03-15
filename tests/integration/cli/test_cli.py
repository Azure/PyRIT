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
        "--config-file 'tests/unit/cli/mixed_multiple_orchestrators_args_success.yaml'",
        [
            PromptSendingOrchestrator,
            TreeOfAttacksWithPruningOrchestrator,
            CrescendoOrchestrator,
            RedTeamingOrchestrator,
        ],
        ["send_normalizer_requests_async", "run_attack_async", "run_attack_async", "run_attack_async"],
    ),
]


@pytest.mark.parametrize("command, orchestrator_classes, methods", test_cases_success)
def test_cli_integration_success(command, orchestrator_classes, methods):
    main(shlex.split(command))
